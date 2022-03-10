import numpy as np
import pandas as pd

from torch import nn, cuda
from collections import OrderedDict
from functools import reduce
from hyperopt import tpe, atpe, mix, rand, anneal, pyll, STATUS_OK
from hyperopt.base import miscs_update_idxs_vals
from sklearn.model_selection import StratifiedKFold, train_test_split

from utils.data_utils import create_data_loader
from utils.constants import *
from utils.in_out import print_model_config, colored, Colors


def prepare_folds(data, task_label, opt_config, seed=None):
    if opt_config['opt_strategy'] == 'cross_validation':
        skf = StratifiedKFold(n_splits=opt_config['folds'], shuffle=True, random_state=seed)
        return list(skf.split(data['text'].values, data[task_label].values))

    raise Exception(f'Optimization strategy {opt_config["opt_strategy"]} is not within supported strategies!')


def suggest(new_ids, domain, trials, seed, nbMaxSucessiveFailures=1000):
    # Build a hash set for previous trials
    hashset = set([hash(frozenset([(key, value[0]) if len(value) > 0 else ((key, None))
                                   for key, value in trial['misc']['vals'].items()])) for trial in trials.trials])

    rng = np.random.RandomState(seed)
    rval = []
    for _, new_id in enumerate(new_ids):
        newSample = False
        nbSucessiveFailures = 0
        while not newSample:
            # -- sample new specs, idxs, vals
            idxs, vals = pyll.rec_eval(
                domain.s_idxs_vals,
                memo={
                    domain.s_new_ids: [new_id],
                    domain.s_rng: rng,
                })
            new_result = domain.new_result()
            new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
            miscs_update_idxs_vals([new_misc], idxs, vals)

            # Compare with previous hashes
            h = hash(frozenset([(key, value[0]) if len(value) > 0 else (
                (key, None)) for key, value in vals.items()]))
            if h not in hashset:
                newSample = True
            else:
                # Duplicated sample, ignore
                nbSucessiveFailures += 1

            if nbSucessiveFailures > nbMaxSucessiveFailures:
                # No more samples to produce
                return []

        rval.extend(trials.new_trial_docs([new_id],
                                          [None], [new_result], [new_misc]))
    return rval


class ModelCoordinator:
    def __init__(self, data, data_folds, model_space, model_class, opt_config, task, evaluator, checkpoint):
        self.data = data
        self.data_folds = data_folds
        self.model_space = model_space
        self.model_class = model_class
        self.opt_config = opt_config
        self.task = task
        self.evaluator = evaluator
        self.checkpoint = checkpoint
        self.hyper_opt_algs = {
            "tpe": tpe.suggest,
            "atpe": atpe.suggest,
            "mix": mix.suggest,
            "rand": rand.suggest,
            "anneal": anneal.suggest,
            "grid": suggest
        }

    def objective(self, args):
        # Configure model parameters
        model_params = self.model_space.copy()
        model_params.update(args)
        model_params['task_name'] = f"{self.task['dataset']}_{self.task['name']}"
        model_params['evaluator'] = self.evaluator
        model_params['n_classes'] = 2
        if self.task['type'] == CAT_TYPE:
            model_params['loss_fn'] = nn.NLLLoss()
            model_params['n_classes'] = len(self.task['map'].keys())
        elif self.task['type'] == ORD_TYPE:
            model_params['loss_fn'] = nn.MSELoss()
        elif self.task['type'] == BIN_TYPE:
            model_params['loss_fn'] = nn.BCEWithLogitsLoss

        print_model_config(self.model_class.__name__, model_params)
        model = self.model_class(**model_params)
        if self.checkpoint is not None:
            model.load(name=self.checkpoint)

        results = []

        for fold, (trn_idxs, tst_idxs) in enumerate(self.data_folds):
            print(colored(f'Fold {fold + 1} / {len(self.data_folds)}...', Colors.PURPLE))
            data_loaders = self.prepare_data(self.data, (trn_idxs, tst_idxs), model.tokenizer,
                                             model_params['batch_size'])
            results.append(model.fit(data_loaders['trn'], val_dataloader=data_loaders['tst']))
            model.reset(model_params['n_classes'])

        trn_results, val_results, max_mem_usage = zip(*results)
        trn_results = pd.concat(trn_results, axis=1).T.mean()
        val_results = pd.concat(val_results, axis=1).T.mean()

        print(f"{colored('Cross-Validation Loss: ', Colors.GREEN)} {val_results['loss']:.4f}")

        return {
            'loss': val_results['loss'],
            'status': STATUS_OK,
            'params': model.state_dict(),
            'trn_results': trn_results,
            'val_results': val_results,
            'max_mem_usage': max(max_mem_usage),
            'name': model.name
        }

    def prepare_data(self, data, trn_tst_idxs, tokenizer, batch_size):
        data_loaders = {}
        for i, phase in enumerate(['trn', 'tst']):
            data_loaders[phase] = create_data_loader(data.iloc[trn_tst_idxs[i]], tokenizer, batch_size,
                                                     cols={'input': 'text', 'target': self.task['name']})
        return data_loaders

    def get_hyperopt_params(self):
        _space = OrderedDict(self.model_space)
        _estimated_evals = reduce(lambda x, y: x * y, [len(param.pos_args) - 1
                                                       for _, param in _space.items() if hasattr(param, 'pos_args')], 1)
        _max_evals = self.opt_config.get('hyper_max_evals', _estimated_evals)
        _opt_alg = self.hyper_opt_algs[self.opt_config.get('hyper_opt_alg', 'grid')]
        return _space, _opt_alg, _max_evals

    def train_single(self, best_params, save=True):
        model_params = self.model_space.copy()
        model_params.update(best_params)
        print_model_config(self.model_class.__name__, model_params)

        model = self.model_class(**model_params)
        # Small train-test split for Early Stopping
        trn_data, val_data = train_test_split(self.data, test_size=0.1, random_state=SEED)
        trn_dl = create_data_loader(trn_data, model.tokenizer, model_params['batch_size'],
                                         cols={'input': 'text', 'target': self.task['name']})
        val_dl = create_data_loader(val_data, model.tokenizer, model_params['batch_size'],
                                    cols={'input': 'text', 'target': self.task['name']})
        model.fit(trn_dl, val_dataloader=val_dl, save=save)

        return model

    @staticmethod
    def test_single(data, task, model=None, model_class=None, model_params=None, checkpoint=None):
        if checkpoint is not None:
            model = model_class(**model_params)
            model.load(checkpoint)
        elif model is None:
            raise Exception('Neither model nor checkpoint were provided to perform the single test!')

        dl = create_data_loader(data, model.tokenizer, model_params['batch_size'],
                                cols={'input': 'text', 'target': task['name']})

        model.evaluate(dl)
