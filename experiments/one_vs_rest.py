import importlib
import torch

from sklearn.model_selection import ParameterGrid, ParameterSampler
from hyperopt import Trials, fmin
from torch import cuda

from models.optimization import ModelCoordinator, prepare_folds
from utils import preprocess
from utils.data_utils import process_task
from utils.evaluation import Evaluator
from utils.in_out import *
from utils.constants import *
from utils.experiment_utils import *

_name = 'One-vs-Rest'


def get_best_model(data, task, opt_config, evaluator, model_class, model_config, checkpoint=None):
    folds = prepare_folds(data,
                          task_label=task['name'],
                          opt_config=opt_config,
                          seed=SEED)

    model_placeholder = ModelCoordinator(data, folds, model_config, model_class,
                                         opt_config, task, evaluator, checkpoint)

    space, opt_alg, max_evals = model_placeholder.get_hyperopt_params()
    trials = Trials()
    fmin(model_placeholder.objective,
         space=space,
         max_evals=max_evals,
         algo=opt_alg,
         trials=trials,
         verbose=False,
         rstate=SEED)

    # Re-train the model with the best parameters (setting aside a small validation set)
    return model_placeholder.train_single(trials.argmin, save=True)


def sequential_process(data, config, model_name, evaluator, device, checkpoint=None, opt_config=None, indent=0):
    model_class = getattr(importlib.import_module('models'), model_name)
    model_config = config['models'][model_name]
    model_config['device'] = device

    for dataset in data:
        log(f'Starting training process using {colored(dataset, Colors.BLUE)} dataset...', indent)

        indent += 1
        # Pre-processing optimization no available yet
        for preprocess_setup in [config['preprocess']]:
            log(f'Preprocess setup: {preprocess_setup}', indent)
            indent += 1

            trn_data, dev_data, tst_data = preprocess.run_config(preprocess_setup, data[dataset])
            if checkpoint is None:
                curr_data = pd.concat((trn_data, dev_data, tst_data), axis=0)
            else:
                curr_data = pd.concat((trn_data, dev_data), axis=0)

            for task in DATASET_TASKS[dataset]:
                task = process_task(task, curr_data)
                task['dataset'] = dataset
                task_id = f'{task["dataset"]}_{task["name"]}'
                evaluator.prepare_task(task)

                log(f'Current task: {task_id}', indent)
                indent += 1

                if checkpoint is None:
                    # First stage: Pre-train model on source task and save it with best hyper-params
                    checkpoint = f'{model_name}_{task_id}.pth'
                    model_path = os.path.join('models', 'pretrained', checkpoint)
                    if not os.path.isfile(model_path) or config['retrain']:
                        get_best_model(data, task, opt_config, evaluator, model_class, model_config, checkpoint=None)

                    # Second stage of the sequential process (Fine-tune and test on target task)
                    sequential_process(data={k: v for k, v in data.items() if k != dataset},
                                       config=config,
                                       model_name=model_name,
                                       checkpoint=checkpoint,
                                       evaluator=evaluator,
                                       device=device
                                       )
                else:
                    model = get_best_model(data, task, opt_config, evaluator, model_class, model_config,
                                           checkpoint=checkpoint)
                    ModelCoordinator.test_single(tst_data, task, model=model)

                indent -= 1
            indent -= 1
        indent -= 1


def run(config):
    config = load_config(config)
    opt_config = config['optimization']
    data = read_datasets(config['datasets'], tasks=DATASET_TASKS, lang='es')
    evaluator = Evaluator(config['evaluation']['metrics'])
    device = 'cuda' if cuda.is_available() else 'cpu'
    indent = 0
    iterations = 0

    for mi, model_name in enumerate(config['models']):
        log(colored('#' * 50, Colors.BLUE), indent)
        log(colored(f'Making experiments for {mi + 1}.{model_name}...', Colors.BLUE), indent)
        log(colored('#' * 50, Colors.BLUE), indent)
        indent += 1

        # Depending on the mode (mtl, sequential)
        if config['transfer_learning']['mode'] == 'sequential':
            log('Sequential Transfer Learning experiment starting...', indent)
            sequential_process(data, config, model_name, evaluator, device, opt_config, indent)

        elif config['transfer_learning']['mode'] == 'mtl':
            print('Multi-Task Learning experiment starting...')
            # TODO: Implement mtl approach
        else:
            raise Exception('"transfer_learning.mode" option must be included in the configuration file')
    indent -= 1

    print(f'Experiment {_name} completed with {iterations} iterations...')
