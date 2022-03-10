import gc
import os
import numpy as np
import pandas as pd
import torch
import time

from itertools import filterfalse
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AdamW, logging

from utils.evaluation import Evaluator
from utils.experiment_utils import get_gpu_usage
from utils.in_out import print_gpu_usage, colored, Colors, print_results
from utils.nn_utils import get_predictions
from utils.regularization import EarlyStopping


class TransformerBase(nn.Module):
    def __init__(self, **kwargs):
        super(TransformerBase, self).__init__()
        logging.set_verbosity_warning()
        logging.set_verbosity_error()
        self.lr = kwargs.get('init_lr', 1e-5)
        self.epochs = kwargs.get('epochs', 50)
        self.batch_size = kwargs.get('batch_size', 16)
        self.drop_rate = kwargs.get('dropout', 0.2)
        self.n_classes = kwargs.get('n_classes', 2)
        self.device = kwargs.get('device', 'cpu')
        self.opt_name = kwargs.get('optimizer', 'AdamW')
        self.task_name = kwargs.get('task_name', 'no-task-name')
        self.optimizer = None
        self.lr_scheduler = None
        self.out_units = 1 if self.n_classes == 2 else self.n_classes
        self.name = 'transformer-base'
        self.clipping_norm = 1.0

        # Layers initialized in init_transformer_block() depending on the transformer's name
        self.tokenizer = None
        self.transformer_block = None
        self.dropout = nn.Dropout(self.drop_rate)
        self.out = None

        self.early_stopping = EarlyStopping(kwargs.get('patience', 5), kwargs.get('delta', 1e-5))
        self.evaluator = kwargs.get('evaluator', Evaluator('default'))
        self.loss_fn = kwargs.get('loss_fn', nn.BCELoss())
        self.last_act = nn.LogSoftmax(dim=-1) if isinstance(self.loss_fn, nn.NLLLoss) else nn.Sigmoid()

    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.huggingface_name)
        self.transformer_block = AutoModel.from_pretrained(self.huggingface_name)
        self.out = nn.Linear(self.transformer_block.embeddings.word_embeddings.embedding_dim, self.out_units)
        self.to(self.device)
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=(lambda epoch: 0.95))

    def create_optimizer(self):
        if self.opt_name.lower() == 'adamw':
            return AdamW(filterfalse(lambda p: not p.requires_grad, self.parameters()), lr=self.lr)
        elif self.opt_name.lower() == 'adam':
            return optim.Adam(filterfalse(lambda p: not p.requires_grad, self.parameters()), lr=self.lr)
        else:
            raise Exception(f'{self.opt_name} is not supported!')

    def reset(self, n_classes=2, file_path=None):
        self.n_classes = n_classes
        self.out_units = 1 if self.n_classes == 2 else self.n_classes

        if file_path is None:
            self.transformer_block = AutoModel.from_pretrained(self.huggingface_name)
            self.out = nn.Linear(self.transformer_block.embeddings.word_embeddings.embedding_dim, self.out_units)
        else:
            self.load(self.name, file_path)
            self.out = nn.Linear(self.transformer_block.embeddings.word_embeddings.embedding_dim, self.out_units)

        self.to(self.device)
        self.early_stopping.reset()
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=(lambda epoch: 0.95))

    def forward(self, input_ids, attention_mask):
        transformer_out = self.transformer_block(input_ids=input_ids, attention_mask=attention_mask)

        # The first element of the transformer output is the last hidden state
        # Shape [BATCH_SIZE, SEQ_LENGTH, EMB_SIZE], First token is CLS
        cls_token = transformer_out[0][:, 0, :]

        return self.last_act(self.out(self.dropout(cls_token))).squeeze()

    def fit(self, trn_dataloader, val_dataloader=None, log=False, save=False):
        trn_scores = None
        time_records = {'trn': [], 'val': []}
        max_mem_usage = 0
        for epoch in range(self.epochs):
            # print(f'Epoch {epoch+1} / {self.epochs}...')
            self.train()
            # Training iteration
            iter_data = tqdm(trn_dataloader, total=len(trn_dataloader), ncols=100,
                             desc=colored(f"Train {epoch+1:>5}/{self.epochs}", Colors.PURPLE))
            start_time = time.time()
            for batch_idx, d in enumerate(iter_data):
                _, loss = self.step(d)

                loss.backward()
                self.clip_grad_norm()
                self.optimizer.step()
                postfix_str = colored(f'Train Loss: {loss.item() / self.evaluator.iter:.4f}', Colors.GREEN)
                if 'cuda' in self.device:
                    mem_usage = torch.cuda.max_memory_reserved(self.device) / 1024 ** 3
                    postfix_str += f", {colored('GPU RAM: ' + get_gpu_usage(self.device), Colors.YELLOW)}"
                    if mem_usage > max_mem_usage:
                        max_mem_usage = mem_usage
                iter_data.set_postfix_str(postfix_str)
                self.optimizer.zero_grad()

                torch.cuda.empty_cache()
                gc.collect()

                if batch_idx > 2:
                    break

            time_records['trn'].append(time.time() - start_time)
            trn_scores = self.evaluator.compute_scores()
            if log:
                print_results(trn_scores, time_records['trn'][-1], phase='train')
            self.evaluator.reset()

            # Validation iteration
            if val_dataloader is not None:
                self.eval()
                start_time = time.time()
                self.evaluate(val_dataloader, epoch=epoch)
                time_records['val'].append(time.time() - start_time)
                val_scores = self.evaluator.compute_scores()

                if log:
                    print_results(val_scores, time_records['val'][-1], phase='validation')

                self.evaluator.reset()
                self.early_stopping(trn_scores, val_scores, self, save=save)

            if self.early_stopping.early_stop:
                break
            else:
                self.lr_scheduler.step()

        if val_dataloader is not None:
            self.early_stopping.best_trn_scores['Time'] = np.mean(time_records['trn'])
            self.early_stopping.best_val_scores['Time'] = np.mean(time_records['val'])
            if log:
                print_results(self.early_stopping.best_trn_scores, phase='train')
                print_results(self.early_stopping.best_val_scores, phase='validation')

            return pd.Series(self.early_stopping.best_trn_scores), pd.Series(self.early_stopping.best_val_scores), \
                   max_mem_usage

        trn_scores['Time'] = np.mean(time_records['trn'])
        if log:
            print_results(trn_scores, phase='train')

        return pd.Series(trn_scores), None, max_mem_usage

    def step(self, data):
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        targets = data['targets'].to(self.device)

        outputs = self(input_ids, attention_mask)
        loss = self.loss_fn(outputs, targets)

        preds = get_predictions(outputs, self.loss_fn)
        self.evaluator.update(y_pred=preds, y_true=targets, loss=loss)

        return preds, loss

    def evaluate(self, data_loader, epoch=None):
        with torch.no_grad():
            if epoch is not None:
                iter_data = tqdm(data_loader, total=len(data_loader), ncols=100,
                                 desc=colored(f"Val {epoch+1:>7}/{self.epochs}", Colors.PURPLE))
            else:
                iter_data = tqdm(data_loader, total=len(data_loader), ncols=100,
                                 desc=colored(f"Evaluation", Colors.PURPLE))
            for batch_idx, d in enumerate(iter_data):
                _, loss = self.step(d)
                postfix_str = colored(f'Eval Loss: {loss.item() / self.evaluator.iter:.4f}', Colors.GREEN)
                if 'cuda' in self.device:
                    postfix_str += f", {colored('GPU RAM: ' + get_gpu_usage(self.device), Colors.YELLOW)}"
                iter_data.set_postfix_str(postfix_str)

                if batch_idx > 2:
                    break

            self.evaluator.compute_scores()

    def clip_grad_norm(self):
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clipping_norm)

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def save(self, path='./models/pretrained'):
        path = os.path.join(path, f'{self.name}_{self.task_name}.pth')
        print(f'Saving the model @ {path}')
        torch.save(self.state_dict(), path)

    def load(self, name, path='./models/pretrained'):
        self.load_state_dict(torch.load(os.path.join(path, name)))
        self.name = name.split('.')[0]  # Remove file extension e.g. .pth

    def print_state(self):
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())


class BETO(TransformerBase):
    def __init__(self, **kwargs):
        super(BETO, self).__init__(**kwargs)
        self.name = 'BETO'
        self.huggingface_name = 'dccuchile/bert-base-spanish-wwm-' + kwargs.get('letter_case', 'uncased')
        self.init()
