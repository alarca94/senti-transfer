import torch

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils.constants import *


def create_data_loader(data, encoder, batch_size, cols):
    dataset = SimpleDataset(
        input_ids=data.index.to_numpy(),
        texts=data[cols.get('input', 'text')].to_numpy(),
        targets=data[cols.get('target', 'toxicity')].to_numpy(),
        encoder=encoder
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn_pad
    )


def collate_fn_pad(batch):
    # transform list of dicts into dict of lists
    new_batch = {k: [] for k in batch[0]}
    for sample in batch:
        for k in new_batch.keys():
            new_batch[k].append(sample[k])

    for k in ['input_ids', 'attention_mask']:
        new_batch[k] = pad_sequence(new_batch[k], batch_first=True)

    new_batch['mask'] = (new_batch['input_ids'] != 0).to(dtype=torch.int)
    new_batch['targets'] = torch.tensor(new_batch['targets'])
    return new_batch


class SimpleDataset(Dataset):
    def __init__(self, input_ids, texts, targets, encoder):
        self.input_ids = input_ids
        self.texts = texts
        self.targets = targets
        self.encoder = encoder

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        t_id = self.input_ids[item]
        text = self.texts[item]
        target = self.targets[item]

        enc_text = self.encoder.encode_plus(text,
                                            add_special_tokens=True,
                                            max_length=512,
                                            truncation=True,
                                            return_token_type_ids=False,
                                            return_attention_mask=True,
                                            return_tensors='pt')

        return {
            'comment_ids': t_id,
            'input_ids': enc_text['input_ids'].flatten(),
            'attention_mask': enc_text['attention_mask'].flatten(),
            'targets': target
        }


def process_task(task, data):
    task_type = None
    task_mapping = None
    if isinstance(task, tuple):
        task_name = task[0]
        task_type = task[1]
        if len(task) == 3:
            task_mapping = task[2]
    else:
        task_name = task

    if task_type == CAT_TYPE:
        task_mapping = {v: i for i, v in enumerate(data[task_name].unique())}

    if task_mapping is not None:
        data[task_name] = data[task_name].map(task_mapping)

    if task_type == ORD_TYPE:
        data[task_name] = data[task_name].astype(np.float64) / len(task_mapping)

    if task_mapping is None:
        task_mapping = {v: v for v in data[task_name].unique()}

    return {'name': task_name, 'type': task_type, 'map': task_mapping}


