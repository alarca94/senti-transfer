import random
import torch

import numpy as np

from types import SimpleNamespace


def create_linspace(value):
    start = value['from']
    end = value['to']
    steps = value['in']
    space = np.linspace(start, end, num=steps).astype(value['type'] if 'type' in value else 'float32').tolist()
    return space


def get_gpu_usage(device=None):
    r""" Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    return '{:.2f} G/{:.2f} G'.format(reserved, total)
