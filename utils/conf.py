# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import numpy as np

def get_device() -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './results/'


#def set_random_seed(seed: int) -> None:
#    """
#    Sets the seeds at a certain value.
#    :param seed: the value to be set
#    """
#    random.seed(seed)
#    np.random.seed(seed)
#    torch.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    # Generate a seed for the worker based on the initial seed
    worker_seed = torch.initial_seed() % 2**32
    # Set the seed for NumPy operations in the worker
    np.random.seed(worker_seed)
    # Set the seed for random number generation in the worker
    random.seed(worker_seed)
