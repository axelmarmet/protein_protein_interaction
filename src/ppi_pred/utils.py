import os
from typing import Any, Dict
import numpy as np
import torch
import random
import torch.distributed as dist



def set_seed(seed=44):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# necessary functions for distributed training

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def validate_config(config:Dict[str,Any]):
    """
    Just a function where we should add all our checks over the config to
    make sure that if it's invalid it fails at the start of a training run
    and not in the middle of it
    """

    architecture_config = config["architecture"]

    supported_transformer_types = ["standard", "dt_fixup"]
    transformer_type = architecture_config["transformer_type"]
    assert transformer_type in supported_transformer_types, \
        f"transformer type {transformer_type} not in supported types {supported_transformer_types}"