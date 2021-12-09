import os
from typing import Any, Dict
import numpy as np
import torch
import random
from torch import nn
import torch.distributed as dist

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, auc, roc_curve, precision_recall_curve, PrecisionRecallDisplay, average_precision_score, fbeta_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from ppi_pred.models.focal_loss import FocalLossWithLogits

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

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_model_bits(model:nn.Module):
    for param in model.parameters():
        param.numel()

#All gather from https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/maskrcnn_benchmark/utils/comm.py making use of picklable data
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        tensor of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    #I modified a bit the end of the code to match our output format
    #data_list = []
    data_tensor = torch.tensor([])
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        #data_list.append(pickle.loads(buffer))
        data_tensor = torch.cat(data_tensor,pickle.loads(buffer))

    #return data_list
    return data_tensor

supported_loss_types = ["BCE_logits", "focal_logits"]

def validate_config(config:Dict[str,Any]):
    """
    Just a function where we should add all our checks over the config to
    make sure that if it's invalid it fails at the start of a training run
    and not in the middle of it
    """

    assert config["training"]["loss_fn"]["type"] in supported_loss_types, f"loss type {type} is not supported"

    architecture_config = config["architecture"]

    supported_transformer_types = ["standard", "dt_fixup"]
    transformer_type = architecture_config["transformer_type"]
    assert transformer_type in supported_transformer_types, \
        f"transformer type {transformer_type} not in supported types {supported_transformer_types}"

def get_loss(loss_config:Dict[str,Any]):
    type = loss_config["type"]
    if type == "BCE_logits":
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([loss_config["pos_weight"]])
        )
    else:
        return FocalLossWithLogits(
            alpha=loss_config["alpha"],
            gamma=loss_config["gamma"]
        )