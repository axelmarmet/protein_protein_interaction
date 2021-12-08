from argparse import Namespace
import copy
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F

from typing import Dict, Any
from torch import Tensor

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DistributedSampler

from ppi_pred.dataset.embedding_seq_dataset import EmbeddingSeqDataset
from ppi_pred.metrics import metrics, print_metrics

from ppi_pred.models.encoder_head.scheduled_optimizer import ScheduledOptim

from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score

import wandb

from ppi_pred.utils import all_gather, get_loss

def train(model, dataset:EmbeddingSeqDataset, config:Dict[str,Any], args:Namespace):

    should_log = not (args.distributed and (not args.is_main))
    if should_log and args.use_wandb:
        wandb.init(
            project="transformer",
            entity="ppi_pred_dl4nlp",
            config=config
        )
        wandb.watch(model, log='all')

    training_config = config["training"]


    # get the dataloaders
    train_split, val_split, test_split = dataset.split()
    dataloaders = {}

    assert training_config["simulated_batch_size"] % training_config["batch_size"] == 0
    steps_before_opt = training_config["simulated_batch_size"] // training_config["batch_size"]

    # we divide the batch size by the world_size so that the total
    # batch size does not vary with the number of GPUs used
    assert training_config["batch_size"] % args.world_size == 0, \
        f"batch size ({training_config['batch_size']}) is not cleanly divided by " \
        f"number of gpus ({args.world_size})"
    batch_size = training_config["batch_size"] // args.world_size

    dataloaders['train'] = dataset.get_dataloader_for_split(train_split, batch_size, True, args.distributed)
    dataloaders['val'] = dataset.get_dataloader_for_split(val_split, batch_size, False, False)
    dataloaders['test'] = dataset.get_dataloader_for_split(test_split, batch_size, False, False)

    # get the optimizer
    opt = ScheduledOptim(
        optim.Adam(model.parameters()),
        training_config["lr_mult"],
        config["architecture"]["e_dim"],
        training_config["warmup_steps"]
    )

    # get the loss
    criterion = get_loss(training_config["loss_fn"]).to(args.device)
    epochs = training_config["epochs"]

    best_model = model
    val_max = -np.inf
    for epoch in range(epochs):

        if args.distributed:
            # necessary for shuffling to work with the distributed sampler
            dataloaders['train'].sampler.set_epoch(epoch) # type:ignore
            dist.barrier()

        total_loss = torch.zeros((1), device=args.device)
        model.train()
        for i, (batch, tgt) in enumerate(tqdm(dataloaders['train'], disable=not should_log)):
            batch.to(args.device)
            tgt = tgt.to(args.device)

            pred = model(batch)
            loss = criterion(pred, tgt.float()) / steps_before_opt

            total_loss += loss
            loss.backward()

            if i % steps_before_opt == steps_before_opt-1:
                opt.step_and_update_lr()
                opt.zero_grad()


        # get mean loss and not sum of mean batch loss
        total_loss /= epochs
        if args.distributed:
            dist.reduce(total_loss, 0, dist.ReduceOp.SUM)
            total_loss /= args.world_size

        # computation done in excess (should be on only 1 GPU but is on all)
        metrics = test(dataloaders["val"], model, args, verbose=should_log)
        if should_log:

            if val_max < metrics['AUC_ROC'] and should_log:
                val_max = metrics['AUC_ROC']
                best_model = copy.deepcopy(model)

            print(f"Epoch {epoch + 1}:")
            print_metrics(metrics)

            if args.use_wandb:
                wandb.log({
                    "training loss": total_loss,
                    "classification metrics" : metrics
                })

    # computation done in excess (should be on only 1 GPU but is on all)
    final_metrics = test(dataloaders["test"], best_model, args, should_log)
    if should_log:
        print("FINAL MODEL:")
        print_metrics(final_metrics)

    return best_model

# import gc

@torch.no_grad()
def test(dataloader, model, args:Namespace, verbose:bool):

    model.eval()

    preds = []
    tgts = []
    for inp, tgt in tqdm(dataloader, disable=not verbose):
        inp.to(args.device)
        pred = model(inp)

        pred:Tensor = model.forward(inp).cpu()
        preds.append(pred)
        tgts.append(tgt)

    predictions = torch.cat(preds)
    targets = torch.cat(tgts)

    return metrics(predictions, targets)
