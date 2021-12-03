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
from ppi_pred.metrics import metrics

from ppi_pred.models.encoder_head.scheduled_optimizer import ScheduledOptim

from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score

import wandb

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
    sampler = DistributedSampler(train_split) if args.distributed else None
    dataloaders = {}

    assert training_config["simulated_batch_size"] % training_config["batch_size"] == 0
    steps_before_opt = training_config["simulated_batch_size"] // training_config["batch_size"]

    # we divide the batch size by the world_size so that the total
    # batch size does not vary with the number of GPUs used
    assert training_config["batch_size"] % args.world_size == 0, \
        f"batch size ({training_config['batch_size']}) is not cleanly divided by " \
        f"number of gpus ({args.world_size})"
    batch_size = training_config["batch_size"] // args.world_size

    dataloaders['train'] = DataLoader(train_split, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, collate_fn=dataset.collate_fn)
    dataloaders['val'] =   DataLoader(val_split,   batch_size=batch_size, collate_fn=dataset.collate_fn)
    dataloaders['test'] =  DataLoader(test_split,  batch_size=batch_size, collate_fn=dataset.collate_fn)

    # get the optimizer
    opt = ScheduledOptim(
        optim.Adam(model.parameters()),
        training_config["lr_mult"],
        config["architecture"]["e_dim"],
        training_config["warmup_steps"]
    )

    # get the loss
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor([training_config["pos_weight"]])
    ).to(args.device)

    epochs = training_config["epochs"]

    best_model = model
    val_max = -np.inf
    for epoch in range(epochs):

        if args.distributed:
            # necessary for shuffling to work with the distributed sampler
            sampler.set_epoch(epoch) # type:ignore
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

            if i == 100:
                break

        # get mean loss and not sum of mean batch loss
        total_loss /= epochs
        if args.distributed:
            dist.reduce(total_loss, 0, dist.ReduceOp.SUM)
            total_loss /= args.world_size

        accs = test(dataloaders, model, args, verbose=should_log)
        if val_max < accs['val'] and should_log:
            val_max = accs['val']
            best_model = copy.deepcopy(model)

        if should_log:
            print("Epoch {}: Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}".format(
                epoch + 1, accs['val'], accs['test'], total_loss.item()))
            if args.use_wandb:
                wandb.log({
                    "training loss": total_loss,
                    "validation accuracy": accs['val'],
                    "test accuracy": accs['test']
                })

    final_accs = test(dataloaders, best_model, args, should_log)
    if should_log:
        print("FINAL MODEL: Validation: {:.4f}. Test: {:.4f}".format(final_accs['val'], final_accs['test']))

    return best_model


def test(dataloaders, model, args:Namespace, verbose:bool):

    model.eval()

    accs = {}
    for dataset in dataloaders:
        if dataset == "train":
            continue

        results = []
        for inp, tgt in tqdm(dataloaders[dataset], disable=not verbose):
            inp.to(args.device)
            pred = model(inp)

            pred:Tensor = model.forward(inp).round().detach().cpu()

            res = metrics(pred.reshape(-1, 1), tgt.reshape(-1))

            results.append(res)

        accs[dataset] = results = torch.cat(results).mean()

    if args.distributed:
        for value in accs.values():
            dist.reduce(value, 0, dist.ReduceOp.SUM)

    accs = {key:val.item() / args.world_size for key, val in accs.items()}

    return accs