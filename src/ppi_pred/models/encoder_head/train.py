from argparse import Namespace
import copy
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F

from typing import Dict, Any

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DistributedSampler

from ppi_pred.dataset.embedding_seq_dataset import EmbeddingSeqDataset

from ppi_pred.models.encoder_head.scheduled_optimizer import ScheduledOptim

from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score

import wandb

def train(model, dataset:EmbeddingSeqDataset, config:Dict[str,Any], args:Namespace):

    should_log = not (args.distributed and (not args.is_main))
    if should_log:
        wandb.init(
            project="transformer",
            entity="ppi_pred_dl4nlp",
            config=config
        )

    training_config = config["training"]


    # get the dataloaders
    train_split, val_split, test_split = dataset.split()
    sampler = DistributedSampler(train_split) if args.distributed else None
    dataloaders = {}

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
        for seq_1, pad_1, seq_2, pad_2, tgt in tqdm(dataloaders['train'], disable=not should_log):
            seq_1 = seq_1.to(args.device)
            pad_1 = pad_1.to(args.device)
            seq_2 = seq_2.to(args.device)
            pad_2 = pad_2.to(args.device)
            tgt = tgt.to(args.device)

            opt.zero_grad()
            pred = model(seq_1, pad_1, seq_2, pad_2)
            loss = criterion(pred, tgt)

            total_loss += loss
            loss.backward()
            opt.step_and_update_lr()

        # get mean loss and not sum of mean batch loss
        total_loss /= epochs
        dist.reduce(total_loss, 0, dist.ReduceOp.SUM)
        total_loss /= args.world_size

        accs = test(dataloaders, model, args.device, verbose=should_log)
        if val_max < accs['val'] and should_log:
            val_max = accs['val']
            best_model = copy.deepcopy(model)

        if should_log:
            print("Epoch {}: Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}".format(
                epoch + 1, accs['val'], accs['test'], total_loss.item()))
            wandb.log({
                "training loss": total_loss,
                "validation accuracy": accs['val'],
                "test accuracy": accs['test']
            })

    final_accs = test(dataloaders, best_model, args.device, should_log)
    if should_log:
        print("FINAL MODEL: Validation: {:.4f}. Test: {:.4f}".format(final_accs['val'], final_accs['test']))

    return best_model

def test(dataloaders, model, device, verbose):
    model.eval()

    accs = {}
    for dataset in dataloaders:
        if dataset == "train":
            continue

        labels = []
        predictions = []
        for seq_1, pad_1, seq_2, pad_2, tgt in tqdm(dataloaders[dataset], disable=not verbose):

            seq_1 = seq_1.to(device)
            pad_1 = pad_1.to(device)
            seq_2 = seq_2.to(device)
            pad_2 = pad_2.to(device)
            tgt = tgt.to(device)

            pred = model(seq_1, pad_1, seq_2, pad_2)
            predictions.append(pred.round().cpu().detach().numpy())
            labels.append(tgt.cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accs[dataset] = accuracy_score(labels, predictions)
    return accs
