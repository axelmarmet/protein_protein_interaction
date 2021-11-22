import copy

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Dict, Any

from torch.utils.data.dataloader import DataLoader
from ppi_pred.dataset.embedding_seq_dataset import EmbeddingSeqDataset

from ppi_pred.models.encoder_head.scheduled_optimizer import ScheduledOptim

from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score

import wandb

def train(model, dataset:EmbeddingSeqDataset, config:Dict[str,Any], device):

    # get the dataloaders
    train_split, val_split, test_split = dataset.split()
    dataloaders = {}
    dataloaders['train'] = DataLoader(train_split, batch_size=config["batch_size"], shuffle=True, collate_fn=dataset.collate_fn)
    dataloaders['val'] =   DataLoader(val_split,   batch_size=config["batch_size"]//2, shuffle=True, collate_fn=dataset.collate_fn)
    dataloaders['test'] =  DataLoader(test_split,  batch_size=config["batch_size"]//2, shuffle=True, collate_fn=dataset.collate_fn)

    # get the optimizer
    opt = ScheduledOptim(
        optim.Adam(model.parameters()),
        config["lr_mult"],
        model.embedding_dim,
        config["warmup_steps"]
    )

    # get the loss
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor([config["pos_weight"]])
    ).to(device)

    epochs = config["epochs"]

    best_model = model
    val_max = -np.inf
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for seq_1, pad_1, seq_2, pad_2, tgt in tqdm(dataloaders['train']):
            seq_1 = seq_1.to(device)
            pad_1 = pad_1.to(device)
            seq_2 = seq_2.to(device)
            pad_2 = pad_2.to(device)
            tgt = tgt.to(device)

            opt.zero_grad()
            pred = model(seq_1, pad_1, seq_2, pad_2)
            loss = criterion(pred, tgt)

            total_loss += loss.item()
            loss.backward()
            opt.step_and_update_lr()

        accs = test(dataloaders, model, device)
        if val_max < accs['val']:
            val_max = accs['val']
            best_model = copy.deepcopy(model)


        print("Epoch {}: Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}".format(
            epoch + 1, accs['val'], accs['test'], total_loss))
        wandb.log({
            "training loss": total_loss,
            "validation accuracy": accs['val'],
            "test accuracy": accs['test']
        })
        wandb.watch(model)

    final_accs = test(dataloaders, best_model, device)
    print("FINAL MODEL: Validation: {:.4f}. Test: {:.4f}".format(final_accs['val'], final_accs['test']))
    return best_model

def test(dataloaders, model, device):
    model.eval()

    accs = {}
    for dataset in dataloaders:
        if dataset == "train":
            continue

        labels = []
        predictions = []
        for seq_1, pad_1, seq_2, pad_2, tgt in tqdm(dataloaders[dataset]):

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
