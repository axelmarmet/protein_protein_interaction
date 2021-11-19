import copy

import torch
from torch import nn
import torch.optim as optim

from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score

def train(dataloaders, model, args, epochs=10):
    opt = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    best_model = model
    val_max = -np.inf
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for seq_1, pad_1, seq_2, pad_2, tgt in dataloaders['train']:
            seq_1 = seq_1.to(args.device)
            pad_1 = pad_1.to(args.device)
            seq_2 = seq_2.to(args.device)
            pad_2 = pad_2.to(args.device)
            tgt = tgt.to(args.device)

            opt.zero_grad()
            pred = model(seq_1, pad_1, seq_2, pad_2)
            loss = criterion(pred, tgt)
            total_loss += loss.item()
            loss.backward()
            opt.step()

        accs = test(dataloaders, model, args)
        if val_max < accs['val']:
            val_max = accs['val']
            best_model = copy.deepcopy(model)

        print("Epoch {}: Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}".format(
            epoch + 1, accs['val'], accs['test'], total_loss))

    final_accs = test(dataloaders, best_model, args)
    print("FINAL MODEL: Validation: {:.4f}. Test: {:.4f}".format(final_accs['val'], final_accs['test']))
    return best_model

def test(dataloaders, model, args):
    model.eval()

    accs = {}
    for dataset in dataloaders:
        if dataset == "train":
            continue

        labels = []
        predictions = []
        for seq_1, pad_1, seq_2, pad_2, tgt in tqdm(dataloaders[dataset]):

            seq_1 = seq_1.to(args.device)
            pad_1 = pad_1.to(args.device)
            seq_2 = seq_2.to(args.device)
            pad_2 = pad_2.to(args.device)
            tgt = tgt.to(args.device)

            pred = model(seq_1, pad_1, seq_2, pad_2)
            predictions.append(pred.round().cpu().detach().numpy())
            labels.append(tgt.cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accs[dataset] = accuracy_score(labels, predictions)
    return accs
