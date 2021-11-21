from pathlib import Path
import os
import copy

import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

from ppi_pred.models.pooling_baseline_model import MLP
from ppi_pred.dataset.pooling_baseline_dataset import PoolingDataset


def set_seed(seed=44):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def arg_parse():
    parser = argparse.ArgumentParser(description='Node classification arguments.')

    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--hidden_dim', type=int,
                        help='Hidden dimension of GNN.')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph convolution layers.')
    parser.add_argument('--opt', type=str,
                        help='Optimizer such as adam, sgd, rmsprop or adagrad.')
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay.')
    parser.add_argument('--dropout', type=float,
                        help='The dropout ratio.')
    parser.add_argument('--lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--pooling', type=str,
                        help='Type of pooling.')
    parser.add_argument('--MSA_layer', type=int,
                        help='Id of the MSA layer (2, 6, or 11).')
    
    parser.set_defaults(
        device='cuda:0',
        epochs=30,
        hidden_dim=128,
        num_layers=4,
        opt='adam',
        weight_decay=1e-5,
        dropout=0.3,
        lr=1e-3,
        pooling='mean',
        MSA_layer=6
    )
    return parser.parse_args()


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    return optimizer


def train(dataloaders, input_dim, args):
    model_cls = MLP
    model = model_cls(input_dim, args.hidden_dim, args).to(args.device)
    opt = build_optimizer(args, model.parameters())

    best_model = model
    val_max = -np.inf
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for (batch, label) in dataloaders['train']:
            batch = batch.to(args.device)
            label = label.to(args.device)
            
            opt.zero_grad()
            pred = model(batch)
            loss = model.loss(pred, label)
            total_loss += loss.item()
            loss.backward()
            opt.step()

        accs = test(dataloaders, model, args)
        if val_max < accs['val']:
            val_max = accs['val']
            best_model = copy.deepcopy(model)

        print("Epoch {}: Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}".format(
            epoch + 1, accs['train'], accs['val'], accs['test'], total_loss))


    final_accs = test(dataloaders, best_model, args)
    print("FINAL MODEL: Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}".format(
            final_accs['train'], final_accs['val'], final_accs['test']))
    return best_model, final_accs


def test(dataloaders, model, args):
    model.eval()

    accs = {}
    for dataset in dataloaders:
        labels = []
        predictions = []
        for (batch, label) in dataloaders[dataset]:
            batch.to(args.device)
            pred = model(batch)
            predictions.append(pred.round().cpu().detach().numpy())
            labels.append(label.cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accs[dataset] = accuracy_score(labels, predictions)
    return accs





if __name__ == "__main__":
    args = arg_parse()
    if args.pooling not in ['mean', 'max']:
        raise ValueError("Unsupported pooling operation.")
     
    args.device = (args.device if torch.cuda.is_available() else 'cpu')
    dataset_directory = "../../dataset"
    labels_file = dataset_directory + "/training_set.pkl"


    accs = np.zeros(5)
    for seed in range(5):
        set_seed(seed + 1)

        print(f"Generating dataset {seed + 1}")

        data = pd.read_pickle(labels_file)
        valid_instances = []
        for i in range(data.shape[0]):
            sequence1 = data.iloc[i, 0]
            sequence2 = data.iloc[i, 1]

            if(not Path(f"{dataset_directory}/{sequence1}").exists()):
                continue

            if(not Path(f"{dataset_directory}/{sequence2}").exists()):
                continue

            valid_instances.append(i)


        valid_instances = np.array(valid_instances)
        data = data.iloc[valid_instances, :]
        data = data.sample(frac=1)

        num_instances = data.shape[0]
        split = [0.8, 0.1, 0.1]
        split_num = [int(split[0] * num_instances), int((split[0] + split[1]) * num_instances)]

        train_dataset = PoolingDataset(data.iloc[:split_num[0], :], dataset_directory, args.pooling, layers=[args.MSA_layer])
        val_dataset = PoolingDataset(data.iloc[split_num[0]:split_num[1], :], dataset_directory, args.pooling, layers=[args.MSA_layer])
        test_dataset = PoolingDataset(data.iloc[split_num[1]:, :], dataset_directory, args.pooling, layers=[args.MSA_layer])


        input_dim = train_dataset.input_dim

        dataloaders = {}
        dataloaders['train'] = DataLoader(train_dataset, batch_size=64, shuffle=True)
        dataloaders['val'] = DataLoader(val_dataset, batch_size=64, shuffle=False)
        dataloaders['test'] = DataLoader(test_dataset, batch_size=64, shuffle=False)

        _, acc = train(dataloaders, input_dim, args)
        accs[seed] = acc['test']

print(f"OVERALL RESULT: {np.mean(accs)} +- {np.std(accs)}")