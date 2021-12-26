from pathlib import Path
import os
import copy
import wandb

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
from ppi_pred.metrics import *

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
    parser.add_argument('--batch_size', type=int,
                        help='Batch size during training.')
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
    parser.add_argument('--gamma', type=float,
                        help='Gamma for focal loss.')
    parser.add_argument('--pooling', type=str,
                        help='Type of pooling.')
    
    parser.set_defaults(
        device='cuda:0',
        epochs=200,
        batch_size=128,
        hidden_dim=128,
        num_layers=4,
        opt='adam',
        weight_decay=5e-4,
        dropout=0.3,
        lr=1e-3,
        pooling='max',
        gamma=2,
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
    
    # scheduler
    scheduler_gamma = 0.9
    lambda1 = lambda epoch: (1 - epoch/args.epochs)**scheduler_gamma 
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)

    config = {k:v for k, v in vars(args).items()}
    wandb.init(
            project="pooling-baseline",
            entity="ppi_pred_dl4nlp",
            config=config
        )

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


        scores = test(dataloaders, model, args)
        if val_max < scores['val']['AUC_PRC']:
            val_max = scores['val']['AUC_PRC']
            best_model = copy.deepcopy(model)

        if(scheduler is not None):
            scheduler.step()

        wandb.log({
                "training loss": total_loss,

                "training accuracy": scores['train']['acc'],
                "validation accuracy": scores['val']['acc'],
                "test accuracy": scores['test']['acc'],

                "training precision": scores['train']['precision'],
                "validation precision": scores['val']['precision'],
                "test precision": scores['test']['precision'],

                "training recall": scores['train']['recall'],
                "validation recall": scores['val']['recall'],
                "test recall": scores['test']['recall'],

                "training fbeta": scores['train']['fbeta'],
                "validation fbeta": scores['val']['fbeta'],
                "test fbeta": scores['test']['fbeta'],

                "training average precision": scores['train']['AP_score'],
                "validation average precision": scores['val']['AP_score'],
                "test average precision": scores['test']['AP_score'],

                "training AUC PR": scores['train']['AUC_PRC'],
                "validation AUC PR": scores['val']['AUC_PRC'],
                "test AUC PR": scores['test']['AUC_PRC'],

                "training AUC ROC": scores['train']['AUC_ROC'],
                "validation AUC ROC": scores['val']['AUC_ROC'],
                "test AUC ROC": scores['test']['AUC_ROC'],
            })

        print("Epoch {}:\nTrain: {}\nValidation: {}\nTest: {}\nLoss: {}\n".format(
              epoch + 1, scores['train'], scores['val'], scores['test'], total_loss))


    final_scores = test(dataloaders, best_model, args)
    print("FINAL MODEL:\nTrain: {}\nValidation: {}\nTest: {}\n".format(
          final_scores['train'], final_scores['val'], final_scores['test']))
    return best_model, final_scores


def test(dataloaders, model, args):
    model.eval()

    scores = {}
    for dataset in dataloaders:
        labels = []
        predictions = []
        for (batch, label) in dataloaders[dataset]:
            batch = batch.to(args.device)
            pred = model(batch)
            predictions.append(pred.cpu().detach().numpy())
            labels.append(label.cpu().numpy())

        predictions = torch.tensor(np.concatenate(predictions))
        labels = torch.tensor(np.concatenate(labels))
        scores[dataset] = metrics(predictions, labels)
    return scores





if __name__ == "__main__":
    args = arg_parse()
    if args.pooling not in ['mean', 'max']:
        raise ValueError("Unsupported pooling operation.")
     
    args.device = (args.device if torch.cuda.is_available() else 'cpu')
    dataset_directory = "../../dataset"
    training_file = dataset_directory + "/training_set.pkl"
    test_file = dataset_directory + "/test_set.pkl"

    set_seed()

    print(f"Generating dataset")
    training_data = pd.read_pickle(training_file)
    test_data = pd.read_pickle(test_file)
    
    """
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
    """
    training_data = training_data.sample(frac=1)

    num_instances = training_data.shape[0]
    split = [0.85, 0.15]
    split_num = int(split[0] * num_instances)

    train_dataset = PoolingDataset(training_data.iloc[:split_num, :], dataset_directory, args.pooling)
    val_dataset = PoolingDataset(training_data.iloc[split_num:, :], dataset_directory, args.pooling)
    test_dataset = PoolingDataset(test_data, dataset_directory, args.pooling)

    input_dim = train_dataset.input_dim

    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dataloaders['val'] = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    dataloaders['test'] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    best_model, scores = train(dataloaders, input_dim, args)
    checkpoint_path = f"checkpoints/{args.pooling}"
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    torch.save(best_model, f"{checkpoint_path}/best_model.pt")