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
    
    parser.set_defaults(
        device='cuda:0',
        epochs=30,
        hidden_dim=128,
        num_layers=4,
        opt='adam',
        weight_decay=1e-5,
        dropout=0.3,
        lr=1e-3,
        pooling='mean'
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





class PoolingDataset(Dataset):
    def __init__(self, data, dataset_directory, pooling_operation, layers=[2, 6, 11]):
        self.X = []
        self.y = []

        for i in range(data.shape[0]):
            sequence1 = data.iloc[i, 0]
            sequence2 = data.iloc[i, 1]

            cur_embedding = []
            for l in layers:
                embedding1 = torch.load(f"{dataset_directory}/{sequence1}/embeddings_layer_{l}_MSA_Transformer.pt",
                                        map_location=torch.device('cpu'))
                embedding2 = torch.load(f"{dataset_directory}/{sequence2}/embeddings_layer_{l}_MSA_Transformer.pt",
                                        map_location=torch.device('cpu'))

                if(pooling_operation == 'mean'):
                    embedding1 = embedding1[0].squeeze().mean(dim=0)
                    embedding2 = embedding2[0].squeeze().mean(dim=0)

                if(pooling_operation == 'max'):
                    embedding1 = embedding1[0].squeeze().max(dim=0)[0]
                    embedding2 = embedding2[0].squeeze().max(dim=0)[0]
        
                cur_embedding.append(embedding1)
                cur_embedding.append(embedding2)


            self.X.append(torch.cat(cur_embedding, dim=0))
            self.y.append(int(data.iloc[i, -1]))

        self.input_dim = self.X[-1].size(0)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])





class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(MLP, self).__init__()
        self.dropout = args.dropout
        self.num_layers = args.num_layers

        self.lin_layer = nn.ModuleList()
        self.lin_layer.append(nn.Linear(input_dim, hidden_dim))

        self.batch_norm = nn.ModuleList()
        self.batch_norm.append(torch.nn.BatchNorm1d(hidden_dim))

        for l in range(args.num_layers - 2):
            self.lin_layer.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norm.append(torch.nn.BatchNorm1d(hidden_dim))

        self.lin_layer.append(nn.Linear(hidden_dim, 1))
        self.sigmoid = nn.Sigmoid()

        self.loss_function = nn.BCELoss()

    
    def forward(self, x):
        for i in range(len(self.lin_layer) - 1):
            x = self.lin_layer[i](x)
            x = self.batch_norm[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)

        x = self.lin_layer[len(self.lin_layer) - 1](x)
        x = self.sigmoid(x).squeeze()
        return x

    def loss(self, pred, label):
        return self.loss_function(pred, label.float())


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
            batch.to(args.device)
            label.to(args.device)
            
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
    return best_model


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
    set_seed()

    args = arg_parse()
    if args.pooling not in ['mean', 'max']:
        raise ValueError("Unsupported pooling operation.")
     
    args.device = (args.device if torch.cuda.is_available() else 'cpu')
    dataset_directory = "../../dataset"
    labels_file = dataset_directory + "/training_set.pkl"


    try:
        train_dataset = torch.load(f"{dataset_directory}/pooling-datasets/train_dataset_{args.pooling}.pt")
        val_dataset = torch.load(f"{dataset_directory}/pooling-datasets/val_dataset_{args.pooling}.pt")
        test_dataset = torch.load(f"{dataset_directory}/pooling-datasets/test_dataset_{args.pooling}.pt")
    except:
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

        train_dataset = PoolingDataset(data.iloc[:split_num[0], :], dataset_directory, args.pooling)
        val_dataset = PoolingDataset(data.iloc[split_num[0]:split_num[1], :], dataset_directory, args.pooling)
        test_dataset = PoolingDataset(data.iloc[split_num[1]:, :], dataset_directory, args.pooling)

        if(not Path(f"{dataset_directory}/pooling-datasets").exists()):
            os.mkdir(f"{dataset_directory}/pooling-datasets")

        torch.save(train_dataset, f"{dataset_directory}/pooling-datasets/train_dataset_{args.pooling}.pt")
        torch.save(val_dataset, f"{dataset_directory}/pooling-datasets/val_dataset_{args.pooling}.pt")
        torch.save(test_dataset, f"{dataset_directory}/pooling-datasets/test_dataset_{args.pooling}.pt")


    input_dim = train_dataset.input_dim

    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dataloaders['val'] = DataLoader(val_dataset, batch_size=64, shuffle=False)
    dataloaders['test'] = DataLoader(test_dataset, batch_size=64, shuffle=False)

    train(dataloaders, input_dim, args)