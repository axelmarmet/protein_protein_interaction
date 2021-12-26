import torch
import torch.nn as nn
import torch.nn.functional as F

from ppi_pred.models.focal_loss import FocalLoss



class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(MLP, self).__init__()
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
        self.dropout = nn.Dropout(p=args.dropout)

        self.loss_function = FocalLoss(gamma=args.gamma)


    def forward(self, x):
        for i in range(len(self.lin_layer) - 1):
            x = self.lin_layer[i](x)
            x = self.batch_norm[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.lin_layer[len(self.lin_layer) - 1](x)
        x = self.sigmoid(x).squeeze()
        return x

    def loss(self, pred, label):
        return self.loss_function(pred, label.float())