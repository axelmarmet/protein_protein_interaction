import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=0):
        super(FocalLoss, self).__init__()
        # alpha is proportion of positive instances
        # gamma is relaxation parameter
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        p_t = torch.exp(-BCE_loss)

        # if target = 1, use (1 - alpha), otherwise alpha 
        alpha_tensor = (1 - self.alpha) * targets + self.alpha * (1 - targets)
        f_loss = alpha_tensor * (1 - p_t) ** self.gamma * BCE_loss
        return f_loss.mean()
    


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

        self.loss_function = FocalLoss(gamma=args.gamma)

    
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