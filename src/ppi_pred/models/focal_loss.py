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

class FocalLossWithLogits(nn.Module):

    def __init__(self, alpha=0.1, gamma=0):
        super(FocalLossWithLogits, self).__init__()
        # alpha is proportion of positive instances
        # gamma is relaxation parameter
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        BCE_loss = torch.log(p_t)

        # if target = 1, use (1 - alpha), otherwise alpha
        alpha_tensor = (1 - self.alpha) * targets + self.alpha * (1 - targets)
        f_loss = (alpha_tensor * (1 - p_t) ** self.gamma) * BCE_loss
        return f_loss.mean()