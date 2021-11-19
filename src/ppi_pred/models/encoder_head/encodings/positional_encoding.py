import math

import torch
from torch import nn

from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    helper Module that adds positional encoding to the token embedding to
    introduce a notion of word order.
    """

    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        """init the encoding

        Args:
            emb_size (int): size of the embedding
            dropout (float): dropout rate
            maxlen (int, optional): Maximal sequence length. Defaults to 5000.
        """
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)


    def forward(self, token_embedding: Tensor)->Tensor:
        """Add a positional encoding to the given embedding

        Args:
            token_embedding (Tensor): original embedding

        Returns:
            Tensor: the sum of the embedding and the positional encoding
        """
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
