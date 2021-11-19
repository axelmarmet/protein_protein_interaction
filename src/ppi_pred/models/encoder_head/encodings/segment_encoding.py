import math

import torch
from torch import nn

from torch import Tensor


class SegmentEncoding(nn.Module):
    """
    helper Module that adds positional encoding to the token embedding to
    introduce a notion of word order.
    """

    def __init__(self,
                 emb_size: int,
                 dropout: float):
        """init the encoding

        Args:
            emb_size (int): size of the embedding
            dropout (float): dropout rate
            maxlen (int, optional): Maximal sequence length. Defaults to 5000.
        """
        super().__init__()
        self.embedding = nn.parameter.Parameter(torch.randn((1,1,emb_size)), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding: Tensor)->Tensor:
        """Add a positional encoding to the given embedding

        Args:
            token_embedding (Tensor): original embedding

        Returns:
            Tensor: the sum of the embedding and the positional encoding
        """
        return self.dropout(token_embedding + self.embedding)
