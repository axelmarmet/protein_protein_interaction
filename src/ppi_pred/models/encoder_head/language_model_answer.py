from argparse import Namespace
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from typing import Optional
from torch import Tensor

import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from ppi_pred.constants import MAX_LENGTH
from ppi_pred.models.encoder_head.encodings.positional_encoding import PositionalEncoding
from ppi_pred.models.encoder_head.encodings.segment_encoding import SegmentEncoding
from ppi_pred.dataset.embedding_seq_dataset import *

from ppi_pred.models.encoder_head.train import train

import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

class MultiheadAttention(nn.Module):
    r"""
    Lifted from the pytorch library, necessary because having all three projections
    (key, query and value) in one matrix was annoying for TD-fixup weight initialization
    """

    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
        self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
        self.register_parameter('in_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)

        # WARNING, substituted NonDynamicallyQuantizableLinear by Linear, should be fine
        # as we never plan to quantize the network
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):

        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)


    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
            when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is the batch size,
            and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
            key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, N, E)` when ``batch_first=False`` or
          :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is
          the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
        - **attn_output_weights** - Attention output weights of shape :math:`(N, L, S)`, where :math:`N` is the batch
          size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. Only returned
          when ``need_weights=True``.
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, use_separate_proj_weight=True,
            q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight)

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

class EncoderLayer(nn.Module):
    r"""
    Lifted from the pytorch library, necessary because I remove the layer normalization
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(EncoderLayer, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)

        # Implementation of Feedforward model
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, **factory_kwargs),
            nn.ReLU(),
            self.dropout,
            nn.Linear(dim_feedforward, d_model, **factory_kwargs),
            self.dropout
        )

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(EncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        x = x + self._sa_block(x, src_mask, src_key_padding_mask)
        x = x + self.ff(x)

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout(x)


def dt_fixup(model, max_norm, num_layers: int):
    """
    Taken from https://github.com/BorealisAI/DT-Fixup

    Paper link https://arxiv.org/pdf/2012.15355.pdf
    """

    def t_fix(params:nn.Parameter, scale):
        for p in params:
            if len(p.data.size()) > 1:
                p.data.div_(scale)

    def is_tfix_params(name):
        if "encoder.layers" in name:
            if "self_attn" in name:
                if "v_proj_weight" in name:
                    return True
                if "out_proj" in name:
                    return True
            elif "ff" in name:
                return True
        return False

    dtfix_params = [p for n, p in model.named_parameters() if is_tfix_params(n)]

    # t_fix divides instead of multiplying, so add a -1 exponent and simplify
    factor = (num_layers ** 0.5) * 2 * max_norm
    t_fix(dtfix_params, factor)

class PPILanguageModel(nn.Module):

    def __init__(self, embedding_dim, num_layers, max_input_norm):
        super(PPILanguageModel, self).__init__()

        self.embedding_dim = embedding_dim

        encoder_layer = EncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=1024,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers
        )

        self.classifier = nn.Linear(embedding_dim, 1)

        self.positional_encoding = PositionalEncoding(
            embedding_dim, dropout=0.1, maxlen=MAX_LENGTH)
        self.fst_seq_encoding = SegmentEncoding(embedding_dim, dropout=0.1)
        self.snd_seq_encoding = SegmentEncoding(embedding_dim, dropout=0.1)

        self.cls_embedding = nn.parameter.Parameter(
            torch.randn(embedding_dim), requires_grad=True)
        self.sep_embedding = nn.parameter.Parameter(
            torch.randn(embedding_dim), requires_grad=True)

        dt_fixup(self, max_input_norm, num_layers)

    def forward(self, fst_seq: Tensor, fst_seq_pad_mask: Tensor,
                  snd_seq: Tensor, snd_seq_pad_mask: Tensor):
        """ forward iteration
        Args:
            fst_seq (Tensor): shape  `seq_1_len x batch_dim x embedding_dim`
            fst_seq_pad_mask (Tensor): shape  `batch_dim x seq_1_len`
            snd_seq (Tensor): shape `seq_2_len x batch_dim x embedding_dim`
            snd_seq_pad_mask (Tensor): shape  `batch_dim x seq_2_len`
        """

        # validate a bit the inputs
        fst_seq_len, batch_dim, embedding_dim = fst_seq.shape
        snd_seq_len, batch_dim_, embedding_dim_ = snd_seq.shape
        batch_dim__, fst_seq_len_ = fst_seq_pad_mask.shape
        batch_dim___, snd_seq_len_ = snd_seq_pad_mask.shape

        assert batch_dim == batch_dim_ and \
               batch_dim_ == batch_dim__ and \
               batch_dim__ == batch_dim___, "batch dim of the two sequences are not the same"
        assert fst_seq_len == fst_seq_len_, "seq and padding do not have the same seq length"
        assert snd_seq_len == snd_seq_len_, "seq and padding do not have the same seq length"
        assert embedding_dim == embedding_dim_, "embedding dim of the two sequences are not the same"

        device = fst_seq.device

        fst_segment = torch.cat([
            self.cls_embedding.tile((1, batch_dim, 1)),
            fst_seq,
            self.sep_embedding.tile((1, batch_dim, 1))
        ], 0)

        snd_segment = torch.cat([
            snd_seq,
            self.sep_embedding.tile((1, batch_dim, 1))
        ], 0)

        fst_segment = self.fst_seq_encoding(fst_segment)
        snd_segment = self.snd_seq_encoding(snd_segment)

        encoder_input = torch.cat([fst_segment, snd_segment], 0)
        padding_mask = torch.cat([
            torch.zeros((batch_dim, 1), dtype=torch.bool, device=device), # for cls token
            fst_seq_pad_mask,
            torch.zeros((batch_dim, 1), dtype=torch.bool, device=device), # for sep token
            snd_seq_pad_mask,
            torch.zeros((batch_dim, 1), dtype=torch.bool, device=device)  # for sep token
        ], dim=1)

        encoder_input = self.positional_encoding(encoder_input)

        cls_token_embedding = self.encoder(encoder_input, src_key_padding_mask=padding_mask)[0,:]
        print(f"mean : {cls_token_embedding.mean()}")
        print(f"std : {cls_token_embedding.std()}")
        print(f"classifier_weights mean: {self.classifier.weight.mean()}" )
        print(f"classifier_weights std: {self.classifier.weight.std()}" )
        logits = self.classifier(cls_token_embedding)

        return torch.sigmoid(logits)

args = Namespace(device="cuda:1")

labels_file = "data/training_set.pkl"
dataframe = pd.read_pickle(labels_file)
dataset_path = "data/MSA_transformer_embeddings"
layer = 11
dataset = EmbeddingSeqDataset(dataframe, dataset_path, layer)
ratios = [29675, 3709, 3709]

training_set, validation_set, testing_set = random_split(dataset, ratios)

model = PPILanguageModel(EMBEDDING_DIM, 2, MAX_NORM).to(args.device)

dataloaders = {}
dataloaders['train'] = DataLoader(training_set, batch_size=64, shuffle=True, collate_fn=collate_fn)
dataloaders['val'] = DataLoader(validation_set, batch_size=64, shuffle=True, collate_fn=collate_fn)
dataloaders['test'] = DataLoader(testing_set, batch_size=64, shuffle=True, collate_fn=collate_fn)

train(dataloaders, model, args)