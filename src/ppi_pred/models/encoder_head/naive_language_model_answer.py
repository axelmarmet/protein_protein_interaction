from argparse import Namespace
import torch
from torch import nn
import torch.optim as optim

from typing import Dict, Any
from torch import Tensor

import argparse

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

import json

from ppi_pred.utils import set_seed

import wandb

class NaivePPILanguageModel(nn.Module):

    def __init__(self, config:Dict[str, Any]):
        super(NaivePPILanguageModel, self).__init__()

        embedding_dim:int = config["e_dim"]

        self.embedding_dim = embedding_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            norm_first=config["norm_first"]
        )

        encoder_norm = nn.LayerNorm(embedding_dim)

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            config["layers"],
            encoder_norm,
        )

        self.classifier = nn.Linear(embedding_dim, 1)

        self.positional_encoding = PositionalEncoding(
            embedding_dim, dropout=config["dropout"], maxlen=MAX_LENGTH)
        self.fst_seq_encoding = SegmentEncoding(embedding_dim, dropout=config["dropout"])
        self.snd_seq_encoding = SegmentEncoding(embedding_dim, dropout=config["dropout"])

        self.cls_embedding = nn.parameter.Parameter(
            torch.randn(embedding_dim), requires_grad=True)
        self.sep_embedding = nn.parameter.Parameter(
            torch.randn(embedding_dim), requires_grad=True)

    def forward(self, fst_seq: Tensor, fst_seq_pad_mask: Tensor,
                      snd_seq: Tensor, snd_seq_pad_mask: Tensor):
        """ forward iteration
        Args:
            fst_seq (Tensor): shape  `batch_dim x seq_1_len x embedding_dim`
            fst_seq_pad_mask (Tensor): shape  `batch_dim x seq_1_len`
            snd_seq (Tensor): shape `batch_dim x seq_2_len x embedding_dim`
            snd_seq_pad_mask (Tensor): shape  `batch_dim x seq_2_len`
        """

        # validate a bit the inputs
        batch_dim, fst_seq_len, embedding_dim = fst_seq.shape
        batch_dim_, snd_seq_len, embedding_dim_ = snd_seq.shape
        batch_dim__, fst_seq_len_ = fst_seq_pad_mask.shape
        batch_dim___, snd_seq_len_ = snd_seq_pad_mask.shape

        assert batch_dim == batch_dim_ and \
               batch_dim_ == batch_dim__ and \
               batch_dim__ == batch_dim___, "batch dim of the two sequences are not the same"
        assert fst_seq_len == fst_seq_len_, "seq and padding do not have the same seq length"
        assert snd_seq_len == snd_seq_len_, "seq and padding do not have the same seq length"
        assert embedding_dim == embedding_dim_, "embedding dim of the two sequences are not the same"

        fst_seq = fst_seq.transpose(1,0)
        snd_seq = snd_seq.transpose(1,0)

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

        logits = self.classifier(cls_token_embedding)

        if self.training:
            return logits
        else:
            return torch.sigmoid(logits)

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="""
            Train an encoder head on top of the embeddings
        """
    )

    arg_parser.add_argument(
        "--config", type=str, required=True, help="The config file that contains all hyperparameters"
    )
    arg_parser.add_argument(
        "--device", type=str, default="cuda:0", help="The device on which to run the training (default : \"cuda:0\")"
    )

    args = arg_parser.parse_args()

    assert os.path.exists(args.config), f"file {args.config} does not exist"

    config = json.load(open(args.config, "r"))
    set_seed(config["seed"])

    wandb.init(project="ppi_pred", entity="axelmarmet", config=config)

    # get the dataset
    labels_file = "data/training_set.pkl"
    dataframe = pd.read_pickle(labels_file)
    dataset_path = "data/MSA_transformer_embeddings"
    layer = 11
    dataset = EmbeddingSeqDataset(dataframe, dataset_path, layer, True)

    model = NaivePPILanguageModel(config["architecture"]).to(args.device)

    best_model = train(model, dataset, config["training"], args.device)
    torch.save(best_model, "my_best_model")