from argparse import Namespace
import torch
from torch import nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from typing import Dict, Any
from torch import Tensor

import argparse

import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from ppi_pred.constants import MAX_LENGTH
from ppi_pred.models.encoder_head.encodings.positional_encoding import SinCosPositionalEncoding, TrainablePositionalEncoding
from ppi_pred.models.encoder_head.encodings.segment_encoding import SegmentEncoding
from ppi_pred.dataset.embedding_seq_dataset import *

from ppi_pred.models.encoder_head.train import train

import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

import json

from ppi_pred.utils import cleanup, set_seed, setup, validate_config

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

        if(config["pos_encoding"] == 'sincos'):
            self.positional_encoding = SinCosPositionalEncoding(embedding_dim, dropout=config["dropout"], 
                                                                maxlen=MAX_LENGTH)
        elif(config["pos_encoding"] == 'trainable'):
            self.positional_encoding = TrainablePositionalEncoding(embedding_dim, dropout=config["dropout"], 
                                                                   maxlen=MAX_LENGTH)



        self.fst_seq_encoding = SegmentEncoding(embedding_dim, dropout=config["dropout"])
        self.snd_seq_encoding = SegmentEncoding(embedding_dim, dropout=config["dropout"])

        self.cls_embedding = nn.parameter.Parameter(
            torch.randn(embedding_dim), requires_grad=True)
        self.sep_embedding = nn.parameter.Parameter(
            torch.randn(embedding_dim), requires_grad=True)

    def forward(self, inp:EmbeddingSeqInput):
        """ forward iteration
        """
        seq = inp.seq
        seq[inp.cls_mask] = self.cls_embedding
        seq[inp.sep_mask] = self.sep_embedding

        device = seq.device

        fst_segment = seq[inp.segment1_mask]
        snd_segment = seq[inp.segment2_mask]

        fst_segment = self.fst_seq_encoding(fst_segment)
        snd_segment = self.snd_seq_encoding(snd_segment)

        encoder_input = torch.zeros(seq.shape)
        encoder_input[inp.segment1_mask] = fst_segment
        encoder_input[inp.segment2_mask] = snd_segment

        encoder_input = encoder_input.transpose(0, 1)
        encoder_input = self.positional_encoding(encoder_input)

        cls_token_embedding = self.encoder.forward(encoder_input, src_key_padding_mask=inp.padding_mask)[0,:]

        logits = self.classifier(cls_token_embedding)

        if self.training:
            return logits
        else:
            return torch.sigmoid(logits)

def run(args:Namespace, rank:int, world_size:int):

    if args.distributed:
        setup(rank, world_size)

    assert os.path.exists(args.config), f"file {args.config} does not exist"

    args.world_size = world_size
    args.rank = rank
    args.is_main = rank == 0
    if args.distributed:
        torch.cuda.device(args.rank)
        args.device = rank

    config = json.load(open(args.config, "r"))
    validate_config(config)
    set_seed(config["seed"])

    # get the dataset
    labels_file = os.path.join(args.data_root, "training_set.pkl")
    assert os.path.exists(labels_file), f"file {labels_file} does not exist"
    dataframe = pd.read_pickle(labels_file)
    layer = 12
    dataset = EmbeddingSeqDataset(dataframe, args.data_root, layer, True)

    model = NaivePPILanguageModel(config["architecture"])
    model = model.to(args.device)

    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.rank],
            output_device=args.rank
        )

    best_model = train(model, dataset, config, args)

    if args.is_main:
        torch.save(best_model, "my_best_model")

    if args.distributed:
        cleanup()


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="""
            Train an encoder head on top of the embeddings
        """
    )
    arg_parser.add_argument(
        "--data_root", type=str, required=True, help="""
        The path to the folder containing training_set.pkl and MSA_transformer_embeddings
    """)
    arg_parser.add_argument(
        "--config", type=str, required=True, help="The config file that contains all hyperparameters"
    )
    arg_parser.add_argument(
        "--device", type=str, default="cpu", help="The device on which to run the training (default : cpu)"
    )
    arg_parser.add_argument('--distributed', dest='distributed', action='store_true', help="""
        use distributed training, if set then device must not be specified
    """)
    arg_parser.set_defaults(feature=True)
    args = arg_parser.parse_args()

    assert not (args.distributed and args.device != "cpu"), "flag --distributed cannot be set at the same time that a device is given"

    if args.distributed:
        # check how many GPUs are available
        size = torch.cuda.device_count()

        # spawn that many processes
        processes = []
        mp.set_start_method("spawn")
        for rank in range(size):
            p = mp.Process(target=run, args=(args, rank, size))
            p.start()
            processes.append(p)

        # wait for all processes to be done to finish
        for p in processes:
            p.join()
    else:
        run(args, 0, 1)


