import json
import os
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm

import argparse
from argparse import Namespace
from ppi_pred.dataset.embedding_seq_dataset import EmbeddingSeqDataset
from ppi_pred.metrics import metrics, print_metrics
from ppi_pred.models.encoder_head.encoder_head import NaivePPILanguageModel
from ppi_pred.models.encoder_head.train import test
from ppi_pred.utils import validate_config


if __name__ == '__main__':


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
    arg_parser.add_argument(
        "--saved_state_dict", type=str, required=True, help="""
            the file of the saved state
        """
    )

    args = arg_parser.parse_args()

    assert os.path.exists(args.config), f"file {args.config} does not exist"
    assert os.path.exists(args.saved_state_dict), f"file {args.saved_state_dict} does not exist"

    labels_file = os.path.join(args.data_root, "test_set.pkl")
    assert os.path.exists(labels_file), f"file {labels_file} does not exist"
    embeddings_dir = os.path.join(args.data_root, "MSA_transformer_embeddings")
    assert os.path.exists(embeddings_dir), f"directory {embeddings_dir} does not exist"
    dataframe = pd.read_pickle(labels_file)
    layer = 12
    dataset = EmbeddingSeqDataset(dataframe, embeddings_dir, layer, True)

    dataloader = DataLoader(dataset, 32, collate_fn=dataset.collate_fn, pin_memory=True)

    config = json.load(open(args.config, "r"))
    validate_config(config)

    model = NaivePPILanguageModel(config["architecture"])
    model.load_state_dict(torch.load(args.saved_state_dict))
    model = model.to(args.device)

    print_metrics(
        test(dataloader, model, args, True)
    )


