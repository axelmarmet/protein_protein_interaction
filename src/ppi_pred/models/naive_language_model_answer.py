from argparse import Namespace
import torch
from torch import nn
import torch.optim as optim

from torch import Tensor

import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from ppi_pred.constants import MAX_LENGTH
from ppi_pred.models.encodings.positional_encoding import PositionalEncoding
from ppi_pred.models.encodings.segment_encoding import SegmentEncoding

from ppi_pred.dataset.utils import clean_dataframe
from ppi_pred.dataset import embedding_seq_dataset
from ppi_pred.dataset.embedding_seq_dataset import EmbeddingSeqDataset, EMBEDDING_DIM

import copy

from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score



class NaivePPILanguageModel(nn.Module):

    def __init__(self, embedding_dim):
        super(NaivePPILanguageModel, self).__init__()

        self.embedding_dim = embedding_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=1024,
        )
        encoder_norm = nn.LayerNorm(embedding_dim)

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            1,
            encoder_norm
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
            torch.zeros((batch_dim, 1), dtype=torch.bool), # for cls token
            fst_seq_pad_mask,
            torch.zeros((batch_dim, 1), dtype=torch.bool), # for sep token
            snd_seq_pad_mask,
            torch.zeros((batch_dim, 1), dtype=torch.bool)  # for sep token
        ], dim=1)

        encoder_input = self.positional_encoding(encoder_input)

        cls_token_embedding = self.encoder(encoder_input, src_key_padding_mask=padding_mask)[0,:]

        return torch.sigmoid(self.classifier(cls_token_embedding))

def train(dataloaders, args):
    model = NaivePPILanguageModel(embedding_seq_dataset.EMBEDDING_DIM)
    opt = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    best_model = model
    val_max = -np.inf
    for epoch in range(10):
        total_loss = 0
        model.train()
        for seq_1, pad_1, seq_2, pad_2, tgt in tqdm(dataloaders['train']):

            seq_1.to(args.device)
            pad_1.to(args.device)
            seq_2.to(args.device)
            pad_2.to(args.device)
            tgt.to(args.device)

            opt.zero_grad()
            pred = model(seq_1, pad_1, seq_2, pad_2)
            loss = criterion(pred, tgt)
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
        for seq_1, pad_1, seq_2, pad_2, tgt in dataloaders[dataset]:

            seq_1.to(args.device)
            pad_1.to(args.device)
            seq_2.to(args.device)
            pad_2.to(args.device)
            tgt.to(args.device)

            pred = model(seq_1, pad_1, seq_2, pad_2)
            predictions.append(pred.round().cpu().detach().numpy())
            labels.append(tgt.cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accs[dataset] = accuracy_score(labels, predictions)
    return accs


args = Namespace(device="cpu")

labels_file = "data/training_set.pkl"
dataframe = pd.read_pickle(labels_file)
dataset_path = "data/MSA_transformer_embeddings"
layer = 11
dataset = EmbeddingSeqDataset(dataframe, dataset_path, layer)
ratios = [29675, 3709, 3709]

training_set, validation_set, testing_set = random_split(dataset, ratios)

model = NaivePPILanguageModel(embedding_seq_dataset.EMBEDDING_DIM)

dataloaders = {}
dataloaders['train'] = DataLoader(training_set, batch_size=64, shuffle=True, collate_fn=embedding_seq_dataset.collate_fn)
dataloaders['val'] = DataLoader(validation_set, batch_size=64, shuffle=True, collate_fn=embedding_seq_dataset.collate_fn)
dataloaders['test'] = DataLoader(testing_set, batch_size=64, shuffle=True, collate_fn=embedding_seq_dataset.collate_fn)

train(dataloaders, args)
# res = model(fst_seq, snd_seq)

# print(res)