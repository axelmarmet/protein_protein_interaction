import os

from pandas import DataFrame

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import torch.nn as nn

# for typing
from torch import Tensor
from typing import List, Tuple

from torch.utils.data.dataset import random_split

@dataclass
class EmbeddingSeqInput:
    seq      : Tensor # shape `B x max_seq_len x e_dim`
    cls_mask : Tensor # shape `B x max_seq_len`
    sep_mask : Tensor # shape `B x max_seq_len`
    segment1_mask : Tensor # shape `B x max_seq_len`
    segment2_mask : Tensor # shape `B x max_seq_len`

    padding_mask : Tensor # shape `B x max_seq_len`

    def to(self, device):
        self.seq = self.seq.to(device)
        self.cls_mask = self.cls_mask.to(device)
        self.sep_mask = self.sep_mask.to(device)
        self.segment1_mask = self.segment1_mask.to(device)
        self.segment2_mask = self.segment2_mask.to(device)
        self.padding_mask = self.padding_mask.to(device)

class EmbeddingSeqDataset(Dataset):

    RATIOS = [411792, 5000, 5000]

    # should be computed dynamically but I'm lazy
    EMBEDDING_DIM = 768
    MAX_NORM = -1

    NUM_NEGATIVE = 383448
    NUM_POSITIVE = 38344

    def __init__(self, dataframe:DataFrame, dataset_directory:str, layer:int, batch_first:bool=False):
        assert layer in [12], f"layer {layer} not in the dataset"
        assert os.path.exists(dataset_directory), f"path {dataset_directory} does not exist"

        self.dataframe = dataframe
        self.layer = layer
        self.dataset_directory = dataset_directory
        self.batch_first = batch_first

    def __len__(self):
        return len(self.dataframe)

    def get_embedding_seq(self, seq_name:str):
        path = os.path.join(self.dataset_directory, seq_name, f"embeddings_layer_{self.layer}_MSA_Transformer.pt")
        return torch.load(path, map_location=torch.device('cpu'))

    def __getitem__(self, idx):
        sequence_name_1 = str(self.dataframe.iloc[idx, 0])
        sequence_name_2 = str(self.dataframe.iloc[idx, 1])

        embedding_seq_1 = self.get_embedding_seq(sequence_name_1)[0].squeeze(dim=0)
        embedding_seq_2 = self.get_embedding_seq(sequence_name_2)[0].squeeze(dim=0)

        target = int(self.dataframe.iloc[idx, -1])

        return embedding_seq_1, embedding_seq_2, target

    def collate_fn(self, batch)->Tuple[EmbeddingSeqInput, Tensor]:
        seq_list, cls_mask_list, sep_mask_list, segment1_mask_list, segment2_mask_list, tgt_list = [], [], [], [], [], []
        max_seq_len = -1
        for seq_1, seq_2, tgt in batch:
            seq_1_len = seq_1.shape[0]
            seq_2_len = seq_2.shape[0]
            seq_len = seq_1_len + seq_2_len + 3

            cls_mask = torch.zeros(seq_len, dtype=torch.bool)
            cls_mask[0] = 1

            sep_mask = torch.zeros(seq_len, dtype=torch.bool)
            sep_mask[seq_1_len+1] = 1
            sep_mask[-1] = 1

            segment1_mask = torch.zeros(seq_len, dtype=torch.bool)
            segment1_mask[torch.arange(0, seq_1_len + 2)] = 1

            segment2_mask = torch.zeros(seq_len, dtype=torch.bool)
            segment2_mask[torch.arange(2 + seq_1_len, seq_1_len + seq_2_len + 3)] = 1

            assert(torch.all(torch.logical_or(segment1_mask, segment2_mask))), "Problem with masks"

            seq = torch.cat([
                torch.zeros(1, self.EMBEDDING_DIM),
                seq_1,
                torch.zeros(1, self.EMBEDDING_DIM),
                seq_2,
                torch.zeros(1, self.EMBEDDING_DIM)
            ], dim=0)

            max_seq_len = max(max_seq_len, seq_len)

            seq_list.append(seq)
            cls_mask_list.append(cls_mask)
            sep_mask_list.append(sep_mask)
            segment1_mask_list.append(segment1_mask)
            segment2_mask_list.append(segment2_mask)
            tgt_list.append(torch.tensor([tgt]))

        padding_masks = []
        padded_sequences = []
        padded_cls_mask = []
        padded_sep_mask = []
        padded_segment1_mask = []
        padded_segment2_mask = []
        for seq, cls_mask, sep_mask, seg1_mask, seg2_mask in zip(seq_list, cls_mask_list, sep_mask_list,
                                                                 segment1_mask_list, segment2_mask_list):
            seq_len:int = seq.shape[0]
            padding_length = max_seq_len - seq_len

            padding_masks.append(torch.arange(max_seq_len) > seq_len-1)
            padded_sequences.append(
                torch.cat([seq, torch.zeros(padding_length, self.EMBEDDING_DIM)], dim=0)
            )
            padded_cls_mask.append(
                torch.cat([cls_mask, torch.zeros(padding_length, dtype=torch.bool)], dim=0)
            )
            padded_sep_mask.append(
                torch.cat([sep_mask, torch.zeros(padding_length, dtype=torch.bool)], dim=0)
            )
            padded_segment1_mask.append(
                torch.cat([seg1_mask, torch.zeros(padding_length, dtype=torch.bool)], dim=0)
            )
            padded_segment2_mask.append(
                torch.cat([seg2_mask, torch.zeros(padding_length, dtype=torch.bool)], dim=0)
            )

        tgt_batch = torch.stack(tgt_list)

        return EmbeddingSeqInput(
            seq=torch.stack(padded_sequences),
            cls_mask=torch.stack(padded_cls_mask),
            sep_mask=torch.stack(padded_sep_mask),
            padding_mask=torch.stack(padding_masks),
            segment1_mask=torch.stack(padded_segment1_mask),
            segment2_mask=torch.stack(padded_segment2_mask)), tgt_batch


    def split(self):
        return random_split(self, self.RATIOS, generator=torch.Generator().manual_seed(0))


    def get_padding_mask(self, lengths:List[int], max_len : int)->Tensor:
        padding_masks = [torch.arange(max_len) > seq_len-1 for seq_len in lengths]
        return torch.stack(padding_masks)

    def pad_sequences(self, sequences:List[Tensor], cls_mask, max_len:int)->Tensor:
        padding_masks = []
        padded_sequences = []
        for seq in sequences:
            seq_len = seq.shape[0]
            padding_masks.append(torch.arange(max_len) > seq_len-1)
            padded_sequences.append(torch.cat((seq, torch.zeros(max_len - seq_len, self.EMBEDDING_DIM))))

        return torch.stack(padded_sequences, dim=0), torch.stack(padding_masks, dim=0)