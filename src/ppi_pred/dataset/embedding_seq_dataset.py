import os

from pandas import DataFrame

import torch
from torch.utils.data import Dataset

# for typing
from torch import Tensor
from typing import List, Tuple

EMBEDDING_DIM = 768
MAX_NORM = 2010.1934

class EmbeddingSeqDataset(Dataset):
    def __init__(self, dataframe:DataFrame, dataset_directory:str, layer:int):
        assert layer in [2, 6, 11], f"layer {layer} not in the dataset"
        assert os.path.exists(dataset_directory), f"path {dataset_directory} does not exist"

        self.dataframe = dataframe
        self.layer = layer
        self.dataset_directory = dataset_directory

    def __len__(self):
        return len(self.dataframe)

    def get_embedding_seq(self, seq_name:str):
        path = os.path.join(self.dataset_directory, seq_name, f"embeddings_layer_{self.layer}_MSA_Transformer.pt")
        return torch.load(path, map_location=torch.device('cpu'))

    def __getitem__(self, idx):
        sequence_name_1 = self.dataframe.iloc[idx, 0]
        sequence_name_2 = self.dataframe.iloc[idx, 1]

        embedding_seq_1 = self.get_embedding_seq(sequence_name_1)[2].squeeze(dim=0)
        embedding_seq_2 = self.get_embedding_seq(sequence_name_2)[2].squeeze(dim=0)

        target = self.dataframe.iloc[idx, -1]

        return embedding_seq_1, embedding_seq_2, target

def pad_sequence(sequences:List[Tensor], max_len:int)->Tuple[Tensor,Tensor]:
    padding_masks = []
    padded_sequences = []
    for seq in sequences:
        seq_len = seq.shape[0]
        padding_masks.append(torch.arange(max_len) > seq_len-1)
        padded_sequences.append(torch.cat((seq, torch.zeros(max_len - seq_len, EMBEDDING_DIM))))

    return torch.stack(padded_sequences, dim=1), torch.stack(padding_masks, dim=0)


def collate_fn(batch):
    seq_1_batch, seq_2_batch, tgt_batch = [], [], []
    seq_1_max_len, seq_2_max_len = 0, 0
    for seq_1, seq_2, tgt in batch:
        seq_1_max_len = max(seq_1_max_len, seq_1.shape[0])
        seq_2_max_len = max(seq_2_max_len, seq_2.shape[0])

        seq_1_batch.append(seq_1)
        seq_2_batch.append(seq_2)
        tgt_batch.append(torch.tensor(int(tgt), dtype=torch.float32))

    seq_1_batch, seq_1_padding_mask_batch = pad_sequence(seq_1_batch, seq_1_max_len)
    seq_2_batch, seq_2_padding_mask_batch = pad_sequence(seq_2_batch, seq_2_max_len)
    tgt_batch = torch.stack(tgt_batch, dim=0).unsqueeze(dim=1)

    return seq_1_batch, seq_1_padding_mask_batch, \
           seq_2_batch, seq_2_padding_mask_batch, tgt_batch