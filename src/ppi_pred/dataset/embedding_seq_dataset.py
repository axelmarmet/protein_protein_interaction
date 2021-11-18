import os

from pandas import DataFrame

import torch
from torch.utils.data import Dataset

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

        embedding_seq_1 = self.get_embedding_seq(sequence_name_1)
        embedding_seq_2 = self.get_embedding_seq(sequence_name_2)

        target = self.dataframe.iloc[idx, -1]

        return embedding_seq_1, embedding_seq_2, target