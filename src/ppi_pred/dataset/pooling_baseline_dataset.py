import torch
from torch.utils.data import Dataset


class PoolingDataset(Dataset):
    def __init__(self, data, dataset_directory, pooling_operation):
        self.X = []
        self.y = []

        for i in range(data.shape[0]):
            sequence1 = data.iloc[i, 0]
            sequence2 = data.iloc[i, 1]

            embedding1 = torch.load(f"{dataset_directory}/{sequence1}/embeddings_layer_12_MSA_Transformer.pt",
                                    map_location=torch.device('cpu'))
            embedding2 = torch.load(f"{dataset_directory}/{sequence2}/embeddings_layer_12_MSA_Transformer.pt",
                                    map_location=torch.device('cpu'))

            if(pooling_operation == 'mean'):
                embedding1 = embedding1[0].squeeze().mean(dim=0)
                embedding2 = embedding2[0].squeeze().mean(dim=0)

            if(pooling_operation == 'max'):
                embedding1 = embedding1[0].squeeze().max(dim=0)[0]
                embedding2 = embedding2[0].squeeze().max(dim=0)[0]

            self.X.append(torch.cat([embedding1, embedding2], dim=0))
            self.y.append(int(data.iloc[i, -1]))

        self.input_dim = self.X[-1].size(0)


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])