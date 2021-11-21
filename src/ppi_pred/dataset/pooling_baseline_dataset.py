import torch
from torch.utils.data import Dataset


class PoolingDataset(Dataset):
    def __init__(self, data, dataset_directory, pooling_operation, layers=[2, 6, 11]):
        self.X = []
        self.y = []
        self.mapping = {2:0, 6:1, 11:2}

        for i in range(data.shape[0]):
            sequence1 = data.iloc[i, 0]
            sequence2 = data.iloc[i, 1]

            embeddings1 = torch.load(f"{dataset_directory}/{sequence1}/embeddings_layer_2_MSA_Transformer.pt",
                                    map_location=torch.device('cpu'))
            embeddings2 = torch.load(f"{dataset_directory}/{sequence2}/embeddings_layer_2_MSA_Transformer.pt",
                                    map_location=torch.device('cpu'))

            cur_embedding = []
            for l in layers:
                idx = self.mapping[l]
                embedding1, embedding2 = None, None 

                if(pooling_operation == 'mean'):
                    embedding1 = embeddings1[idx].squeeze().mean(dim=0)
                    embedding2 = embeddings2[idx].squeeze().mean(dim=0)

                if(pooling_operation == 'max'):
                    embedding1 = embeddings1[idx].squeeze().max(dim=0)[0]
                    embedding2 = embeddings2[idx].squeeze().max(dim=0)[0]
        
                cur_embedding.append(embedding1)
                cur_embedding.append(embedding2)


            self.X.append(torch.cat(cur_embedding, dim=0))
            self.y.append(int(data.iloc[i, -1]))

        self.input_dim = self.X[-1].size(0)


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])