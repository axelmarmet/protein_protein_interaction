import os

from pandas import DataFrame

from tqdm import tqdm
import numpy as np

def clean_dataframe(data:DataFrame, dataset_directory:str):
    valid_instances = []
    for i in tqdm(range(data.shape[0])):
        sequence1 = data.iloc[i, 0]
        sequence2 = data.iloc[i, 1]

        if(not os.path.exists(f"{dataset_directory}/{sequence1}")):
            continue

        if(not os.path.exists(f"{dataset_directory}/{sequence2}")):
            continue

        valid_instances.append(i)

    valid_instances = np.array(valid_instances)
    data = data.iloc[valid_instances, :]
    return data.sample(frac=1)