import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data, compound_rdkit_data, compound_sub_data, compound_len_data, edge_rdkit, edge_attr,
                 protein_data,
                 protein_len_data):
        self.compound_rdkit_data = compound_rdkit_data
        self.compound_sub_data = compound_sub_data
        self.compound_len_data = compound_len_data
        self.edge_rdkit = edge_rdkit
        self.edge_attr = edge_attr
        self.protein_data = protein_data
        self.protein_len_data = protein_len_data
        data = pd.read_csv(data)
        data = data.sample(frac=1, random_state=42)
        self.compound_ids = data['compound'].values
        self.protein_ids = data['protein'].values
        self.labels = data['label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        compound_idx = self.compound_ids[idx]
        protein_idx = self.protein_ids[idx]

        compound_rdkit_data = torch.tensor(np.array(self.compound_rdkit_data[compound_idx]), dtype=torch.float32)
        compound_sub_data = torch.tensor(self.compound_sub_data[compound_idx])
        compound_len_data = self.compound_len_data[compound_idx]
        edge_rdkit = self.edge_rdkit[compound_idx]
        edge_attr = torch.tensor(self.edge_attr[compound_idx])

        protein_data = torch.tensor(self.protein_data[protein_idx], dtype=torch.float32)
        protein_len_data = self.protein_len_data[protein_idx]

        label = torch.tensor([self.labels[idx]])

        return {
            'idx': compound_idx,
            'compound_rdkit_data': compound_rdkit_data,
            'compound_sub_data': compound_sub_data,
            'compound_len_data': compound_len_data,
            'edge_rdkit': edge_rdkit,
            'edge_attr': edge_attr,
            'protein_data': protein_data,
            'protein_len_data': protein_len_data,
            'labels': label
        }


def data_loader_chembl(dataset):
    # 示例数据
    compound_rdkit_data = f'{dataset}/chembl_2933_compound_feature_pangu.pkl'
    compound_sub_data = f'{dataset}/chembl_2933_compound_feature_subgraph.pkl'
    compound_len_data = f'{dataset}/chembl_2933_compound_len.pkl'
    edge_rdkit = f'{dataset}/chembl_2933_edge_indices_rdkit.pkl'
    edge_attr = f'{dataset}/chembl_2933_edge_feature_rdkit.pkl'
    protein_data = f'{dataset}/chembl_2933_protein_bert_768.pkl'
    protein_len_data = f'{dataset}/chembl_2933_protein_len.pkl'
    train_data = f'{dataset}/chembl_29_train_idmap.csv'
    dev_data = f'{dataset}/chembl_29_val_idmap.csv'
    test_data = f'{dataset}/chembl_29_test_idmap.csv'

    with open(compound_len_data, 'rb') as file:
        compound_len_data = pickle.load(file)
    with open(protein_len_data, 'rb') as file:
        protein_len_data = pickle.load(file)
    with open(edge_rdkit, 'rb') as file:
        edge_rdkit = pickle.load(file)
    with open(edge_attr, 'rb') as file:
        edge_attr = pickle.load(file)
    with open(compound_rdkit_data, 'rb') as file:
        compound_rdkit_data = pickle.load(file)
    with open(compound_sub_data, 'rb') as file:
        compound_sub_data = pickle.load(file)
    with open(protein_data, 'rb') as file:
        protein_data = pickle.load(file)

    # create dataset
    train_dataset = CustomDataset(train_data, compound_rdkit_data, compound_sub_data, compound_len_data, edge_rdkit,
                                  edge_attr,
                                  protein_data,
                                  protein_len_data)
    dev_dataset = CustomDataset(dev_data, compound_rdkit_data, compound_sub_data, compound_len_data, edge_rdkit,
                                edge_attr,
                                protein_data,
                                protein_len_data)
    test_dataset = CustomDataset(test_data, compound_rdkit_data, compound_sub_data, compound_len_data, edge_rdkit,
                                 edge_attr,
                                 protein_data,
                                 protein_len_data)
    return train_dataset, dev_dataset, test_dataset
