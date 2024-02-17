import torch
import torch.nn as nn
from compound_model import Compound_model
from bilstm import BiLstm


class CPI_model_1024(nn.Module):
    def __init__(self, compound_input_size, protein_input_size, output_size, gat_head, compound_head, decoder_head,
                 num_layers_compound_decoder, num_layers_protein_bilstm, num_layers_decoder, dataset, device):
        super().__init__()
        self.compound_model = Compound_model(compound_input_size, output_size, gat_head, compound_head
                                             , num_layers_compound_decoder, dataset)
        self.protein_model = BiLstm(protein_input_size, num_layers_protein_bilstm)
        decoder = nn.TransformerDecoderLayer(output_size, nhead=decoder_head, batch_first=True, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder, num_layers_decoder)
        self.fc = nn.Sequential(
            nn.Linear(output_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
        self.device = device

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len, device):
        N = len(atom_num)  # batch size
        compound_mask = torch.ones((N, compound_max_len))
        protein_mask = torch.ones((N, protein_max_len))
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 0
            protein_mask[i, :protein_num[i]] = 0
        return compound_mask.to(device), protein_mask.to(device)

    def forward(self, node_feature_rdkit, node_feature_subgraph, edge_indices, edge_attr, compound_len, protein_embedding,
                protein_len, device):
        compound_feature = self.compound_model(node_feature_rdkit, node_feature_subgraph, edge_indices, edge_attr, compound_len,
                                               device)
        protein_feature = self.protein_model(protein_embedding)
        compound_mask, protein_mask = self.make_masks(compound_len, protein_len, compound_len, protein_len, self.device)
        compound_mask = (compound_mask == 1)
        protein_mask = (protein_mask == 1)
        interaction_feature = self.decoder(tgt=compound_feature, memory=protein_feature,
                                           tgt_key_padding_mask=compound_mask, memory_key_padding_mask=protein_mask)
        interaction_feature = torch.mean(interaction_feature, dim=1)
        output = self.fc(interaction_feature)

        return output


class CPI_model_23(nn.Module):
    def __init__(self, compound_input_size, protein_input_size, output_size, gat_head, compound_head, decoder_head,
                 num_layers_compound_decoder, num_layers_protein_bilstm, num_layers_decoder, dataset, device):
        super().__init__()
        self.compound_model = Compound_model(int(compound_input_size / gat_head), int(output_size / gat_head), gat_head,
                                             compound_head
                                             , num_layers_compound_decoder, dataset)
        self.prtein = nn.Sequential(
            nn.Linear(protein_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024)
        )
        self.protein_model = BiLstm(compound_input_size, num_layers_protein_bilstm)
        decoder = nn.TransformerDecoderLayer(compound_input_size, nhead=decoder_head, batch_first=True, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder, num_layers_decoder)
        self.fc = nn.Sequential(
            nn.Linear(compound_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
        self.device = device

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len, device):
        N = len(atom_num)  # batch size
        compound_mask = torch.ones((N, compound_max_len))
        protein_mask = torch.ones((N, protein_max_len))
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 0
            protein_mask[i, :protein_num[i]] = 0
        return compound_mask.to(device), protein_mask.to(device)

    def forward(self, node_feature_rdkit, node_feature_subgraph, edge_indices, compound_len, compound_max_len,
                protein_embedding, protein_len, protein_max_len, device):
        compound_feature = self.compound_model(node_feature_rdkit, node_feature_subgraph, edge_indices, compound_len,
                                               compound_max_len, device)
        protein_embedding_1024 = self.prtein(protein_embedding)

        protein_feature = self.protein_model(protein_embedding_1024)
        compound_mask, protein_mask = self.make_masks(compound_len, protein_len, compound_max_len, protein_max_len,
                                                      self.device)
        compound_mask = (compound_mask == 1)
        protein_mask = (protein_mask == 1)
        interaction_feature = self.decoder(tgt=compound_feature, memory=protein_feature,
                                           tgt_key_padding_mask=compound_mask, memory_key_padding_mask=protein_mask)
        interaction_feature = torch.mean(interaction_feature, dim=1)
        output = self.fc(interaction_feature)

        return output


class CPI_model_768(nn.Module):
    def __init__(self, compound_input_size, protein_input_size, output_size, gat_head, compound_head, decoder_head,
                 num_layers_compound_decoder, num_layers_protein_bilstm, num_layers_decoder, dataset, device):
        super().__init__()
        self.compound_model = Compound_model(compound_input_size, output_size, gat_head, compound_head
                                             , num_layers_compound_decoder, dataset)
        self.protein_model = BiLstm(protein_input_size, num_layers_protein_bilstm)
        decoder = nn.TransformerDecoderLayer(output_size, nhead=decoder_head, dim_feedforward=4*output_size, batch_first=True, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder, num_layers_decoder)
        self.fc = nn.Sequential(
            nn.Linear(output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )
        self.device = device

    def make_masks(self, compound_len, protein_len, device):
        # N = len(atom_num)  # batch size
        compound_mask = torch.zeros((1, compound_len))
        protein_mask = torch.zeros((1, protein_len))
        return compound_mask.to(device), protein_mask.to(device)

    def forward(self, node_feature_rdkit, node_feature_subgraph, edge_indices, edge_attr, compound_len, compound_max_len,
                protein_embedding, protein_len, protein_max_len, device):
        compound_feature = self.compound_model(node_feature_rdkit, node_feature_subgraph, edge_indices, edge_attr, compound_len,
                                               device)
        # print(compound_feature.shape)
        protein_feature = self.protein_model(protein_embedding).unsqueeze(0)
        # print(protein_feature.shape)
        compound_mask, protein_mask = self.make_masks(compound_len, protein_len, self.device)
        compound_mask = (compound_mask == 1)
        protein_mask = (protein_mask == 1)
        interaction_feature = self.decoder(tgt=compound_feature, memory=protein_feature,
                                           tgt_key_padding_mask=compound_mask, memory_key_padding_mask=protein_mask)
        interaction_feature = torch.mean(interaction_feature, dim=1)
        output = self.fc(interaction_feature)

        return output
