import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class Compound_model(nn.Module):
    def __init__(self, input_size, output_size, gat_head,compound_head, num_layers,dataset): # 78->64*4->768
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gat_head = gat_head
        self.compound_head = compound_head
        self.fc = nn.Linear(input_size*gat_head, output_size)
        self.embed_fingerprint = nn.Embedding(62780, input_size*gat_head)
        self.transformerconv = TransformerConv(110, input_size, gat_head, edge_dim=13)
        self.fc_1 = nn.Linear(input_size*gat_head, output_size)
        decoder = nn.TransformerDecoderLayer(output_size, nhead=compound_head, dim_feedforward=4*output_size, batch_first=True,dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder, num_layers)

    def make_masks(self, batch_size, compound_max_len, device):
        compound_mask = torch.zeros((batch_size, compound_max_len))
        return compound_mask.to(device)

    # def batch_concat(self, batch_feature, batch_edge, max_len,device):
    #     batch = len(batch_feature)
    #     batch_feature = torch.flatten(batch_feature, start_dim=0, end_dim=1)
    #     edge = []
    #     for i in range(batch):
    #         for e in batch_edge[i]:
    #             # print(e)
    #             e = torch.tensor(e).to(device)
    #             edge.append([e[0]+i*max_len, e[1]+i*max_len])
    #     edge = torch.IntTensor(edge).transpose(1,0)
    #     return batch_feature, edge.to(device)

    def forward(self, node_feature_rdkit, node_feature_subgraph, edge_indices, edge_attr, compound_len,device):
        batch_size = 1
        mask = self.make_masks(batch_size, compound_len, device)
        mask = (mask == 1)
        node_feature_rdkit = node_feature_rdkit.to(torch.float32)
        # print(node_feature_subgraph.shape)
        node_feature_subgraph = self.embed_fingerprint(node_feature_subgraph)
        node_feature_subgraph = torch.reshape(node_feature_subgraph, (batch_size, compound_len, self.input_size*self.gat_head))
        node_feature_subgraph = F.relu(self.fc_1(node_feature_subgraph))
        # node_feature_rdkit, edge_indices = self.batch_concat(node_feature_rdkit, edge_indices, compound_max_len, device)
        edge_indices = torch.LongTensor(edge_indices).transpose(1,0).to(device)
        node_feature_rdkit = F.elu(self.transformerconv(node_feature_rdkit, edge_indices, edge_attr))
        node_feature_rdkit = torch.reshape(node_feature_rdkit, (batch_size, compound_len, self.input_size*self.gat_head))
        node_feature_rdkit = F.relu(self.fc(node_feature_rdkit))
        compound_feature = self.decoder(node_feature_subgraph, node_feature_rdkit, tgt_key_padding_mask=mask, memory_key_padding_mask=mask)
        return compound_feature

# compound_model = Compound_model(128, 128, 8, 129)
# with torch.no_grad():
#     x = torch.randn(2, 129, 128)
#     sub_x = torch.randint(low=0, high=10, size=(2, 129,))
#     print(sub_x)
#     x_edge = [[[0,1],[1,0],[0,2],[2,0],[2,3],[3,2]],[[0,1],[1,0],[0,2],[2,0],[2,3]]]
#     x_len = 4
#     x_max_len = 129
#     # compound_model.batch_concat(x, x_edge, x_max_len)
#     # mask = compound_model.make_masks(2, [x_len, x_len], x_max_len)
#     compound_model.forward(x, sub_x, x_edge, [x_len, x_len], x_max_len)