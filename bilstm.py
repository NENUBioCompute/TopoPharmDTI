import torch
from torch import nn


class BiLstm(nn.Module):
    def __init__(self, hidden, num_layers):
        super().__init__()
        # self.embed_smi = nn.Embedding(size, hidden)
        self.bilstm = nn.LSTM(hidden, hidden, num_layers=num_layers, batch_first=True, dropout=0.1, bidirectional=True)
        self.lin1 = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            # nn.Dropout(0.5),
            # nn.Dropout(0.1),
            nn.PReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, protein):
        # embed = self.embed_smi(data)  # (N,L,128)
        out, (h_n, _) = self.bilstm(protein)
        # out = self.pool(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.lin1(out)

        return out