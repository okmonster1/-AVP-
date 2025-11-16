import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter_sizes = [1, 2, 3, 4, 6, 8, 16, 32]
        filter_num = 64

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, 45)) for fsz in self.filter_sizes]
        )
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            input_size=len(self.filter_sizes) * filter_num,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        # x: (batch, max_len, 45)
        x = x.unsqueeze(1)  # (batch, 1, max_len, 45)
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool2d(out, (out.size(2), out.size(3))).squeeze() for out in conv_outs]
        x = torch.cat(pooled, dim=1)  # (batch, len(filter_sizes)*filter_num)

        x = x.unsqueeze(1)  # (batch, 1, features)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # (batch, 256)
        x = self.dropout(x)
        return self.fc(x)  # (batch, 64)

    def classify(self, x):
        features = self.forward(x)
        return nn.Linear(64, 2).to(x.device)(features)