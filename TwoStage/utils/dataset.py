import torch
from torch.utils.data import Dataset

class ContrastDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def collate(self, batch):
        # 简单实现：前半批 vs 后半批 成对
        batch_size = len(batch)
        half = batch_size // 2
        seq1 = torch.stack([batch[i][0] for i in range(half)])
        seq2 = torch.stack([batch[i + half][0] for i in range(half)])
        label1 = torch.tensor([batch[i][1] for i in range(half)])
        label2 = torch.tensor([batch[i + half][1] for i in range(half)])
        binary_label = (label1 == label2).int()
        return seq1, seq2, binary_label, label1, label2