import numpy as np
import torch
from encoders.binary import binary_encode
from encoders.blosum62 import blosum62_encode
from encoders.zscale import zscale_encode
from utils.tools import pad_sequences

def contrast_encode(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sequences = []
    labels = []
    for i in range(0, len(lines), 2):
        header = lines[i].strip()
        seq = lines[i+1].strip()
        label = 1 if 'pos' in header else 0
        sequences.append(seq)
        labels.append(label)

    encoded = []
    for seq in sequences:
        bin_feat = binary_encode(seq)
        blo_feat = blosum62_encode(seq)
        zsc_feat = zscale_encode(seq)
        combined = np.hstack([bin_feat, blo_feat, zsc_feat])  # (L, 45)
        encoded.append(combined)

    padded = pad_sequences(encoded)  # (batch, max_len, 45)
    return torch.tensor(padded, dtype=torch.float32), torch.tensor(labels)