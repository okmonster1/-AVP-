import pandas as pd
import torch
from encoders.distance_pair import distance_pair_encode
from encoders.cksaagp import cksaagp_encode
from encoders.qsorder import qsorder_encode
from encoders.dde import dde_from_file
import numpy as np

def fem_encode(file_path):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()

    names, seqs = [], []
    for i in range(0, len(lines), 2):
        names.append(lines[i].strip())
        seqs.append(lines[i+1].strip())

    # 批量编码
    dp_feat   = np.array([distance_pair_encode(s) for s in seqs])
    cksaagp_feat = np.array([cksaagp_encode(s) for s in seqs])
    qsorder_feat = np.array([qsorder_encode(s) for s in seqs])
    dde_feat  = dde_from_file(file_path)          # 你已经有了

    # 合并
    df = pd.concat([
        pd.DataFrame(dp_feat),
        pd.DataFrame(cksaagp_feat),
        pd.DataFrame(qsorder_feat),
        pd.DataFrame(dde_feat)
    ], axis=1)

    # 补齐到 566 维
    if df.shape[1] < 566:
        df = pd.concat([df, pd.DataFrame(np.zeros((df.shape[0], 566 - df.shape[1])))], axis=1)
    elif df.shape[1] > 566:
        df = df.iloc[:, :566]

    labels = [1 if 'pos' in n else 0 for n in names]
    return torch.tensor(df.values, dtype=torch.float32), torch.tensor(labels)
