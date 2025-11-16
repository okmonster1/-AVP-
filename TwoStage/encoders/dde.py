import re
import math
import numpy as np
import pandas as pd

AA = 'ACDEFGHIKLMNPQRSTVWY'
diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]

# 密码子使用频率
myCodons = {
    'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2,
    'I': 3, 'K': 2, 'L': 6, 'M': 1, 'N': 2, 'P': 4, 'Q': 2,
    'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
}

# 理论二肽频率
myTM = [(myCodons[p[0]] / 61) * (myCodons[p[1]] / 61) for p in diPeptides]

def dde_encode(sequence):
    sequence = re.sub('-', '', sequence)
    length = len(sequence)
    if length < 2:
        return [0.0] * 400

    # 实际二肽频率
    tmpCode = [0] * 400
    for j in range(length - 1):
        idx = AA.index(sequence[j]) * 20 + AA.index(sequence[j+1])
        tmpCode[idx] += 1
    tmpCode = [i / sum(tmpCode) if sum(tmpCode) != 0 else 0 for i in tmpCode]

    # 方差
    myTV = [myTM[j] * (1 - myTM[j]) / (length - 1) for j in range(400)]

    # 标准化
    dde_feat = [(tmpCode[j] - myTM[j]) / math.sqrt(myTV[j]) if myTV[j] != 0 else 0.0 for j in range(400)]
    return dde_feat

def dde_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    features = []
    for i in range(0, len(lines), 2):
        seq = lines[i+1].strip()
        feat = dde_encode(seq)
        features.append(feat)
    return np.array(features, dtype=np.float32)