import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'

def distance_pair_encode(sequence):
    """
    简化版 DistancePair：统计相邻氨基酸对出现次数，归一化
    返回 400 维向量
    """
    seq = sequence.replace('-', '')
    length = len(seq)
    if length < 2:
        return np.zeros(400, dtype=float)

    dp = np.zeros(400)
    for i in range(length - 1):
        a1 = AA.index(seq[i])
        a2 = AA.index(seq[i+1])
        dp[a1 * 20 + a2] += 1
    return dp / (length - 1) if (length - 1) != 0 else dp