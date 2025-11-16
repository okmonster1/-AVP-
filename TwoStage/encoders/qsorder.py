import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'

def qsorder_encode(sequence, weight=0.1, maxlag=3):
    """
    简化 QSOrder：前 20 维 = AA 组成，后 maxlag 维 = 序列顺序因子
    返回 20 + maxlag = 23 维向量
    """
    seq = sequence.replace('-', '')
    length = len(seq)
    if length == 0:
        return np.zeros(23, dtype=float)

    # 1. AA 组成
    comp = np.array([seq.count(aa) / length for aa in AA])

    # 2. 序列顺序因子
    tau = np.zeros(maxlag)
    for lag in range(1, maxlag + 1):
        denominator = length - lag
        if denominator == 0:
            tau[lag - 1] = 0.0
        else:
            tau[lag - 1] = sum(
                (AA.index(seq[i]) - AA.index(seq[i + lag])) ** 2
                for i in range(denominator)
            ) / denominator

    # 3. 合并
    return np.concatenate([comp, tau * weight])