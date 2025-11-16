import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'

def cksaagp_encode(sequence, k=2):
    """
    CKSAAGP type 2：k-间隔氨基酸组对频率（k=2）
    返回 400 维向量
    """
    seq = sequence.replace('-', '')
    length = len(seq)
    if length < k + 2:
        return np.zeros(400, dtype=float)

    feat = np.zeros(400)
    for i in range(length - k - 1):
        a1 = AA.index(seq[i])
        a2 = AA.index(seq[i + k + 1])
        feat[a1 * 20 + a2] += 1
    return feat / (length - k - 1) if (length - k - 1) != 0 else feat