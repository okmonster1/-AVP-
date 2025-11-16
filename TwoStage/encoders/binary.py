import numpy as np
def binary_encode(sequence):
    """
    将序列转换为 binary 编码
    :param sequence: str, 如 'ACDEFG'
    :return: np.array, shape (L, 20)
    """
    aa_order = 'ARNDCQEGHILKMFPSTWYV'
    encoding = []
    for aa in sequence:
        vec = [1 if aa == a else 0 for a in aa_order]
        encoding.append(vec)
    return np.array(encoding, dtype=np.float32)