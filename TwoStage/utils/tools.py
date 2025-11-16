import numpy as np

def pad_sequences(encoded_list):
    """
    将特征列表补零到相同长度
    :param encoded_list: list of np.array, shape (L, feature_dim)
    :return: np.array, shape (batch, max_len, feature_dim)
    """
    max_len = max([arr.shape[0] for arr in encoded_list])
    feature_dim = encoded_list[0].shape[1]
    padded = []
    for arr in encoded_list:
        padded_arr = np.zeros((max_len, feature_dim))
        padded_arr[:arr.shape[0], :] = arr
        padded.append(padded_arr)
    return np.array(padded)