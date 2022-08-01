import numpy as np
import math

def get_kernel_matrix(B: np.ndarray):
    # note that the definition of B is slightly different from that in the paper: the first dimension is the batch_size and second dimension is feature size
    N = B.shape[0]
    D = B.shape[1]
    # assert D <= N
    L = np.matmul(B, B.transpose())
    return L

def map_inference(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items

def get_posterior_prob(kernel_matrix: np.ndarray, subset: np.ndarray):
    item_size = kernel_matrix.shape[0]
    sample_size = subset.shape[0]
    assert sample_size <= item_size
    sub_matrix = kernel_matrix[subset][:, subset]
    prob = np.linalg.det(sub_matrix) / np.linalg.det(kernel_matrix + np.eye(item_size, dtype=kernel_matrix.dtype)) # a very small number
    prob *= 1e3 # need to be fine-tuned with different problems
    # prob = np.linalg.det(sub_matrix)
    if prob < 1e-8:
        print("Bad sampling!!!")

    return prob

def get_expected_sampling_length(kernel_matrix: np.ndarray):
    values, vectors = np.linalg.eig(kernel_matrix)
    exp_len = np.sum(values / (values + 1.0))
    return exp_len


