from scipy.sparse import csr_matrix, vstack
import numpy as np
import tensorflow as tf
import operator


def pointwise_mult(a1, a2):
    if isinstance(a1, csr_matrix):
        m = a1.multiply(a2)
        return m.tocsr()
    else:
        return a1 * a2


def mask_mult(s_list, mask):
    mask_ext = np.repeat(mask, len(s_list[0]))
    mask_ext = np.reshape(mask_ext, (len(mask), len(s_list[0])))
    res = np.asarray(s_list) * mask_ext
    return res


def tensor_mask_mult(s_tensor, mask_tensor):
    mask_ext = tf.repeat(mask_tensor, s_tensor.shape[1])
    mask_ext = np.reshape(mask_ext, (mask_tensor.shape[0], s_tensor.shape[1]))
    res = s_tensor * mask_ext
    return res


def avg_mask_mult(s_list, mask):
    res = mask_mult(s_list, mask)
    avg = np.sum(res, axis=0) / np.sum(mask) if np.sum(mask) > 0 else np.sum(res, axis=0)
    return avg


def avg_vec_mask(vec, mask):
    avg = np.sum(vec, axis=0) / np.sum(mask) if np.sum(mask) > 0 else np.sum(vec, axis=0)
    return avg


def avg_emb_vec(id_vec, emb_matrix):
    if len(id_vec) > 0:
        emb_vec = np.asarray([emb_matrix[id] for id in id_vec])
        avg = np.mean(emb_vec, axis=0)
    else:
        avg = np.zeros(emb_matrix.shape[1])
    return avg


def max_vec_mask(vec, mask):
    new_vec = np.copy(vec)
    new_vec[mask == 0] = -1e9
    max_vec = np.max(new_vec, axis=0)
    return max_vec


def mat_average(mat, axis=0):
    if isinstance(mat, csr_matrix):
        avg_mat = np.asarray(mat.mean(axis=axis))
    else:
        avg_mat = np.average(mat, axis=axis)
        avg_mat = np.reshape(avg_mat, (1,) + avg_mat.shape)
    return avg_mat


def mat_sum(mat, axis=0):
    if isinstance(mat, csr_matrix):
        sum_mat = np.asarray(mat.sum(axis=axis))
    else:
        sum_mat = np.sum(mat, axis=axis)
        sum_mat = np.reshape(sum_mat, (1,) + sum_mat.shape)
    return sum_mat


def matmul(mat1, mat2):
    if isinstance(mat1, csr_matrix):
        m = mat1*mat2
        return m
    else:
        return np.matmul(mat1, mat2)


def compute_norm(vecs):
    norms = np.sqrt(np.sum(pointwise_mult(vecs, vecs), axis=1))
    norms[norms == 0] = 1
    norms = norms.reshape((norms.shape[0], 1))

    return norms


def mat_concat(matrices):
    if isinstance(matrices[0], csr_matrix):
        return vstack(matrices).tocsr()
    else:
        return np.concatenate(matrices)


def list_subtract(l1, l2):
    res = list(map(operator.sub, l1, l2))
    if isinstance(l1, np.ndarray):
        res = np.asarray(res)
    return res


def z_score_normalize_tensor(t):
    mean = tf.math.reduce_mean(t)
    std = tf.math.reduce_std(t)
    new_t = (t - mean) / std
    return new_t


def normalize_tensor(t):
    norm = tf.norm(t)
    return t / norm


def normalize_vec(v):
    shape = v.shape
    v = v.squeeze()
    norm = np.sqrt(np.dot(v, v))
    v = v / norm if norm != 0 else v
    v = np.reshape(v, shape)
    return v


def normalize(m):
    norm = np.sqrt(np.sum(m * m, axis=1))
    norm[norm == 0.0] = 1.0
    return m / norm[:, np.newaxis]


def apply_bias_metric(target_vecs, bias_strength, bias_vecs):
    """
    :param target_vecs: vectors of words
    :param bias_strength:
    :param bias_vecs: the point bow is to be biased towards
    :return: vector of bias weights (per each word in the input)
    """

    # bias_vecs = np.stack(bias_vecs)
    bias_vecs = mat_concat(bias_vecs)

    target_norms = compute_norm(target_vecs)
    bias_norms = compute_norm(bias_vecs)
    norms = matmul(target_norms, bias_norms.T)
    bias_weights = matmul(target_vecs, bias_vecs.T) / norms

    bias_weights = np.max(bias_weights, axis=1)
    bias_weights = np.exp(bias_weights - np.max(bias_weights))
    bias_weights = np.power(bias_weights, bias_strength)
    return np.asarray(bias_weights)


def centroid(vecs):
    """
    :param vecs: list of vectors
    :return: centroid of the vectors
    """
    mat = np.asarray(vecs)
    centroid = np.average(mat, axis=0)
    centroid = normalize_vec(centroid)
    return centroid


def add_vecs(v1, v2):
    v3 = v1+v2
    v3 = normalize_vec(v3)
    return v3


def softmax(scores):
    """
    :param scores : an array of shape instances x classes; instance scores for each class
    :return: an array of shape instances x classes; instance probabilities for each class
    """
    x = scores - np.max(scores)
    return np.exp(x)/np.reshape(np.sum(np.exp(x), axis=1), (scores.shape[0], 1))


def vec_similarity(v1, v2):
    return v1.dot(v2)
