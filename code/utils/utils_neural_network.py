import numpy as np
import math
import string
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow_addons.activations import sparsemax
from tensorflow_addons.layers import MultiHeadAttention
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.losses import Reduction
from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import softmax, sigmoid, relu
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

from utils.utils_metrics import custom_att_metrics_dict

_CHR_IDX = string.ascii_lowercase


# ---------------- Callback --------------
class TestCallback(Callback):
    def __init__(self, test_data, num_class):
        super().__init__()
        self.X_test = test_data[0]
        self.y_test = test_data[1]
        self.num_class = num_class

    def on_train_begin(self, logs=None):
        self.acc_list, self.f1_list = [], []
        return

    def on_epoch_end(self, epoch, logs=None):
        y_prob = self.model.predict(self.X_test)
        y_pred = y_prob > 0.5 if self.num_class == 2 else np.argmax(y_prob, axis=1)

        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='binary' if self.num_class == 2 else 'macro')

        self.acc_list.append(acc)
        self.f1_list.append(f1)
        return


class AttCallback(Callback):
    def __init__(self, data, num_class):
        super().__init__()
        self.X = data[0]
        self.y_inst, self.y_att = data[1]
        self.num_class = num_class

    def on_train_begin(self, logs=None):
        self.acc_list, self.f1_list = [], []
        self.att_dict_list = []
        return

    def on_epoch_end(self, epoch, logs=None):
        y_pred, y_att_pred = self.model.predict_label(self.X)

        acc = accuracy_score(self.y_inst, y_pred)
        f1 = f1_score(self.y_inst, y_pred, average='binary' if self.num_class == 2 else 'macro')
        att_metrics_dict = custom_att_metrics_dict(self.y_att, y_att_pred)

        self.acc_list.append(acc)
        self.f1_list.append(f1)
        self.att_dict_list.append(att_metrics_dict)
        return


# class IGTestCallback(Callback):
#     def __init__(self, test_data):
#         super().__init__()
#         self.X_test = test_data[0]
#         self.y_inst_test, self.y_att_test = test_data[1]
#
#     def on_train_begin(self, logs=None):
#         self.test_acc = []
#         self.test_att_zero, self.test_att_normal = [], []
#         return
#
#     def on_epoch_end(self, epoch, logs=None):
#         y_prob_test = self.model.predict(self.X_test)
#         y_pred_test = (y_prob_test > 0.5)
#         y_att_prob_test = self.model.get_ig_saliency(self.X_test, y_pred_test)
#
#         acc = accuracy_score(self.y_inst_test, y_pred_test)
#         att_zero, att_normal = custom_group_score(self.y_att_test, y_att_prob_test > 0)
#
#         self.test_acc.append(acc)
#         self.test_att_zero.append(att_zero), self.test_att_normal.append(att_normal)
#         return


class CLF_Fea_Callback(Callback):
    def __init__(self, train_data, test_data):
        super().__init__()
        self.X_train, self.y_att_train = train_data
        self.X_test, self.y_att_test = test_data

    def on_train_begin(self, logs=None):
        self.train_att_dict_list = []
        self.test_att_dict_list = []
        return

    def on_epoch_end(self, epoch, logs=None):
        # Train
        y_att_train_prob = self.model.predict(self.X_train)
        y_att_train_pred = y_att_train_prob > 0.5

        train_att_metrics_dict = custom_att_metrics_dict(self.y_att_train, y_att_train_pred)
        self.train_att_dict_list.append(train_att_metrics_dict)

        # Test
        y_att_test_prob = self.model.predict(self.X_test)
        y_att_test_pred = y_att_test_prob > 0.5

        test_att_metrics_dict = custom_att_metrics_dict(self.y_att_test, y_att_test_pred)
        self.test_att_dict_list.append(test_att_metrics_dict)
        return


class CLF_Fea_MC_Callback(Callback):
    def __init__(self, data):
        super().__init__()
        self.X, self.y_att = data

    def on_train_begin(self, logs=None):
        self.att_dict_list = []
        return

    def on_epoch_end(self, epoch, logs=None):
        y_att_prob = self.model.predict(self.X)
        y_att_pred = np.argmax(y_att_prob, axis=-1)

        att_w_P = precision_score(y_true=self.y_att, y_pred=y_att_pred, average='weighted')
        att_macro_P = precision_score(y_true=self.y_att, y_pred=y_att_pred, average='macro')

        att_w_R = recall_score(y_true=self.y_att, y_pred=y_att_pred, average='weighted')
        att_macro_R = recall_score(y_true=self.y_att, y_pred=y_att_pred, average='macro')

        att_w_f1 = f1_score(y_true=self.y_att, y_pred=y_att_pred, average='weighted')
        att_macro_f1 = f1_score(y_true=self.y_att, y_pred=y_att_pred, average='macro')

        att_metrics_dict = {'att_w_P': att_w_P, 'att_macro_P': att_macro_P,
                            'att_w_R': att_w_R, 'att_macro_R': att_macro_R,
                            'att_w_f1': att_w_f1, 'att_macro_f1': att_macro_f1}
        self.att_dict_list.append(att_metrics_dict)
        return


# ---------------- Layer --------------
class AttLayer(Layer):
    def __init__(self, attention_dim, act_func):
        self.supports_masking = True
        self.attention_dim = attention_dim
        self.act_func = act_func
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1], self.attention_dim),
                                 initializer='normal', trainable=True, name='W')
        self.b = self.add_weight(shape=(self.attention_dim, ),
                                 initializer='normal', trainable=True, name='b')
        self.u = self.add_weight(shape=(self.attention_dim, 1),
                                 initializer='normal', trainable=True, name='u')

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = tf.tanh(tf.nn.bias_add(tf.matmul(x, self.W), self.b))
        ait = tf.matmul(uit, self.u)
        ait = tf.squeeze(ait, -1)

        # ait = tf.exp(ait)

        if mask is not None:
            ait *= tf.cast(mask, tf.float32)

        if self.act_func == 'sparsemax':
            ait = sparsemax(ait)

        else:
            ait = softmax(ait, axis=-1)
            # ait /= tf.cast(tf.reduce_sum(ait, axis=1, keepdims=True) + K.epsilon(), tf.float32)

        # att = (batch_size, num_fea, hidden_dim)
        reshape_ait = tf.expand_dims(ait, axis=-1)
        reshape_ait = tf.repeat(reshape_ait, tf.shape(x)[-1])
        reshape_ait = tf.reshape(reshape_ait, (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[-1]))

        # Weighted hidden states
        weighted_input = x * reshape_ait
        output = tf.reduce_sum(weighted_input, axis=1)

        return ait, output


class SparseMultiHeadAttention(MultiHeadAttention):
    """Custom MultiHeadAttention layer.
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
    """
    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        """Applies Dot-product attention with query, key, value tensors.
        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for
        customized attention implementation.
        Args:
          query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
          key: Projected key `Tensor` of shape `(B, S, N, key_dim)`.
          value: Projected value `Tensor` of shape `(B, S, N, value_dim)`.
          attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions. It is generally not needed if the
            `query` and `value` (and/or `key`) are masked.
          training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).
        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
        """
        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.
        query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = tf.einsum(self._dot_product_equation, key, query)

        attention_scores = sparsemax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )

        # `context_layer` = [B, T, N, H]
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores


# ---------------- Activation --------------
def select_att_act(att_out_act):
    if att_out_act == 'sign':
        return custom_sign_binary_prob
    elif att_out_act == 'sparsemax':
        return custom_sparse_binary_prob
    elif att_out_act == 'relu':
        return relu
    else:
        return sigmoid


def gumbel_softmax(logit, temperature=1.0, training=True):
    """
    Multi-label learning version
    """
    if not training:
        # Return hard mask
        x = tf.cast(logit > 0, dtype=tf.float32)
        return x

    complete_logit = tf.stack([-logit, logit], axis=-1)
    noise = tf.random.uniform(tf.shape(complete_logit), minval=0, maxval=1)
    noise = -tf.math.log(-tf.math.log(noise))

    x = (complete_logit + noise) / temperature
    x = tf.nn.softmax(x, axis=-1)
    x = tf.squeeze(x[:, :, 1])
    return x


def gumbel_softmax_MultiC(logit, temperature=1.0, training=True):
    """
    Multi-class learning version
    """
    if not training:
        # Return hard mask
        x_argmax = tf.argmax(logit, axis=-1)
        x = tf.one_hot(x_argmax, depth=tf.shape(logit)[-1])
        return x

    # Return soft mask
    noise = tf.random.uniform(tf.shape(logit), minval=0, maxval=1)
    noise = -tf.math.log(-tf.math.log(noise))

    x = (logit + noise) / temperature
    x = tf.nn.softmax(x, axis=-1)
    return x


# ---------------- Loss --------------
def select_att_loss(att_loss_func, y_att, y_att_value):
    if att_loss_func == 'sparsemax':
        return custom_sparse_loss(y_att, y_att_value)
    elif att_loss_func == 'sign':
        return custom_sign_loss(y_att, y_att_value)
    elif att_loss_func == 'mse':
        clip_t = tf.clip_by_value(y_att_value, clip_value_min=0., clip_value_max=1.)
        clip_y_att_value = y_att_value + tf.stop_gradient(clip_t - y_att_value)
        return MeanSquaredError()(y_true=y_att, y_pred=clip_y_att_value)
    elif att_loss_func == 'relu_bce':
        return custom_relu_crossentropy(y_att, y_att_value)
    else:
        return BinaryCrossentropy(from_logits=True)(y_true=y_att, y_pred=y_att_value)


def custom_relu_crossentropy(y_true, pos_logits):
    prob = sigmoid(pos_logits)
    epsilon = tf.keras.backend.epsilon()
    epsilon_ = tf.constant(tf.keras.backend.epsilon(), dtype=tf.float32)

    output = tf.clip_by_value(prob, epsilon_, 1. - epsilon_)

    # Compute cross entropy from probabilities.
    bce = y_true * tf.math.log(output + epsilon - 0.5)
    bce += (1 - y_true) * tf.math.log(1 - output + epsilon)
    mean_fea_bce = tf.reduce_sum(-bce, axis=1)
    mean_bce = tf.reduce_mean(mean_fea_bce)
    return mean_bce


def custom_sparse_binary_prob(pos_logits):
    logits = tf.stack([-pos_logits, pos_logits], axis=-1)
    sparse_fea_prob = sparsemax(logits)
    sparse_y_prob = sparse_fea_prob[:, :, 1]
    return sparse_y_prob


def custom_sign_binary_prob(pos_logits):
    prob = (tf.sign(pos_logits) + 1) / 2
    return prob


def custom_sparse_loss(y_pos_true, pos_logits):
    logits = tf.stack([-pos_logits, pos_logits], axis=-1)
    # logits = pos_logits

    sparse_fea_prob = sparsemax(logits)

    y_true = tf.stack([1-y_pos_true, y_pos_true], axis=-1)
    # y_true = y_pos_true

    logits = tf.convert_to_tensor(logits, name="logits")
    sparse_fea_prob = tf.convert_to_tensor(sparse_fea_prob, name="sparsemax")
    labels = tf.convert_to_tensor(y_true, name="labels")

    # In the paper, they call the logits z.
    # A constant can be substracted from logits to make the algorithm
    # more numerically stable in theory. However, there are really no major
    # source numerical instability in this algorithm.
    z = logits

    # sum over support
    # Use a conditional where instead of a multiplication to support z = -inf.
    # If z = -inf, and there is no support (sparsemax = 0), a multiplication
    # would cause 0 * -inf = nan, which is not correct in this case.
    sum_s = tf.where(
        tf.math.logical_or(sparse_fea_prob > 0, tf.math.is_nan(sparse_fea_prob)),
        sparse_fea_prob * (z - 0.5 * sparse_fea_prob),
        tf.zeros_like(sparse_fea_prob),
    )

    # - z_k + ||q||^2
    q_part = labels * (0.5 * labels - z)
    # Fix the case where labels = 0 and z = -inf, where q_part would
    # otherwise be 0 * -inf = nan. But since the lables = 0, no cost for
    # z = -inf should be consideredself.
    # The code below also coveres the case where z = inf. Howeverm in this
    # caose the sparsemax will be nan, which means the sum_s will also be nan,
    # therefor this case doesn't need addtional special treatment.
    q_part_safe = tf.where(
        tf.math.logical_and(tf.math.equal(labels, 0), tf.math.is_inf(z)),
        tf.zeros_like(z),
        q_part,
    )

    loss = tf.math.reduce_sum(sum_s + q_part_safe, axis=-1)
    loss = tf.math.reduce_sum(loss, axis=-1)
    loss = tf.reduce_mean(loss)
    return loss


def custom_sign_loss(y_pos_true, pos_logits):
    logits = tf.convert_to_tensor(pos_logits, name="logits")
    labels = tf.convert_to_tensor(y_pos_true, name="labels")

    L = -tf.multiply(labels, logits) + 0.5 * tf.abs(logits) + 0.5 * logits
    loss = tf.reduce_sum(L, axis=-1)
    loss = tf.reduce_mean(loss)

    return loss


def custom_sigmoid_focal_crossentropy(y_true, y_pred,
                                      reduction=Reduction.SUM_OVER_BATCH_SIZE, alpha=None, gamma=2.0, from_logits=True):
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    # Get the cross_entropy for each entry
    ce = BinaryCrossentropy(from_logits=from_logits)(y_true, y_pred)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha is not None:
        alpha = tf.cast(alpha, dtype=y_true.dtype)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        alpha_factor = tf.where(alpha_factor == 0, 1, alpha_factor)
        alpha_factor = tf.where(alpha_factor == 1, tf.reduce_min(alpha_factor), alpha_factor)

    if gamma:
        gamma = tf.cast(gamma, dtype=y_true.dtype)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    loss = tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)
    if reduction == Reduction.SUM_OVER_BATCH_SIZE:
        reduction_loss = tf.reduce_mean(loss)
    elif reduction == Reduction.SUM:
        reduction_loss = tf.reduce_sum(loss)
    else:
        reduction_loss = loss
    return reduction_loss
