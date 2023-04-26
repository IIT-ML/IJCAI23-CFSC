import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.metrics import Mean, Accuracy
from tensorflow.keras.layers import Dense, Bidirectional, LSTM
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam

from utils.utils_metrics import custom_att_metrics_dict
from utils.utils_neural_network import AttLayer, AttCallback


loss_tracker = Mean(name="loss")
att_loss_tracker = Mean(name="att_loss")
acc_tracker = Accuracy(name="acc")
val_loss_tracker = Mean(name="loss")
val_att_loss_tracker = Mean(name="att_loss")
val_acc_tracker = Accuracy(name="acc")


class Att_Classifier(Model):
    """
    BiLSTM classification model where the attentions are supervised.
    """
    def __init__(self, num_class, num_fl, input_dim, input_act, att_dim, lstm_dim, lstm_att_act, regular_w, lambda_att):
        super(Att_Classifier, self).__init__()

        self.num_class = num_class
        self.num_fl = num_fl

        self.input_dim = input_dim
        self.input_act = input_act
        self.att_dim = att_dim
        self.lstm_dim = lstm_dim
        self.lstm_att_act = lstm_att_act
        self.regular_w = regular_w
        self.lambda_att = lambda_att

        num_out_unit = 1 if num_class == 2 else num_class
        self.clf_output_act = 'sigmoid' if self.num_class == 2 else 'softmax'
        self.clf_loss_func = MeanSquaredError()
        self.att_loss_func = MeanSquaredError()

        self.vec_input_layer = Dense(self.input_dim, activation=self.input_act, name='fea_vec_input')
        self.lstm_layer = Bidirectional(LSTM(self.lstm_dim, return_sequences=True), merge_mode='concat')
        self.att_layer = AttLayer(attention_dim=self.att_dim, act_func=self.lstm_att_act)
        self.clf_out_layer = Dense(num_out_unit, activation=self.clf_output_act, name='clf_out',
                                   kernel_regularizer=l1_l2(l1=0, l2=self.regular_w))

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, **kwargs):
        fea_vec_input = inputs  # (batch, num_fl, 1)

        fea_emb = self.vec_input_layer(fea_vec_input)
        lstm_emb = self.lstm_layer(fea_emb)
        z_att, z = self.att_layer(lstm_emb)

        x_output = self.clf_out_layer(z)

        return x_output, z_att

    def get_num_para(self):
        total_parameters = 0
        for variable in self.trainable_variables:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1

            for dim in shape:
                variable_parameters *= dim

            total_parameters += variable_parameters
        return total_parameters

    def gradient(self, x, y, y_att):
        with tf.GradientTape() as tape:
            y_train_prob, y_train_att_prob = self(x, training=True)

            mean_clf_loss = self.clf_loss_func(y_true=y, y_pred=y_train_prob)  # Classification loss
            mean_att_loss = self.att_loss_func(y_true=y_att, y_pred=y_train_att_prob)  # Attention loss

            # MSE to SE
            clf_loss = tf.cast(tf.shape(y_train_prob)[0]*tf.shape(y_train_prob)[1], tf.float32)*mean_clf_loss
            att_loss = tf.cast(tf.shape(y_train_att_prob)[0]*tf.shape(y_train_att_prob)[1], tf.float32)*mean_att_loss

            # Regularization
            reg_loss = tf.add_n(self.losses)

            # Hybrid loss
            hybrid_loss = clf_loss + self.lambda_att * att_loss + reg_loss
        grads = tape.gradient(hybrid_loss, self.trainable_variables)
        return [hybrid_loss, clf_loss, att_loss], y_train_prob, grads

    def train_step(self, data):
        x_fea, y = data
        y_inst, y_att = y

        # Apply gradients
        loss_values, y_prob, grads = self.gradient(x_fea, y_inst, y_att)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        y_pred = y_prob > 0.5 if self.num_class == 2 else tf.argmax(y_prob, axis=1)
        y_inst_cat = y_inst if self.num_class == 2 else tf.argmax(y_inst, axis=1)

        loss_tracker.update_state(loss_values[0])
        att_loss_tracker.update_state(loss_values[2])
        acc_tracker.update_state(y_true=y_inst_cat, y_pred=y_pred)

        return {"loss": loss_tracker.result(), "att_loss": att_loss_tracker.result(), "acc": acc_tracker.result()}

    def test_step(self, data):
        x_fea, y = data
        y_inst, y_att = y

        y_prob, y_att_prob = self(x_fea, training=False)
        y_pred = y_prob > 0.5 if self.num_class == 2 else tf.argmax(y_prob, axis=1)
        y_inst_cat = y_inst if self.num_class == 2 else tf.argmax(y_inst, axis=1)

        # Loss
        clf_loss = self.clf_loss_func(y_true=y_inst, y_pred=y_prob)
        att_loss = self.att_loss_func(y_true=y_att, y_pred=y_att_prob)

        # MSE to SE
        clf_loss = tf.cast(tf.shape(y_prob)[0] * tf.shape(y_prob)[1], tf.float32) * clf_loss
        att_loss = tf.cast(tf.shape(y_att_prob)[0] * tf.shape(y_att_prob)[1], tf.float32) * att_loss

        # Regularization
        reg_loss = tf.add_n(self.losses)

        # Hybrid loss
        hybrid_loss = clf_loss + self.lambda_att * att_loss + reg_loss

        # Update metrics
        val_loss_tracker.update_state(hybrid_loss)
        val_att_loss_tracker.update_state(att_loss)
        val_acc_tracker.update_state(y_true=y_inst_cat, y_pred=y_pred)

        return {"loss": val_loss_tracker.result(), "att_loss": val_att_loss_tracker.result(), "acc": val_acc_tracker.result()}

    def predict_prob(self, X_data):
        y_inst_prob, y_att_prob = self(X_data, training=False)

        y_inst_prob, y_att_prob = y_inst_prob.numpy(), y_att_prob.numpy()
        return y_inst_prob, y_att_prob

    def predict_label(self, X_data):
        y_inst_prob, y_att_prob = self.predict_prob(X_data)
        y_inst_pred = y_inst_prob > 0.5 if self.num_class == 2 else np.argmax(y_inst_prob, axis=1)
        y_att_pred = y_att_prob > 0

        return y_inst_pred, y_att_pred

    @property
    def metrics(self):
        return [loss_tracker, att_loss_tracker, acc_tracker, val_loss_tracker, val_att_loss_tracker, val_acc_tracker]


class BiLSTMAtt_Model:
    def __init__(self, model_name, num_class, num_fl, input_dim, input_act, att_dim, lstm_dim, lstm_att_act,
                 learn_rate, num_epoch, batch_size, regular_w, lambda_att):
        self.model_name = model_name
        self.num_class = num_class
        self.num_fl = num_fl

        self.input_dim = input_dim
        self.input_act = input_act
        self.att_dim = att_dim
        self.lstm_dim = lstm_dim
        self.lstm_att_act = lstm_att_act

        self.learn_rate = learn_rate
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.regular_w = regular_w
        self.lambda_att = lambda_att

        self.model = None
        self.intmd_res = None

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # Convert one-hot y to category for eval
        y_inst_val_cat = y_val[0] if self.num_class == 2 else np.argmax(y_val[0], axis=1)
        y_inst_test_cat = y_test[0] if self.num_class == 2 else np.argmax(y_test[0], axis=1)

        # Create model
        model = Att_Classifier(num_class=self.num_class, num_fl=self.num_fl,
                               input_dim=self.input_dim, input_act=self.input_act,
                               att_dim=self.att_dim, lstm_dim=self.lstm_dim, lstm_att_act=self.lstm_att_act,
                               regular_w=self.regular_w, lambda_att=self.lambda_att)

        # Set up optimizer
        optimizer = Adam(self.learn_rate)
        model.compile(optimizer=optimizer)

        # Fit
        test_callback = AttCallback((X_test, (y_inst_test_cat, y_test[1])), self.num_class)
        val_callback = AttCallback((X_val, (y_inst_val_cat, y_val[1])), self.num_class)
        model.fit(X_train, (y_train[0], y_train[1]), validation_data=(X_val, (y_val[0], y_val[1])),
                  epochs=self.num_epoch, batch_size=self.batch_size, callbacks=[test_callback, val_callback])

        self.model = model
        self.intmd_res = model.history.history

        self.intmd_res['test_acc'] = test_callback.acc_list
        self.intmd_res['test_f1'] = test_callback.f1_list
        self.intmd_res['test_att'] = test_callback.att_dict_list
        self.intmd_res['val_f1'] = val_callback.f1_list
        self.intmd_res['val_att'] = val_callback.att_dict_list

    def predict_prob(self, X_data):
        y_inst_prob, y_att_prob = self.model.predict_prob(X_data)
        return y_inst_prob, y_att_prob

    def predict_label(self, X_data):
        y_inst_pred, y_att_pred = self.model.predict_label(X_data)
        return y_inst_pred, y_att_pred

    def evaluate(self, X_test, y_inst_test, y_att_test):
        y_pred, y_att_pred = self.predict_label(X_test)
        y_inst_test_cat = y_inst_test if self.num_class == 2 else np.argmax(y_inst_test, axis=1)

        # Evaluation metrics
        acc = accuracy_score(y_inst_test_cat, y_pred)
        f1 = f1_score(y_inst_test_cat, y_pred, average='binary' if self.num_class == 2 else 'macro')
        att_metrics_dict = custom_att_metrics_dict(y_att_test, y_att_pred)
        metrics_dict = dict.copy(att_metrics_dict)
        metrics_dict['acc'] = acc
        metrics_dict['f1'] = f1

        return metrics_dict
