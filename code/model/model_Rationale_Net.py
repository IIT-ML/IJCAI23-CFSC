import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Mean, Accuracy
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam

from utils.utils_metrics import custom_att_metrics_dict
from utils.utils_neural_network import AttCallback, gumbel_softmax

loss_tracker = Mean(name="loss")
clf_loss_tracker = Mean(name="clf_loss")
fea_loss_tracker = Mean(name="fea_loss")
acc_tracker = Accuracy(name="acc")

val_loss_tracker = Mean(name="loss")
val_clf_loss_tracker = Mean(name="clf_loss")
val_fea_loss_tracker = Mean(name="fea_loss")
val_acc_tracker = Accuracy(name="acc")


class RN_Classifier(Model):
    """
    Adapted from https://github.com/yala/text_nn
    Rationale Net that automatically select features to do classification. Used gumbel-softmax. No feature supervision provided.
    """
    def __init__(self, num_class, num_fl, input_dim, input_act, att_dim, att_act, hidden_dim, hidden_act, lambda_omega):
        super(RN_Classifier, self).__init__()

        self.num_class = num_class
        self.num_fl = num_fl
        self.lambda_omega = lambda_omega

        num_out_unit = 1 if num_class == 2 else num_class
        self.clf_output_act = 'sigmoid' if self.num_class == 2 else 'softmax'
        self.clf_loss_func = MeanSquaredError()
        self.fea_att_act = gumbel_softmax

        # Layers
        # Feature selection module
        self.fs_input_layer = Dense(input_dim, activation=input_act, name='fs_input')
        self.fs_hidden_layer1 = Dense(att_dim, activation=att_act, name='fs_h1')
        self.fs_out_layer = Dense(num_fl, activation='linear', name='fs_out')

        # Classification module
        self.clf_hidden_layer = Dense(hidden_dim, activation=hidden_act, name='clf_h1')
        self.clf_out_layer = Dense(num_out_unit, activation=self.clf_output_act, name='clf_out')

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, **kwargs):
        fea_vec_input = inputs  # (batch, num_fl, 1)

        # Generator
        fea_input = tf.squeeze(fea_vec_input, axis=-1)  # (batch, num_fl)
        fea_hidden = self.fs_input_layer(fea_input)
        fea_hidden = self.fs_hidden_layer1(fea_hidden)
        fea_out = self.fs_out_layer(fea_hidden)
        fea_sparse_out = self.fea_att_act(fea_out, training=training)

        # Encoder
        reshape_att = tf.expand_dims(fea_sparse_out, axis=-1)  # reshape_att: (batch_size, num_fl, hidden_dim)
        reshape_att = tf.repeat(reshape_att, tf.shape(fea_vec_input)[-1])
        reshape_att = tf.reshape(reshape_att, (-1, tf.shape(fea_vec_input)[1], tf.shape(fea_vec_input)[-1]))

        # Concatenate
        weighted_emb = fea_vec_input * reshape_att
        concat_emb = Reshape((-1, self.num_fl*1))(weighted_emb)
        concat_emb = tf.keras.backend.squeeze(concat_emb, axis=1)

        # Hidden
        clf_hidden = self.clf_hidden_layer(concat_emb)

        # Output
        clf_output = self.clf_out_layer(clf_hidden)

        return clf_output, fea_sparse_out

    def gradient(self, x, y_inst, y_att):
        with tf.GradientTape() as tape:
            y_prob, y_att_sparse = self(x, training=True)

            # Classification loss
            clf_loss = self.clf_loss_func(y_true=y_inst, y_pred=y_prob)

            # Selection loss
            select_loss = tf.reduce_mean(tf.norm(y_att_sparse, ord=1, axis=-1))

            # Hybrid loss
            hybrid_loss = clf_loss + self.lambda_omega*select_loss

        grads = tape.gradient(hybrid_loss, self.trainable_variables)
        return [hybrid_loss, clf_loss, select_loss], y_prob, grads

    def train_step(self, data):
        x, y = data
        y_inst, y_att = y

        # Apply gradients
        loss_values, y_prob, grads = self.gradient(x, y_inst, y_att)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        y_pred = y_prob > 0.5 if self.num_class == 2 else tf.argmax(y_prob, axis=1)
        y_inst_cat = y_inst if self.num_class == 2 else tf.argmax(y_inst, axis=1)

        loss_tracker.update_state(loss_values[0])
        clf_loss_tracker.update_state(loss_values[1])
        fea_loss_tracker.update_state(loss_values[2])
        acc_tracker.update_state(y_true=y_inst_cat, y_pred=y_pred)

        return {"loss": loss_tracker.result(), "clf_loss": clf_loss_tracker.result(), "fea_loss": fea_loss_tracker.result(),
                "acc": acc_tracker.result()}

    def test_step(self, data):
        x, y = data
        y_inst, y_att = y

        y_prob, y_att_sparse = self(x, training=False)
        y_pred = y_prob > 0.5 if self.num_class == 2 else tf.argmax(y_prob, axis=1)
        y_inst_cat = y_inst if self.num_class == 2 else tf.argmax(y_inst, axis=1)

        # Loss values
        clf_loss = self.clf_loss_func(y_true=y_inst, y_pred=y_prob)
        select_loss = tf.reduce_mean(tf.norm(y_att_sparse, ord=1, axis=-1))
        hybrid_loss = clf_loss + self.lambda_omega*select_loss

        # Update metrics
        val_loss_tracker.update_state(hybrid_loss)
        val_clf_loss_tracker.update_state(clf_loss)
        val_fea_loss_tracker.update_state(select_loss)
        val_acc_tracker.update_state(y_true=y_inst_cat, y_pred=y_pred)

        return {"loss": val_loss_tracker.result(), "clf_loss": val_clf_loss_tracker.result(), "fea_loss": val_fea_loss_tracker.result(),
                "acc": val_acc_tracker.result()}

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
        return [loss_tracker, clf_loss_tracker, fea_loss_tracker, acc_tracker,
                val_loss_tracker, val_clf_loss_tracker, val_fea_loss_tracker, val_acc_tracker]


class Rationale_Net_Model:
    def __init__(self, model_name, num_class, num_fl, input_dim, input_act, att_dim, att_act, hidden_dim, hidden_act,
                 learn_rate, num_epoch, batch_size, lambda_omega):
        self.model_name = model_name
        self.num_class = num_class
        self.num_fl = num_fl

        self.input_dim = input_dim
        self.input_act = input_act
        self.att_dim = att_dim
        self.att_act = att_act
        self.hidden_dim = hidden_dim
        self.hidden_act = hidden_act

        self.learn_rate = learn_rate
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.lambda_omega = lambda_omega

        self.model = None
        self.intmd_res = None

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # Convert one-hot y to category for eval
        y_inst_val_cat = y_val[0] if self.num_class == 2 else np.argmax(y_val[0], axis=1)
        y_inst_test_cat = y_test[0] if self.num_class == 2 else np.argmax(y_test[0], axis=1)

        # Train classifier
        model = RN_Classifier(num_class=self.num_class, num_fl=self.num_fl, input_dim=self.input_dim, input_act=self.input_act,
                              att_dim=self.att_dim, att_act=self.att_act, hidden_dim=self.hidden_dim, hidden_act=self.hidden_act,
                              lambda_omega=self.lambda_omega)
        optimizer = Adam(self.learn_rate)
        model.compile(optimizer=optimizer, run_eagerly=True)

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
