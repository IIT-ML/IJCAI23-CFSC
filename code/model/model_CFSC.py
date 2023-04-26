import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Mean, Accuracy
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam

from utils.utils_metrics import custom_att_metrics_dict
from utils.utils_neural_network import AttCallback

loss_tracker = Mean(name="loss")
clf_loss_tracker = Mean(name="clf_loss")
fea_loss_tracker = Mean(name="fea_loss")
acc_tracker = Accuracy(name="acc")

val_loss_tracker = Mean(name="loss")
val_clf_loss_tracker = Mean(name="clf_loss")
val_fea_loss_tracker = Mean(name="fea_loss")
val_acc_tracker = Accuracy(name="acc")


class Classifier(Model):
    """
    Our classification model based on local feature selection. The feature selection module will be trained first.
    """
    def __init__(self, num_class, num_fl, hidden_dim, hidden_act, fea_selector, ft_opt, lambda_a):
        super(Classifier, self).__init__()

        self.num_class = num_class
        self.num_fl = num_fl
        self.lambda_a = lambda_a

        num_out_unit = 1 if num_class == 2 else num_class
        self.clf_output_act = 'sigmoid' if self.num_class == 2 else 'softmax'
        self.clf_loss_func = BinaryCrossentropy() if self.num_class == 2 else SparseCategoricalCrossentropy()
        self.att_loss_func = BinaryCrossentropy(from_logits=True)

        # Layers
        self.fea_selector = fea_selector
        self.fea_selector.trainable = ft_opt  # Freeze selector or not

        self.hidden_layer1 = Dense(hidden_dim, activation=hidden_act, name='hidden')
        # self.hidden_layer2 = Dense(8, activation=self.hidden_act, name='hidden')

        self.out_layer = Dense(num_out_unit, activation=self.clf_output_act, name='clf_out')

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, **kwargs):
        fea_vec_input = inputs  # (batch, num_fl, 1)
        fea_input = tf.squeeze(fea_vec_input, axis=-1)  # (batch, num_fl)

        # Get feature attention
        z_att = self.fea_selector(fea_input)
        z_sparse_att = relu(z_att)

        # reshape_att: (batch_size, num_fl, hidden_dim)
        reshape_att = tf.expand_dims(z_sparse_att, axis=-1)
        reshape_att = tf.repeat(reshape_att, tf.shape(fea_vec_input)[-1])
        reshape_att = tf.reshape(reshape_att, (-1, tf.shape(fea_vec_input)[1], tf.shape(fea_vec_input)[-1]))

        # Concatenate
        weighted_emb = fea_vec_input * reshape_att
        concat_emb = Reshape((-1, self.num_fl*1))(weighted_emb)
        concat_emb = tf.keras.backend.squeeze(concat_emb, axis=1)

        # Hidden
        hidden = self.hidden_layer1(concat_emb)

        # Output
        output = self.out_layer(hidden)

        return output, z_att, z_sparse_att

    def gradient(self, x, y_inst, y_att):
        with tf.GradientTape() as tape:
            y_prob, y_att_value, y_att_sparse = self(x, training=True)

            # Classification loss
            clf_loss = self.clf_loss_func(y_true=y_inst, y_pred=y_prob)

            # Att loss
            att_loss = self.att_loss_func(y_true=y_att, y_pred=y_att_value)

            # Hybrid loss
            if self.lambda_a is not None:
                hybrid_loss = (1-self.lambda_a)*clf_loss + self.lambda_a*att_loss
            else:
                hybrid_loss = clf_loss + att_loss

        grads = tape.gradient(hybrid_loss, self.trainable_variables)
        return [hybrid_loss, clf_loss, att_loss], y_prob, grads

    def train_step(self, data):
        x, y = data
        y_inst, y_att = y

        # Apply gradients
        loss_values, y_prob, grads = self.gradient(x, y_inst, y_att)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        y_pred = y_prob > 0.5 if self.num_class == 2 else tf.argmax(y_prob, axis=1)

        loss_tracker.update_state(loss_values[0])
        clf_loss_tracker.update_state(loss_values[1])
        fea_loss_tracker.update_state(loss_values[2])
        acc_tracker.update_state(y_true=y_inst, y_pred=y_pred)

        return {"loss": loss_tracker.result(), "clf_loss": clf_loss_tracker.result(), "fea_loss": fea_loss_tracker.result(),
                "acc": acc_tracker.result()}

    def test_step(self, data):
        x, y = data
        y_inst, y_att = y

        y_prob, y_att_value, y_att_sparse = self(x, training=False)
        y_pred = y_prob > 0.5 if self.num_class == 2 else tf.argmax(y_prob, axis=1)

        # Loss values
        clf_loss = self.clf_loss_func(y_true=y_inst, y_pred=y_prob)
        att_loss = self.att_loss_func(y_true=y_att, y_pred=y_att_value)
        if self.lambda_a is not None:
            hybrid_loss = (1 - self.lambda_a) * clf_loss + self.lambda_a * att_loss
        else:
            hybrid_loss = clf_loss + att_loss

        # Update metrics
        val_loss_tracker.update_state(hybrid_loss)
        val_clf_loss_tracker.update_state(clf_loss)
        val_fea_loss_tracker.update_state(att_loss)
        val_acc_tracker.update_state(y_true=y_inst, y_pred=y_pred)

        return {"loss": val_loss_tracker.result(), "clf_loss": val_clf_loss_tracker.result(), "fea_loss": val_fea_loss_tracker.result(),
                "acc": val_acc_tracker.result()}

    def predict_prob(self, X_data):
        y_inst_prob, _, _ = self(X_data, training=False)
        y_att_prob = tf.math.sigmoid(self.fea_selector(X_data))

        y_inst_prob, y_att_prob = y_inst_prob.numpy(), y_att_prob.numpy()
        return y_inst_prob, y_att_prob

    def predict_label(self, X_data):
        y_inst_prob, y_att_prob = self.predict_prob(X_data)
        y_inst_pred = y_inst_prob > 0.5 if self.num_class == 2 else np.argmax(y_inst_prob, axis=1)
        y_att_pred = y_att_prob > 0.5

        return y_inst_pred, y_att_pred

    @property
    def metrics(self):
        return [loss_tracker, clf_loss_tracker, fea_loss_tracker, acc_tracker,
                val_loss_tracker, val_clf_loss_tracker, val_fea_loss_tracker, val_acc_tracker]

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


class CFSC_Model:
    def __init__(self, model_name, num_class, num_fl, input_dim, input_act, att_dim, att_act, hidden_dim, hidden_act,
                 learn_rate, num_epoch, batch_size, lambda_a):
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
        self.lambda_a = lambda_a

        self.ft_opt = True if self.model_name == 'CFSC_ft' else False

        self.model = None
        self.intmd_res = None

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # Pretrain Feature Selector
        fea_selector = Sequential([
            Dense(self.input_dim, activation=self.input_act, name='fea_select_h1'),
            Dense(self.att_dim, activation=self.att_act, name='fea_select_h2'),
            Dense(self.num_fl, activation='linear', name='fea_select_out')
        ])
        fea_selector.compile(optimizer=Adam(1e-3), loss=BinaryCrossentropy(from_logits=True), metrics=['acc'])
        X_fea_train, y_fea_train = np.squeeze(X_train), y_train[1]
        X_fea_val, y_fea_val = np.squeeze(X_val), y_val[1]
        fea_selector.fit(X_fea_train, y_fea_train, validation_data=(X_fea_val, y_fea_val), epochs=100, batch_size=32)

        # Train classifier
        model = Classifier(num_class=self.num_class, num_fl=self.num_fl, hidden_dim=self.hidden_dim, hidden_act=self.hidden_act,
                           fea_selector=fea_selector, ft_opt=self.ft_opt, lambda_a=self.lambda_a)
        optimizer = Adam(self.learn_rate)
        model.compile(optimizer=optimizer)

        test_callback = AttCallback((X_test, y_test), self.num_class)
        val_callback = AttCallback((X_val, y_val), self.num_class)

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

        # Evaluation metrics
        acc = accuracy_score(y_inst_test, y_pred)
        f1 = f1_score(y_inst_test, y_pred, average='binary' if self.num_class == 2 else 'macro')
        att_metrics_dict = custom_att_metrics_dict(y_att_test, y_att_pred)
        metrics_dict = dict.copy(att_metrics_dict)
        metrics_dict['acc'] = acc
        metrics_dict['f1'] = f1

        return metrics_dict
