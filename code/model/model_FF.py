import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from utils.utils_neural_network import TestCallback


class FF_Model:
    """
    Feed-forward NN using all features or partial features (globally) to classify.
    """
    def __init__(self, model_name, num_class, num_fl, hidden_dim, hidden_act, learn_rate, num_epoch, batch_size):
        self.model_name = model_name
        self.num_class = num_class
        self.num_fl = num_fl

        self.hidden_dim = hidden_dim
        self.hidden_act = hidden_act

        self.learn_rate = learn_rate
        self.num_epoch = num_epoch
        self.batch_size = batch_size

        self.model = None
        self.filter_out_fea_dix = None
        self.intmd_res = None

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        y_inst_train, y_fea_train = y_train
        y_inst_val, y_fea_val = y_val
        y_inst_test, y_fea_test = y_test

        # Use all features or filtered features
        if self.filter_out_fea_dix is None:
            X_train_new, X_test_new, X_val_new = X_train, X_test, X_val
        else:
            X_train_new, X_test_new, X_val_new = \
                X_train[:, self.filter_out_fea_dix], X_test[:, self.filter_out_fea_dix], X_val[:, self.filter_out_fea_dix]

        # Fit
        model = None
        if self.model_name == 'FF':
            num_out_unit = 1 if self.num_class == 2 else self.num_class
            out_act = 'sigmoid' if self.num_class == 2 else 'softmax'
            loss_func = BinaryCrossentropy() if self.num_class == 2 else SparseCategoricalCrossentropy()

            model = Sequential([
                Dense(self.hidden_dim, activation=self.hidden_act),
                Dense(num_out_unit, activation=out_act)
            ])

            test_callback = TestCallback((X_test_new, y_inst_test), self.num_class)
            val_callback = TestCallback((X_test_new, y_inst_test), self.num_class)

            model.compile(optimizer=Adam(self.learn_rate), loss=loss_func, metrics=['acc'])
            model.fit(X_train_new, y_inst_train,
                      validation_data=(X_val_new, y_inst_val),
                      epochs=self.num_epoch, batch_size=self.batch_size,
                      callbacks=[test_callback, val_callback])

        self.model = model
        self.intmd_res = model.history.history
        self.intmd_res['test_acc'] = test_callback.acc_list
        self.intmd_res['test_f1'] = test_callback.f1_list
        self.intmd_res['val_acc'] = val_callback.acc_list
        self.intmd_res['val_f1'] = val_callback.f1_list

    def evaluate(self, X_test, y_inst_test, y_att_test):
        if self.filter_out_fea_dix is not None:
            X_test_new = X_test[:, self.filter_out_fea_dix]
        else:
            X_test_new = X_test

        y_prob = self.model.predict(X_test_new)
        y_pred = y_prob > 0.5 if self.num_class == 2 else np.argmax(y_prob, axis=1)

        # Evaluation metrics
        acc = accuracy_score(y_inst_test, y_pred)
        f1 = f1_score(y_inst_test, y_pred, average='binary' if self.num_class == 2 else 'macro')

        return {'acc': acc, 'f1': f1}
