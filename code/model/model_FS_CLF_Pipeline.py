import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score

from utils.utils_metrics import custom_att_metrics_dict


class FS_CLF_model:
    def __init__(self, model_name, num_class, num_fl, fs_C, clf_C):
        self.model_name = model_name
        self.num_class = num_class
        self.num_fl = num_fl

        self.fs_C = fs_C
        self.clf_C = clf_C

        self.fs_model = None
        self.clf_model = None
        self.intmd_res = None

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        y_inst_train, y_att_train = y_train
        y_inst_val, y_att_val = y_val
        self.intmd_res = {}

        # Feature selector
        fs_model = OneVsRestClassifier(LogisticRegression(C=self.fs_C, penalty='l2', max_iter=10000))
        fs_model.fit(X_train, y_att_train)

        # Classifier
        att_mask = fs_model.predict(X_train)
        X_train_new = X_train * att_mask

        if self.num_class == 2:
            clf_model = LogisticRegression(C=self.clf_C, penalty='l2', max_iter=10000)
        else:
            clf_model = OneVsRestClassifier(LogisticRegression(C=self.clf_C, penalty='l2', max_iter=10000))
        clf_model.fit(X_train_new, y_inst_train)

        self.fs_model = fs_model
        self.clf_model = clf_model

        # Val metrics
        y_att_pred = self.fs_model.predict(X_val)
        y_pred = self.clf_model.predict(X_val*y_att_pred)
        att_metrics_dict = custom_att_metrics_dict(y_att_val, y_att_pred)
        self.intmd_res['val_acc'] = accuracy_score(y_inst_val, y_pred)
        self.intmd_res['val_f1'] = f1_score(y_inst_val, y_pred, average='binary' if self.num_class == 2 else 'macro')
        self.intmd_res['val_att'] = att_metrics_dict

    def predict(self, X):
        y_att_pred = self.fs_model.predict(X)
        y_pred = self.clf_model.predict(X * y_att_pred)

        return y_pred, y_att_pred

    def evaluate(self, X_test, y_inst_test, y_att_test):
        y_pred, y_att_pred = self.predict(X_test)

        # Evaluation metrics
        acc = accuracy_score(y_inst_test, y_pred)
        f1 = f1_score(y_inst_test, y_pred, average='binary' if self.num_class == 2 else 'macro')
        att_metrics_dict = custom_att_metrics_dict(y_att_test, y_att_pred)
        metrics_dict = dict.copy(att_metrics_dict)
        metrics_dict['acc'] = acc
        metrics_dict['f1'] = f1

        return metrics_dict
