import collections

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import wittgenstein as lw
from sklearn.metrics import accuracy_score, f1_score

from utils.utils_metrics import custom_att_metrics_dict, custom_sparsity


class CLF_Model:
    def __init__(self, model_name, num_class, num_fl,
                 lr_C,
                 max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_imp_dec,
                 max_rules, max_rule_conds, max_total_conds, k):
        self.model_name = model_name
        self.num_class = num_class
        self.num_fl = num_fl

        self.lr_C = lr_C

        self.max_depth = None if max_depth == -1 else max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = None if max_leaf_nodes == -1 else max_leaf_nodes
        self.min_imp_dec = min_imp_dec

        self.max_rules = None if max_rules == -1 else max_rules
        self.max_rule_conds = None if max_rule_conds == -1 else max_rule_conds
        self.max_total_conds = None if max_total_conds == -1 else max_total_conds
        self.k = k

        self.model = None
        self.intmd_res = None

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        y_inst_train, y_att_train = y_train
        y_inst_val, y_att_val = y_val
        self.intmd_res = {}

        model = None
        if 'LR' in self.model_name:
            if self.model_name == 'LR_l1':
                model = LogisticRegression(C=self.lr_C, penalty='l1', solver='saga', max_iter=10000)

            elif self.model_name == 'LR_l2':
                model = LogisticRegression(C=self.lr_C, penalty='l2', max_iter=10000)

            if self.num_class == 2:
                # Binary
                # Fit model
                model.fit(X_train, y_inst_train)

                # Use normalized weights as attention
                fea_weights = model.coef_.flatten()
                abs_weights = np.absolute(fea_weights)
                normalized_weights = abs_weights / np.sum(abs_weights)
                num_zero_weights = np.nonzero(normalized_weights == 0)[0].shape[0]
                self.fea_weights = normalized_weights

            else:
                # Multiclass
                # Fit model
                mc_model = OneVsRestClassifier(model)
                mc_model.fit(X_train, y_inst_train)

                # Use normalized weights as attention
                self.fea_weights = []
                for c_model in mc_model.estimators_:
                    fea_weights = c_model.coef_.flatten()
                    abs_weights = np.absolute(fea_weights)
                    normalized_weights = abs_weights / np.sum(abs_weights)
                    num_zero_weights = np.nonzero(normalized_weights == 0)[0].shape[0]
                    self.fea_weights.append(normalized_weights)

                self.fea_weights = np.asarray(self.fea_weights)
                model = mc_model

        elif self.model_name == 'DT':
            model = DecisionTreeClassifier(max_depth=self.max_depth,
                                           min_samples_split=self.min_samples_split,
                                           min_samples_leaf=self.min_samples_leaf,
                                           max_leaf_nodes=self.max_leaf_nodes,
                                           min_impurity_decrease=self.min_imp_dec)

            # Fit model
            model.fit(X_train, y_inst_train)

            # Get rules
            self.rules = self.get_rules(model)

            # tree.plot_tree(model, filled=True, node_ids=True, fontsize=8)
            # plt.savefig("DT.png")
            # print()

        elif self.model_name == 'Rule_mix':
            pos_model = lw.RIPPER(max_rules=self.max_rules,
                                  max_rule_conds=self.max_rule_conds,
                                  max_total_conds=self.max_total_conds,
                                  k=self.k, n_discretize_bins=None)
            neg_model = lw.RIPPER(max_rules=self.max_rules,
                                  max_rule_conds=self.max_rule_conds,
                                  max_total_conds=self.max_total_conds,
                                  k=self.k, n_discretize_bins=None)
            pos_model.fit(X_train, y_inst_train, pos_class=1)
            neg_model.fit(X_train, y_inst_train, pos_class=0)
            model = [neg_model, pos_model]

        elif self.model_name in ['Rule_pos', 'Rule_neg']:
            pos_class = 1 if self.model_name == 'Rule_pos' else 0
            model = lw.RIPPER(max_rules=self.max_rules,
                              max_rule_conds=self.max_rule_conds,
                              max_total_conds=self.max_total_conds,
                              k=self.k, n_discretize_bins=None)
            model.fit(X_train, y_inst_train, pos_class=pos_class)

        self.model = model

        # Val metrics
        y_pred, y_att_prob = self.predict(X_val)
        att_metrics_dict = custom_att_metrics_dict(y_att_val, y_att_prob > 0)
        self.intmd_res['val_acc'] = accuracy_score(y_inst_val, y_pred)
        self.intmd_res['val_f1'] = f1_score(y_inst_val, y_pred, average='binary' if self.num_class == 2 else 'macro')
        self.intmd_res['val_att'] = att_metrics_dict

    def get_rules(self, tree):
        """
        https://mljar.com/blog/extract-rules-decision-tree/
        :param tree:
        :return:
        """
        tree_ = tree.tree_
        paths = []
        path = []

        def recurse(node, path, paths):
            if tree_.feature[node] >= 0:
                name = 'feature' + str(tree_.feature[node])
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [[name, "<=", threshold]]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [[name, ">", threshold]]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths += [path]

        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        # Append rule path
        rules = []
        for path in paths:
            rule = []

            for p in path[:-1]:
                rule.append(p)

            # Add class
            classes = path[-1][0][0]
            rule.append(int(np.argmax(classes)))

            rules.append(rule)

        return rules

    def apply_rule(self, X):
        y_pred, y_att_pred = [], []
        for x in X:
            for rule in self.rules:
                for cond in rule[:-1]:
                    fea_idx = int(cond[0].split('feature')[1])
                    exp_str = "%s %s %s" % (x[fea_idx], cond[1], cond[2])
                    if not eval(exp_str):
                        break
                else:
                    features = [int(cond[0].split('feature')[1]) for cond in rule[:-1]]
                    att = np.zeros(X.shape[1])
                    att[features] = 1
                    y_att_pred.append(att)
                    y_pred.append(rule[-1])

        return np.asarray(y_pred), np.asarray(y_att_pred)

    def predict(self, X):
        if self.model_name == 'Rule_mix':
            neg_pred = self.model[0].predict(X, give_reasons=True)
            pos_pred = self.model[1].predict(X, give_reasons=True)
            final_pred, y_att_prob = [], []
            for i, pred in enumerate(pos_pred[0]):
                att = np.zeros(X.shape[1])

                if pred:
                    # Use rule from pos model
                    rules = pos_pred[1][i]
                else:
                    # Use rule from neg model
                    rules = neg_pred[1][i]

                if len(rules) > 0:
                    features = [c.feature for c in rules[0].conds]
                    att[features] = 1

                final_pred.append(pred)
                y_att_prob.append(att)
            return np.asarray(final_pred), np.asarray(y_att_prob)

        elif self.model_name in ['Rule_pos', 'Rule_neg']:
            y_pred_tuple = self.model.predict(X, give_reasons=True)
            y_pred = np.logical_not(y_pred_tuple[0]) if self.model_name == 'Rule_neg' else y_pred_tuple[0]

            y_att_prob = []
            for rules in y_pred_tuple[1]:
                att = np.zeros(X.shape[1])

                if len(rules) > 0:
                    features = [c.feature for c in rules[0].conds]
                    att[features] = 1

                y_att_prob.append(att)
            return np.asarray(y_pred), np.asarray(y_att_prob)

        elif 'LR' in self.model_name:
            if self.num_class == 2:
                # Binary
                y_pred = self.model.predict(X)
                y_att_prob = np.asarray(np.tile(self.fea_weights, (len(X), 1)))
                # y_att_prob = y_att_prob * X  # Use the product of feature value and coefficient as attention
                y_att_pred = y_att_prob > 0

            else:
                # Multiclass
                y_pred = self.model.predict(X)
                # y_att_prob = np.asarray(np.tile(self.fea_weights, (len(X), 1, 1)))
                y_att_prob = self.fea_weights[y_pred, :]
                y_att_pred = y_att_prob > 0

            return y_pred, y_att_pred

        elif self.model_name == 'DT':
            _, y_att_pred = self.apply_rule(X)
            return self.model.predict(X), np.asarray(y_att_pred)

    def evaluate(self, X_test, y_inst_test, y_att_test):
        y_pred, y_att_pred = self.predict(X_test)

        # Evaluation metrics
        acc = accuracy_score(y_inst_test, y_pred)
        f1 = f1_score(y_inst_test, y_pred, average='binary' if self.num_class == 2 else 'macro')
        att_metrics_dict = custom_att_metrics_dict(y_att_test, y_att_pred)
        metrics_dict = dict.copy(att_metrics_dict)
        metrics_dict['acc'] = acc
        metrics_dict['f1'] = f1

        a = custom_att_metrics_dict(y_att_test, np.zeros(y_att_test.shape))
        b = custom_att_metrics_dict(y_att_test, np.ones(y_att_test.shape))
        c = custom_sparsity(y_att_test)

        return metrics_dict
