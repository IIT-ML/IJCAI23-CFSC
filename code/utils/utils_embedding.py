import numpy as np
import re
from collections import Counter, defaultdict


class EmbeddingService:
    def __init__(self, dataset, model_name, learn_strategy):
        self.dataset = dataset
        self.model_name = model_name
        self.learn_strategy = learn_strategy

        self.fea_name_list = None
        self.multiL2multiC_dict = defaultdict(lambda: 0)
        self.multiC2multiL_dict = {}

        # Select extraction techniques
        self.extract_func = self.binary_emb_extraction

    def build_y_label(self, y_inst):
        self.num_class = len(np.unique(y_inst))

        if self.model_name in ['LR_l1', 'LR_l2', 'LR_FS-CLF']:
            y_inst = np.asarray(y_inst, dtype=int)
        elif self.model_name in ['BiLSTM-ATT', 'RNP'] and self.num_class > 2:  # models using MSE
            # One-hot encoding for multiclass
            mc_y_inst = np.zeros(shape=(len(y_inst), self.num_class))
            for x, y in enumerate(y_inst):
                mc_y_inst[(x, y)] = 1
            y_inst = mc_y_inst
        else:
            y_inst = np.reshape(np.asarray(y_inst, dtype=np.float32), newshape=(len(y_inst), 1))

        return y_inst

    def build_att_label(self, X_text, mode, th=0.90, num_top_rules=10):
        y_att = [self.feature_name2label(text) for text in X_text]
        self.num_fl = len(self.fea_name_list)

        if "MultiC" in self.learn_strategy and mode == "train":
            y_att_tuple = [tuple(y) for y in y_att]
            y_att_count = sorted(Counter(y_att_tuple).items(), key=lambda x: -x[1])

            count_array = [y[1] for y in y_att_count]
            count_prob = np.cumsum(count_array)/len(X_text)
            if len(count_array) <= num_top_rules:
                use_idx = list(np.arange(len(count_array)))
            else:
                use_idx = list(np.nonzero(count_prob < th)[0])
                use_idx.append((use_idx[-1])+1)

            # Print
            a = np.asarray(y_att_count)[use_idx][:, 0]
            a_name = ["^".join(np.asarray(self.fea_name_list)[np.nonzero(a_i)[0]]) for a_i in a]
            a_count = np.asarray(count_array)[use_idx]
            a_prob = count_prob[use_idx]

            # print("\n".join(a_name))
            # print("\n".join([str(a_i) for a_i in a_count]))
            # print("\n".join([str(a_i) for a_i in a_prob]))

            # Construct label dict
            for idx in use_idx:
                self.multiL2multiC_dict[y_att_count[idx][0]] = int(idx+1)
            self.multiC2multiL_dict = {v: k for k, v in self.multiL2multiC_dict.items()}
            self.multiC2multiL_dict[0] = tuple(np.zeros(len(y_att_tuple[0]), dtype=int))  # Add default feature label

            # Convert to multiclass
            y_att_new = [self.multiL2multiC_dict[tuple(y)] for y in y_att]

            return y_att_new

        else:
            return y_att

    def feature_name2label(self, x):
        text = x.lower()
        text_label = []
        for name in self.fea_name_list:
            # Search for feature as exact word
            if re.search(r'\b' + name.lower() + r'\b', text):
                text_label.append(1)
            else:
                text_label.append(0)
        return text_label

# ------------------------------------------------------------------------
    def binary_emb_extraction(self, X, y, mode):
        X_fea, X_text = X.iloc[:, :-1].values, X.iloc[:, -1].values
        self.fea_name_list = list(X.iloc[:, :-1].columns.values)

        # Construct labels
        y_att = self.build_att_label(X_text, mode)
        y_inst = y

        return X_fea, y_inst, y_att
