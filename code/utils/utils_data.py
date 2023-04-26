import collections

import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataMgmt:
    def __init__(self, dataset, model_name, file_path, n_bins):
        self.dataset = dataset
        self.model_name = model_name
        self.file_path = file_path
        self.scaler = StandardScaler()
        self.discretizer = {}
        self.n_bins = n_bins

        self.continuous_idx = None

        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val = None, None, None, None, None, None

    def continuous_discrete(self, X, n_bins, mode):
        fea_mat = X.iloc[:, :-1].values

        # Only for continuous features
        fea_bin_mat = np.copy(fea_mat)
        for col_idx in self.continuous_idx:
            cur_col = fea_mat[:, col_idx]

            # Get bins on train set
            if mode == 'train':
                bins = list(np.linspace(np.min(cur_col), np.max(cur_col), n_bins - 1))
                bins.insert(0, -np.inf)
                bins.append(np.inf)
                self.discretizer[col_idx] = bins

            # Apply to train and test set
            inds = np.digitize(cur_col, self.discretizer[col_idx])
            fea_bin_mat[:, col_idx] = inds

        # Apply to original features
        X.iloc[:, :-1] = fea_bin_mat

        return X

    # def category_binarize(self, df):
    #     path = self.file_path.split(self.dataset)[0] + "/%s_fea_val.json" % self.dataset
    #     with open(path) as f:
    #         feature_value_mapping = json.load(f)
    #     for fea, fea_dict in feature_value_mapping.items():
    #         feature_value_mapping[fea] = {int(k): v for k, v in feature_value_mapping[fea].items()}
    #
    #     if self.dataset == 'credit':
    #         new_df, unchanged_df = df.iloc[:, :-3], df.iloc[:, -3:]
    #     elif self.dataset == 'mimic':
    #         new_df, unchanged_df = df.iloc[:, 3:-2], df.iloc[:, [0, 1, 2, -1, -2]]
    #
    #     new_df = new_df.replace(feature_value_mapping)
    #     new_df = pd.get_dummies(new_df, prefix_sep="=", columns=list(feature_value_mapping.keys()))
    #     new_df = pd.concat([new_df, unchanged_df], axis=1)
    #     return new_df

    def data_preprocessing(self, X, method=None, mode='train'):
        if self.model_name in ['DT']:
            return X

        # Discretize continuous features
        if self.model_name in ['Rule_pos', 'Rule_neg', 'Rule_mix']:
            X = self.continuous_discrete(X, self.n_bins, mode)
            return X

        # Only apply z-score to continuous variables
        if method == 'z-score':
            fea_mat = np.copy(X.iloc[:, self.continuous_idx].values)

            if mode == 'train':
                fea_mat_scale = self.scaler.fit_transform(fea_mat)
            else:
                fea_mat_scale = self.scaler.transform(fea_mat)

            X.iloc[:, self.continuous_idx] = fea_mat_scale

        return X

    def load_data(self):
        X_raw = pd.read_csv(self.file_path).fillna('')

        if self.dataset == 'credit':
            self.continuous_idx = list(range(14))

        elif self.dataset == 'mobile':
            self.continuous_idx = [0, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        elif self.dataset in ['nhisCancer', 'nhisCOPD']:
            self.continuous_idx = list(range(6))

        elif self.dataset == 'ride':
            self.continuous_idx = list(range(6))
            self.continuous_idx.extend(list(range(11, 27)))

        else:
            # All features
            self.continuous_idx = list(range(len(X_raw.columns[:-3])))

        return X_raw

    def select_train_test(self, X_raw, data_prepro='z-score'):
        # Train test val split
        if self.dataset in ['credit', 'company', 'mobile', 'nhisCancer', 'nhisCOPD', 'ride']:
            train_df, test_df = X_raw[X_raw['Type'] == 'train'], X_raw[X_raw['Type'] == 'test']
            X_train, y_train = train_df.drop(columns=['Type', 'Label']), train_df['Label'].values
            X_test, y_test = test_df.drop(columns=['Type', 'Label']), test_df['Label'].values

        else:
            X_raw, y_raw = X_raw.drop(columns=['Type', 'Label']), X_raw['Label'].values
            X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=2./3, random_state=1234, stratify=y_raw)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=1./2, random_state=1234, stratify=y_test)

        # Preprocessing
        self.X_train = self.data_preprocessing(X_train, data_prepro, mode='train')
        self.X_test = self.data_preprocessing(X_test, data_prepro, mode='test')
        self.X_val = self.data_preprocessing(X_val, data_prepro, mode='test')

        self.y_train, self.y_test, self.y_val = np.asarray(y_train), np.asarray(y_test), np.asarray(y_val)

        return
