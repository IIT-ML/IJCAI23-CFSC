import argparse
import collections
import os

import LwFL as LwFL
from utils.utils_data import DataMgmt
from utils.utils_embedding import EmbeddingService

parser = argparse.ArgumentParser(description='LwFL')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# ------------------------------------------------------------------------
# Task Settings
# ------------------------------------------------------------------------
parser.add_argument('--root', type=str, default='..', help='Root path of code folder')

parser.add_argument('--dataset_name', type=str, default='credit', help='Dataset name',
                    choices=['credit', 'company', 'mobile', 'nhisCOPD', 'ride'
                             'synthetic1', 'synthetic2', 'synthetic3'])

parser.add_argument('--seed', type=int, default='11039', help='Random seed')


# ------------------------------------------------------------------------
# Model Settings
# ------------------------------------------------------------------------
parser.add_argument('--model_name', type=str, default='CFSC_ft', help='Type of the learning model',
    choices=['CFSC_ft', 'BiLSTM-ATT', 'LR_l1', 'LR_l2', 'DT', 'Rule_pos', 'Rule_neg', 'RNP', 'LR_FS-CLF', 'FF'])

parser.add_argument('--type_learn', type=str, default='FS_MultiL_all', help='Type of the learning strategy',
    choices=['FS_MultiL_all', 'Clf_all'])

parser.add_argument('--type_fl', type=str, default='no_bias_flip',
                    help='Type of the feature labels generation strategy: '
                         '"None" is for synthetic datasets; '
                         '"no_bias_flip" is the evidence counterfactual simulation; '
                         '"dt" is the decision tree simulation',
    choices=['None', 'no_bias_flip', 'dt'])

# ------------------------------------------------------------------------
# Training Settings
# ------------------------------------------------------------------------
# general
parser.add_argument('--num_trial', type=int, default=0, help='Number of normal train-test split trials')
parser.add_argument('--num_fold', type=int, default=5, help='Number of folds for k-fold cross validation')

# neural network
parser.add_argument('--input_dim', type=int, default=64, help='Dim. of the input layer')
parser.add_argument('--input_act', type=str, default='tanh', help='Activation function of input layer',
                    choices=['relu', 'tanh', 'sigmoid'])
parser.add_argument('--att_dim', type=int, default=256, help='Dim. of attention layer')
parser.add_argument('--att_act', type=str, default='tanh', help='Activation function of attention layer',
                    choices=['relu', 'tanh', 'sigmoid'])

parser.add_argument('--hidden_dim', type=int, default=16, help='Dim. of the hidden layer before output')
parser.add_argument('--hidden_act', type=str, default='tanh', help='Activation function of hidden layer',
                    choices=['relu', 'tanh', 'sigmoid'])

parser.add_argument('--num_epoch', type=int, default=100, help='Number of epochs for deep learning model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for deep learning model')
parser.add_argument('--learn_rate', type=float, default=1e-2, help='Learning rate of optimizer')
parser.add_argument('--regular_w', type=float, default=0.0, help='Regularization value')


# ------------------------------------------------------------------------
# Model Parameters
# ------------------------------------------------------------------------
# CFSC
parser.add_argument('--lambda_a', type=float, default=0.0, help='Parameter to control the importance of feature selection loss')

# BiLSTM-ATT
parser.add_argument('--lstm_dim', type=int, default=32, help='Dim. of the lstm layer')
parser.add_argument('--lstm_att_act', type=str, default='sparsemax', help='Activation function of attention layer',
                    choices=['softmax', 'sparsemax'])
parser.add_argument('--lambda_att', type=float, default=0.0, help='Parameter to control attention regularization')

# RNP
parser.add_argument('--lambda_omega', type=float, default=1e-4, help='Parameter to control the regularization on the number of selected feature')

# LR FS-CLF Pipeline
parser.add_argument('--fs_C', type=float, default=1.0, help='C value for feature selector')
parser.add_argument('--clf_C', type=float, default=1.0, help='C value for classifier')

# Logistic Regression
parser.add_argument('--lr_C', type=float, default=1.0, help='C value for logistic regression')

# Decision Tree
parser.add_argument('--max_depth', type=int, default=-1, help='Max. depth for decision tree classifier')
parser.add_argument('--min_samples_split', type=int, default=2, help='Min. number of samples to split an internal node')
parser.add_argument('--min_samples_leaf', type=int, default=1, help='Min. number of samples for a leaf node')
parser.add_argument('--max_leaf_nodes', type=int, default=-1, help='Grow a tree with max_leaf_nodes in best-first fashion')
parser.add_argument('--min_imp_dec', type=float, default=0.0, help='A node will be split if this split induces a decrease of the impurity greater than or equal to this value.')

# Rule-based system
parser.add_argument('--max_rules', type=int, default=-1, help='Maximum number of rules')
parser.add_argument('--max_rule_conds', type=int, default=2, help='Maximum number of conds per rule')
parser.add_argument('--max_total_conds', type=int, default=-1, help='Maximum number of total conds in entire ruleset')
parser.add_argument('--k', type=int, default=2, help='Number of RIPPERk optimization iterations')
parser.add_argument('--n_discretize_bins', type=int, default=10, help='Fit apparent numeric attributes into n_discretize_bins discrete bins')


# ------------------------------------------------------------------------
if __name__ == '__main__':
    args = parser.parse_args()
    root_path = args.root

    # ------------- Construct required paths -------------
    dataset = args.dataset_name.split("_")[0]

    # Feature labels type
    if args.type_fl == 'None':
        data_path = '%s/datasets/%s_wn_dataset.csv' % (root_path, args.dataset_name)
    elif args.type_fl == 'dt':
        data_path = '%s/datasets/%s_dt_wn_dataset.csv' % (root_path, args.dataset_name)
    else:
        data_path = '%s/datasets/%s_lr_no_bias_wn_dataset.csv' % (root_path, args.dataset_name)

    # Trial file path
    trial_path = '%s/code/%s_%s_%s' % (root_path, args.model_name, args.type_learn, args.dataset_name)

    if 'FS_' in args.type_learn and args.model_name in ['CFSC', 'CFSC_ft']:
        trial_path = '%s_c=%.3g_trials' % (trial_path, args.lambda_a)
        print("{Model: %s\nDataset: %s\nFL: %s\nlambda_a: %s\nlearn_rate: %s\n}"
              % (args.model_name, args.dataset_name, args.type_fl, args.lambda_a, args.learn_rate))

    elif 'FS_' in args.type_learn and args.model_name in ['BiLSTM-ATT']:
        trial_path = '%s_c=%.3g_trials' % (trial_path, args.lambda_att)
        print("{Model: %s\nDataset: %s\nFL: %s\nlambda_att: %s\nlearn_rate: %s\n}"
              % (args.model_name, args.dataset_name, args.type_fl, args.lambda_att, args.learn_rate))

    elif 'FS_' in args.type_learn and args.model_name in ['RNP']:
        trial_path = '%s_c=%s_trials' % (trial_path, args.lambda_omega)
        print("{Model: %s\nDataset: %s\nFL: %s\nlambda_omega: %s\nlearn_rate: %s\n}"
              % (args.model_name, args.dataset_name, args.type_fl, args.lambda_omega, args.learn_rate))

    elif 'FS_' in args.type_learn and args.model_name in ['LR_l1', 'LR_l2']:
        trial_path = '%s_c=%.3g_trials' % (trial_path, args.lr_C)
        print("{Model: %s\nDataset: %s\nFL: %s\nC: %s\n}"
              % (args.model_name, args.dataset_name, args.type_fl, args.lr_C))

    elif 'FS_' in args.type_learn and args.model_name in ['LR_FS-CLF']:
        trial_path = '%s_c1=%.3g_c2=%.3g_trials' % (trial_path, args.fs_C, args.clf_C)
        print("{Model: %s\nDataset: %s\nFL: %s\nfs_C: %s\nclf_C: %s\n}"
              % (args.model_name, args.dataset_name, args.type_fl, args.fs_C, args.clf_C))

    elif 'FS_' in args.type_learn and args.model_name in ['DT']:
        trial_path = '%s_%s_%s_%s_%s_%s_trials' % (
            trial_path, args.max_depth, args.min_samples_split, args.min_samples_leaf, args.max_leaf_nodes, args.min_imp_dec)
        print("{Model: %s\nDataset: %s\nFL: %s\n}"
              % (args.model_name, args.dataset_name, args.type_fl))

    elif 'FS_' in args.type_learn and args.model_name in ['Rule_pos', 'Rule_neg', 'Rule_mix']:
        trial_path = '%s_%s_%s_%s_%s_%s_trials' % (
            trial_path, args.max_rules, args.max_rule_conds, args.max_total_conds, args.k, args.n_discretize_bins)
        print("{Model: %s\nDataset: %s\nFL: %s\n}"
              % (args.model_name, args.dataset_name, args.type_fl))

    else:
        trial_path = '%s_trials' % trial_path
        print("{Model: %s\nDataset: %s\nFL: %s\nlearn_rate: %s\n}"
              % (args.model_name, args.dataset_name, args.type_fl, args.learn_rate))

    # ------------- Start task -------------
    # Load dataset
    data_mgmt = DataMgmt(dataset, args.model_name, data_path, args.n_discretize_bins)
    X_raw = data_mgmt.load_data()
    data_mgmt.select_train_test(X_raw)

    a = collections.Counter(X_raw['Label'])

    # Run trials
    emb_service = EmbeddingService(dataset, args.model_name, args.type_learn)
    seeds = [args.seed + i for i in range(args.num_trial)]
    LwFL.run_trials(
        emb_service, data_mgmt, trial_path, args.num_trial, args.num_fold,
        args.input_dim, args.input_act, args.att_dim, args.att_act, args.hidden_dim, args.hidden_act,
        args.learn_rate, args.num_epoch, args.batch_size, args.regular_w, args.lambda_a,
        args.lstm_dim, args.lstm_att_act, args.lambda_att,
        args.lambda_omega,
        args.lr_C, args.fs_C, args.clf_C,
        args.max_depth, args.min_samples_split, args.min_samples_leaf, args.max_leaf_nodes, args.min_imp_dec,
        args.max_rules, args.max_rule_conds, args.max_total_conds, args.k)
