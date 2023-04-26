import numpy as np
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import StratifiedKFold

import utils.utils_learning as utils_learning

from model.model_FF import FF_Model
from model.model_CLF import CLF_Model
from model.model_BiLSTM_Att import BiLSTMAtt_Model
from model.model_CFSC import CFSC_Model
from model.model_Rationale_Net import Rationale_Net_Model
from model.model_FS_CLF_Pipeline import FS_CLF_model


def build_input(X_data, y_data, emb_service, mode='train'):
    X_fea, y_inst, y_att = emb_service.extract_func(X_data, y_data, mode)
    model_name, learn_strategy = emb_service.model_name, emb_service.learn_strategy

    # Build y
    y_inst = emb_service.build_y_label(y_inst)

    # Build input
    if model_name in ['CFSC', 'CFSC_ft', 'CFSC-ATT', 'BiLSTM-ATT', 'RNP']:
        X_fea = np.reshape(X_fea, newshape=(len(X_fea), len(X_fea[0]), 1))
    X_input, y_att = np.asarray(X_fea, dtype=np.float32), np.asarray(y_att, dtype=np.float32)

    input_dict = {"X": X_input,
                  "y_inst": y_inst, "y_att": y_att}
    return input_dict


def learning_step(emb_service, clf, train_input, val_input, test_input, cur_num_doc, res_dict):
    if cur_num_doc > 0:
        # Fit
        clf.fit(
            train_input['X'], (train_input['y_inst'], train_input['y_att']),
            val_input['X'], (val_input['y_inst'], val_input['y_att']),
            test_input['X'], (test_input['y_inst'], test_input['y_att']))

        # Print evaluate metrics
        evl_res = clf.evaluate(test_input['X'], test_input['y_inst'], test_input['y_att'])
        for k, v in evl_res.items():
            res_dict[k].append(v)
            print("%s docs Test %s: %s" % (cur_num_doc, k, v))
        res_dict['intmd'].append(clf.intmd_res)
        res_dict['num_doc'].append(cur_num_doc)
        print()
    return res_dict


def train(emb_service, train_data, test_data, val_data,
          input_dim, input_act, att_dim, att_act, hidden_dim, hidden_act,
          learn_rate, num_epoch, batch_size, regular_w, lambda_a,
          lstm_dim, lstm_att_act, lambda_att,
          lambda_omega,
          lr_C, fs_C, clf_C,
          max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_imp_dec,
          max_rules, max_rule_conds, max_total_conds, k):

    res_dict = defaultdict(list)

    # Build input
    train_input = build_input(train_data[0], train_data[1], emb_service, mode='train')
    test_input = build_input(test_data[0], test_data[1], emb_service, mode='test')
    val_input = build_input(val_data[0], val_data[1], emb_service, mode='val')

    # Model
    clf = None
    model_name, learn_strategy = emb_service.model_name, emb_service.learn_strategy
    num_class, num_fl = emb_service.num_class, emb_service.num_fl
    if model_name == 'BiLSTM-ATT':
        clf = BiLSTMAtt_Model(model_name, num_class, num_fl, input_dim, input_act, att_dim, lstm_dim, lstm_att_act,
                              learn_rate, num_epoch, batch_size, regular_w, lambda_att)

    elif model_name in ['LR_l1', 'LR_l2', 'DT', 'Rule_pos', 'Rule_neg', 'Rule_mix']:
        clf = CLF_Model(model_name, num_class, num_fl,
                        lr_C,
                        max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_imp_dec,
                        max_rules, max_rule_conds, max_total_conds, k)

    elif model_name in ['RNP']:
        clf = Rationale_Net_Model(model_name, num_class, num_fl, input_dim, input_act, att_dim, att_act, hidden_dim, hidden_act,
                                  learn_rate, num_epoch, batch_size, lambda_omega)

    elif model_name in ['LR_FS-CLF']:
        clf = FS_CLF_model(model_name, num_class, num_fl, fs_C, clf_C)

    elif model_name == 'FF':
        clf = FF_Model(model_name, num_class, num_fl, hidden_dim, hidden_act, learn_rate, num_epoch, batch_size)

    elif model_name in ['CFSC', 'CFSC_ft']:
        # Our model
        clf = CFSC_Model(model_name, num_class, num_fl, input_dim, input_act, att_dim, att_act, hidden_dim, hidden_act,
                         learn_rate, num_epoch, batch_size, lambda_a)

    # Learning
    res_dict = learning_step(emb_service, clf, train_input, val_input, test_input, train_input['X'].shape[0], res_dict)

    return clf, res_dict


def cross_val_trial(emb_service, data_mgmt, trial_path, num_fold,
                    input_dim, input_act, att_dim, att_act, hidden_dim, hidden_act,
                    learn_rate, num_epoch, batch_size, regular_w, lambda_a,
                    lstm_dim, lstm_att_act, lambda_att,
                    lambda_omega,
                    lr_C, fs_C, clf_C,
                    max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_imp_dec,
                    max_rules, max_rule_conds, max_total_conds, k):

    sk = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=1234)

    # Combine all data for k-fold split
    total_X_data = pd.concat([data_mgmt.X_train, data_mgmt.X_val]).reset_index(drop=True)
    total_y_data = np.concatenate([data_mgmt.y_train, data_mgmt.y_val])
    test_data = [data_mgmt.X_test, data_mgmt.y_test]

    cur_pos, cur_res_dict_list = utils_learning.load_save_point(trial_path)
    for fold_idx, (train_idx, val_idx) in enumerate(sk.split(total_X_data, total_y_data)):
        # if fold_idx > 0:
        #     break

        if cur_pos > fold_idx:
            continue
        print("\nk-fold Trial %s:" % (fold_idx + 1))

        # Build input for each iteration
        train_data = [total_X_data.iloc[train_idx], total_y_data[train_idx]]
        val_data = [total_X_data.iloc[val_idx], total_y_data[val_idx]]

        # Run one trial
        clf, res_dict = train(
            emb_service, train_data, test_data, val_data,
            input_dim, input_act, att_dim, att_act, hidden_dim, hidden_act,
            learn_rate, num_epoch, batch_size, regular_w, lambda_a,
            lstm_dim, lstm_att_act, lambda_att,
            lambda_omega,
            lr_C, fs_C, clf_C,
            max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_imp_dec,
            max_rules, max_rule_conds, max_total_conds, k)
        cur_res_dict_list.append(res_dict)

        # Save results
        utils_learning.save_trial(trial_path, cur_res_dict_list)
        utils_learning.save_intmd(trial_path, cur_res_dict_list)

        # # Post-processing for LwN
        # if 'FS' in emb_service.learn_strategy:
        #     # train_input = build_input(train_data[0], train_data[1], emb_service, mode='train')
        #     # val_input = build_input(val_data[0], val_data[1], emb_service, mode='test')
        #     test_input = build_input(test_data[0], test_data[1], emb_service, mode='test')
        #
        #     # # Train classifier per feature to generate rules in input space
        #     # interp_path = '%s_interpretation' % trial_path
        #     # train_input_rules(interp_path, trial_idx, clf, train_input, val_input, test_input, emb_service.fea_names, emb_service.dataset)
        #
        #     # Save explanation results
        #     exp_path = '%s_exp.csv' % trial_path
        #     utils_learning.save_test_results_with_explanation(exp_path, clf, emb_service.fea_name_list, test_input['X'], test_data)
        #     print()


def run_trials(emb_service, data_mgmt, trial_path, num_trial, num_fold,
               input_dim, input_act, att_dim, att_act, hidden_dim, hidden_act,
               learn_rate, num_epoch, batch_size, regular_w, lambda_a,
               lstm_dim, lstm_att_act, lambda_att,
               lambda_omega,
               lr_C, fs_C, clf_C,
               max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_imp_dec,
               max_rules, max_rule_conds, max_total_conds, k):

    if num_fold > 0:
        # k-fold cross validation
        cross_val_trial(
            emb_service, data_mgmt, trial_path, num_fold,
            input_dim, input_act, att_dim, att_act, hidden_dim, hidden_act, learn_rate, num_epoch, batch_size, regular_w, lambda_a,
            lstm_dim, lstm_att_act, lambda_att,
            lambda_omega,
            lr_C, fs_C, clf_C,
            max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_imp_dec,
            max_rules, max_rule_conds, max_total_conds, k)

    else:
        # train-test-val validation
        train_data = [data_mgmt.X_train, data_mgmt.y_train]
        test_data = [data_mgmt.X_test, data_mgmt.y_test]
        val_data = [data_mgmt.X_val, data_mgmt.y_val]

        cur_pos, cur_res_dict_list = utils_learning.load_save_point(trial_path)
        for trial_idx in range(cur_pos, num_trial):
            # Run one trial
            print("\nTrial %s:" % (trial_idx+1))
            clf, res_dict = train(
                emb_service, train_data, test_data, val_data,
                input_dim, input_act, att_dim, att_act, hidden_dim, hidden_act,
                learn_rate, num_epoch, batch_size, regular_w, lambda_a,
                lstm_dim, lstm_att_act, lambda_att,
                lambda_omega,
                lr_C, fs_C, clf_C,
                max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_imp_dec,
                max_rules, max_rule_conds, max_total_conds, k)
            cur_res_dict_list.append(res_dict)

            # Save results
            utils_learning.save_trial(trial_path, cur_res_dict_list)
            utils_learning.save_intmd(trial_path, cur_res_dict_list)

            # Post-processing for LwN
            # if 'FS' in emb_service.learn_strategy:
            #     train_input = build_input(train_data[0], train_data[1], emb_service, mode='train')
            #     val_input = build_input(val_data[0], val_data[1], emb_service, mode='test')
            #     test_input = build_input(test_data[0], test_data[1], emb_service, mode='test')
            #
            #     # # Train classifier per feature to generate rules in input space
            #     # interp_path = '%s_interpretation' % trial_path
            #     # train_input_rules(interp_path, trial_idx, clf, train_input, val_input, test_input, emb_service.fea_names, emb_service.dataset)
            #
            #     # Save explanation results
            #     exp_path = '%s_exp.csv' % trial_path
            #     utils_learning.save_test_results_with_explanation(exp_path, clf, emb_service.fea_names, test_input['X'], test_data)
            #     print()
