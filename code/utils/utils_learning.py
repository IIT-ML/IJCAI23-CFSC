import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from scipy.stats import stats


def list_inst_stack(data, query_inst):
    # Stack X
    X_fea_new = np.concatenate((data['X'][0], query_inst['X'][0]), axis=0)
    X_word_new = np.concatenate((data['X'][1], query_inst['X'][1]), axis=0)

    # Stack mask
    X_att_mask_new = np.concatenate((data['X_mask'], query_inst['X_mask']), axis=0)
    X_fea_mask_new = np.concatenate((data['X_fea_mask'], query_inst['X_fea_mask']), axis=0)

    # Stack y
    y_inst_new = np.concatenate((data['y_inst'], query_inst['y_inst']), axis=0)
    y_att_new = np.concatenate((data['y_att'], query_inst['y_att']), axis=0) if data['y_att'] is not None else None

    input_dict = {
        "X": [X_fea_new, X_word_new],
        "X_mask": X_att_mask_new, "X_fea_mask": X_fea_mask_new,
        "y_inst": y_inst_new, "y_att": y_att_new
    }
    return input_dict


def ndarray_inst_stack(data, query_inst):
    # Stack X
    X_new = np.concatenate((data['X'], query_inst['X']), axis=0)

    # Stack y
    y_inst_new = np.concatenate((data['y_inst'], query_inst['y_inst']), axis=0)
    y_att_new = np.concatenate((data['y_att'], query_inst['y_att']), axis=0) if data['y_att'] is not None else None

    input_dict = {
        "X": X_new,
        "X_mask": data['X_mask'], "X_fea_mask": data['X_fea_mask'],
        "y_inst": y_inst_new, "y_att": y_att_new
    }
    return input_dict


def obj_ndarray_inst_stack(data, query_inst):
    # Stack X
    X_new, X_prime_new, X_rat_new = list(data['X']), list(data['X_prime']), list(data['X_rat'])
    q = [list(query_inst['X'][0])] if len(query_inst['X']) == 1 else list(query_inst['X'])
    q_prime = [list(query_inst['X_prime'][0])] if len(query_inst['X_prime']) == 1 else list(query_inst['X_prime'])
    q_rat = [list(query_inst['X_rat'][0])] if len(query_inst['X_rat']) == 1 else list(query_inst['X_rat'])
    X_new.extend(q)
    X_prime_new.extend(q_prime)
    X_rat_new.extend(q_rat)

    # Stack y
    y_inst_new = np.concatenate((data['y_inst'], query_inst['y_inst']), axis=0)
    y_att_new = np.concatenate((data['y_att'], query_inst['y_att']), axis=0) if data['y_att'] is not None else None

    input_dict = {
        "X": np.asarray(X_new), "X_prime": np.asarray(X_prime_new), "X_rat": np.asarray(X_rat_new),
        "y_inst": y_inst_new, "y_att": y_att_new, "emb_matrix": data['emb_matrix']
    }
    return input_dict


def extract_feature_neg_attention(dataset, fea_names, y_att):
    # if dataset == 'synthetic':
    #     return [1, 2, 4]

    on_idx, on_fea, off_idx, off_fea = [], [], [], []
    for i, fea in enumerate(fea_names):
        count = Counter(y_att[:, i])
        if len(count.keys()) == 1 and np.max(y_att[:, i]) == 0:
            off_idx.append(i), off_fea.append(fea)
        if len(count.keys()) == 1 and np.max(y_att[:, i]) == 1:
            on_idx.append(i), on_fea.append(fea)
    print("%s Always on features:\n%s\n%s" % (len(on_idx), on_idx, on_fea))
    print("%s Always off features:\n%s\n%s" % (len(off_idx), off_idx, off_fea))
    print()


def save_test_results_with_explanation(exp_path, clf, fea_names, X_test, test_data):
    # Get predicted test attentions
    if "LR" in exp_path or "Rule" in exp_path or "DT" in exp_path:
        y_pred, y_att_pred = clf.predict(X_test)
    else:
        y_pred, y_att_pred = clf.predict_label(X_test)

    y_att_pred_label = [np.asarray(fea_names)[np.nonzero(y)[0]] for y in y_att_pred]

    # with open("feature_names.json", "w") as f:
    #     a = {'feature_name': fea_names}
    #     json.dump(a, f)

    test_df = pd.DataFrame(np.squeeze(X_test), columns=fea_names)
    test_df['Label'] = test_data[1]
    test_df['Narrative'] = test_data[0]['Narrative'].values
    test_df['Predict_label'] = [int(y) for y in y_pred]
    test_df['Predict_Att'] = y_att_pred_label
    test_df.to_csv(exp_path, index=False)


def save_intmd(trial_path, res_dict_list):
    contents_list = []
    for res_dict in res_dict_list:
        if 'intmd' not in res_dict.keys():
            return

        contents = defaultdict(list)
        for item in res_dict['intmd']:
            for k, v in item.items():
                contents[k].append(v)
        contents_list.append(contents)
    with open(trial_path+"_loss", 'w') as f:
        json.dump(contents_list, f, indent=4)


def save_trial(trial_path, res_dict_list):
    contents_list = []
    for budget_idx in range(len(res_dict_list[0]['num_doc'])):
        contents = defaultdict(list)
        for trial_idx, res_dict in enumerate(res_dict_list):
            for k, v_list in res_dict.items():
                if k != 'intmd':
                    contents[k].append(v_list[budget_idx])
        contents['num_doc'] = np.mean(contents['num_doc'], dtype=int).tolist()
        contents_list.append(contents)

    with open(trial_path, 'w') as f:
        json.dump(contents_list, f, indent=4)


def load_save_point(trials_path):
    """
    Restore training process from saved trial file
    """
    cur_pos, len_data_point = 0, 0
    res_dict_list = []
    if not os.path.exists(trials_path):
        return cur_pos, res_dict_list

    with open(trials_path, 'r') as f:
        files = json.load(f)
        cur_pos, len_data_point = len(files[0]['acc']), len(files)
        for trial_idx in range(cur_pos):
            res_dict = defaultdict(list)
            for budget_row in files:
                for k, v in budget_row.items():
                    res_dict[k].append(v[trial_idx] if k != 'num_doc' else v)
            res_dict_list.append(res_dict)

    if os.path.exists(trials_path+"_loss"):
        with open(trials_path+"_loss", 'r') as f:
            files = json.load(f)
            for trial_idx, trial_row in enumerate(files):
                res_dict_list[trial_idx]['intmd'] = [{} for _ in range(len_data_point)]
                for k, v in trial_row.items():
                    for i in range(len_data_point):
                        res_dict_list[trial_idx]['intmd'][i][k] = v[i]
    return cur_pos, res_dict_list


def print_top_weight_feature(emb_service, weight_arr, top=20):
    # Binary class
    for c in range(len(emb_service.label_vocab.word2id.items())):
        top_idx = np.argsort((-1)**c * weight_arr)[:top]
        top_w = weight_arr[top_idx]
        top_word = emb_service.word_vocab.get_words(top_idx)
        top_pair = [(word, weight) for word, weight in zip(top_word, top_w)]
        print("Top %s %s words:" % (top, emb_service.label_vocab.get_word(c)))
        print(top_pair)


def plot_result(trial_path, plot_path, plot_title):
    # Load from trial file
    with open(trial_path, 'r') as f:
        file = json.load(f)
        rows, col_acc = [], []
        for line in file:
            rows.append(line['num_doc'])
            col_acc.append(line['acc'])
            # if 'LwN' in trial_path:
            #     col_att_hl.append(line['att_hl']), col_att_acc_macro.append(line['att_acc_macro'])
            #     col_att_acc.append(line['att_acc'])

    y_acc = [np.average(i) for i in col_acc]

    # Print Att accuracy per feature
    # if 'all' in trial_path:
    #     y_att_acc = [np.average(np.asarray(i), axis=0) for i in col_att_acc]
    #     for att_fea in y_att_acc:
    #         print("Accuracy per feature: [%s]" % ", ".join(['{:.4g}'.format(s) for s in att_fea]))

    # Plot
    plt.figure(figsize=(12, 7))
    plt.title(plot_title)
    plt.ylabel('Score')
    plt.xlabel('Number of documents')
    # plt.plot(rows, y_auc, color='r', label='AUC')
    plt.plot(rows, y_acc, color='b', label='Accuracy')
    # if 'LwN' in trial_path:
    #     plt.plot(rows, y_att_hl, color='m', label='att_hl')
    #     plt.plot(rows, y_att_acc_macro, color='k', label='att_acc_macro')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)


def t_test(col1, col2):
    p_list, t_list = [], []
    for samp1, samp2 in zip(col1, col2):
        t, p = stats.ttest_rel(samp1, samp2)
        p_list.append(p)
        t_list.append(t)

    cell = np.zeros((2, len(col1)))
    for i in range(len(col1)):
        cell[0][i] = round(p_list[i], 3)
        cell[1][i] = round(t_list[i], 3)
    return cell


# def train_input_rules(file_path, trial_idx, trained_clf, train_input, val_input, test_input, fea_names, dataset, seed):
#     post_clf_list, eval_list = {}, {}
#
#     # Construct training set
#     use_set = 'train'
#     print("Using %s set" % use_set)
#     if use_set == 'val':
#         att_prob = trained_clf.predict_prob(val_input['X'])
#         X_train = np.squeeze(val_input['X'])
#     else:
#         att_prob = trained_clf.predict_prob(train_input['X'])
#         X_train = np.squeeze(train_input['X'])
#
#     # Construct test set
#     test_att_prob, val_att_prob = [], []
#     X_test, y_test = np.squeeze(test_input['X']), test_input['y_att']
#     X_val, y_val = np.squeeze(val_input['X']), val_input['y_att']
#     used_fea = list(range(len(fea_names)))
#
#     y_val_prob = trained_clf.predict_prob(val_input['X'])
#     y_test_prob = trained_clf.predict_prob(test_input['X'])
#
#     # Train independent clf for each feature
#     for fea_idx in used_fea:
#         fea = fea_names[fea_idx]
#         y_train = att_prob[:, fea_idx]
#
#         # Fit interpretable model
#         post_clf = Post_exp_Model(model_name='lasso', seed=seed, learn_rate=1e-3, batch_size=64, num_epoch=100)
#         # post_clf.fit(X_train, y_train)
#
#         # Save figures
#         if not os.path.exists("%s_trial%s" % (dataset, trial_idx)):
#             os.mkdir("%s_trial%s" % (dataset, trial_idx))
#         if post_clf.model_name == 'dt':
#             tree.plot_tree(post_clf.model, filled=True, node_ids=True, fontsize=8, feature_names=fea_names)
#             plt.savefig("%s_trial%s/%s_%s_att_dt.png" % (dataset, trial_idx, fea_idx, fea))
#         elif post_clf.model_name == 'lasso':
#             # Probability histogram on training set
#             plt.rcParams.update({'font.size': 12})
#             plt.hist(y_train.flatten(), range=[0, 1])
#             plt.title("Train - %s %s" % (fea_idx, fea))
#             plt.savefig("%s_trial%s/train_%s_%s_att_hist.png" % (dataset, trial_idx, fea_idx, fea))
#             plt.clf()
#
#             # Probability histogram on val set
#             if use_set != 'val':
#                 # y_val_prob = trained_clf.predict_prob(val_input['X'])
#                 # val_att_prob.extend(y_val_prob)
#                 plt.rcParams.update({'font.size': 12})
#                 plt.hist(y_val_prob[:, fea_idx].flatten(), range=[0, 1])
#                 plt.title("Val - %s %s" % (fea_idx, fea))
#                 plt.savefig("%s_trial%s/val_%s_%s_att_hist.png" % (dataset, trial_idx, fea_idx, fea))
#                 plt.clf()
#
#             # Probability histogram on test set
#             # y_test_prob = trained_clf.predict_prob(test_input['X'])
#             # test_att_prob.extend(y_test_prob[:, fea_idx].flatten())
#             plt.rcParams.update({'font.size': 12})
#             plt.hist(y_test_prob[:, fea_idx].flatten(), range=[0, 1])
#             plt.title("Test - %s %s" % (fea_idx, fea))
#             plt.savefig("%s_trial%s/test_%s_%s_att_hist.png" % (dataset, trial_idx, fea_idx, fea))
#             plt.clf()
#
#             # # Ground truth histogram
#             # plt.rcParams.update({'font.size': 12})
#             # plt.hist(val_input['y_att'][:, fea_idx].flatten(), range=[0, 1])
#             # plt.title("%s %s (True label)" % (fea_idx, fea))
#             # plt.savefig("%s_trial%s/true_%s_%s_att_hist.png" % (dataset, trial_idx, fea_idx, fea))
#             # plt.clf()
#
#         # Evaluate
#         # eval_res = post_clf.evaluate(X_test, y_test[:, fea_idx])
#         # post_clf_list[fea] = post_clf
#         # eval_list[fea] = eval_res
#         # if use_set != 'val':
#         #     val_eval_res = post_clf.evaluate(X_val, y_val[:, fea_idx])
#         #     for val_k, val_v in val_eval_res.items():
#         #         if val_k != 'weight':
#         #             eval_list[fea]["val_" + val_k] = val_v
#         #
#         # time_count = Counter(y_train > 0.5)
#         # eval_list[fea]['id'] = fea_idx
#         # eval_list[fea]['count_on'] = time_count[True]
#         # eval_list[fea]['count_off'] = time_count[False]
#         # eval_list[fea]['%_on'] = time_count[True] / (eval_list[fea]['count_on'] + eval_list[fea]['count_off'])
#         # eval_list[fea]['%_off'] = time_count[False] / (eval_list[fea]['count_on'] + eval_list[fea]['count_off'])
#
#     # Save histogram for all attentions
#     plt.hist(att_prob.flatten(), range=[0, 1])
#     plt.title("Train - All features")
#     plt.savefig("%s_trial%s/train_att_hist.png" % (dataset, trial_idx))
#     plt.clf()
#
#     if use_set != 'val':
#         plt.hist(val_att_prob, range=[0, 1])
#         plt.title("Val - All features")
#         plt.savefig("%s_trial%s/val_att_hist.png" % (dataset, trial_idx))
#         plt.clf()
#
#     plt.hist(test_att_prob, range=[0, 1])
#     plt.title("Test - All features")
#     plt.savefig("%s_trial%s/test_att_hist.png" % (dataset, trial_idx))
#     plt.clf()
#
#     # with open("%s_trial%s" % (file_path, trial_idx), 'w') as f:
#     #     json.dump(eval_list, f, indent=4)
#     print()
