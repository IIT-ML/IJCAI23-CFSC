import numpy as np
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, confusion_matrix, precision_score, recall_score


def custom_f1_macro(y_true, y_pred, fea_att_neg=None):
    num_fea = y_true.shape[1]
    f1_scores = []
    for i in range(num_fea):
        y_true_fea = y_true[:, i]
        y_pred_fea = y_pred[:, i]
        pos_label = 0 if i in fea_att_neg else 1

        f1_fea = f1_score(y_true_fea, y_pred_fea, pos_label=pos_label, zero_division=0)
        f1_scores.append(f1_fea)
    print("F1* per feature: [%s]" % ", ".join(['{:.4g}'.format(f) for f in f1_scores]))
    f1_macro = np.average(f1_scores)
    return f1_macro


def custom_f1_micro(y_true, y_pred, fea_att_neg=None):
    num_fea = y_true.shape[1]
    tps, fps, tns, fns = [], [], [], []
    for i in range(num_fea):
        y_true_fea = y_true[:, i]
        y_pred_fea = y_pred[:, i]
        pos_label = 0 if i in fea_att_neg else 1

        if pos_label:
            tn, fp, fn, tp = confusion_matrix(y_true_fea, y_pred_fea, labels=[0, 1]).ravel()
        else:
            tn, fp, fn, tp = confusion_matrix(y_true_fea, y_pred_fea, labels=[1, 0]).ravel()
        tps.append(tp), fps.append(fp), tns.append(tn), fns.append(fn)

    tp_avg, fp_avg, tn_avg, fn_avg = np.average(tps), np.average(fps), np.average(tns), np.average(fns)
    p_micro = tp_avg / (tp_avg + fp_avg)
    r_micro = tp_avg / (tp_avg + fn_avg)
    f1_micro = 2*p_micro*r_micro / (p_micro+r_micro)

    return f1_micro


def custom_acc_macro(y_true, y_pred, cost=None):
    num_fea = y_true.shape[1]
    acc_scores = []
    for i in range(num_fea):
        y_true_fea = y_true[:, i]
        y_pred_fea = y_pred[:, i]

        # Use cost matrix
        if cost is None:
            acc = accuracy_score(y_true_fea, y_pred_fea)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true_fea, y_pred_fea, labels=[0, 1]).ravel()
            if cost[i][0] > 0 and cost[i][1] > 0:
                if cost[i][1] == cost[i][0]:
                    fp_cost, fn_cost = 1, 1
                elif cost[i][1] > cost[i][0]:
                    fp_cost, fn_cost = (10, 1)
                else:
                    fp_cost, fn_cost = (1, 10)
            else:
                fp_cost, fn_cost = 1, 1

            acc = (tp + tn) / (tn + fp*fp_cost + fn*fn_cost + tp)

        acc_scores.append(acc)
    acc_macro = np.average(acc_scores)
    return acc_scores, acc_macro


def macro_averaged_mean_absolute_error(y_true, y_pred):
    labels = [0, 1]
    mae = []
    for possible_class in labels:
        indices = np.flatnonzero(y_true == possible_class)

        if len(indices) > 0:
            mae.append(float(mean_absolute_error(y_true[indices], y_pred[indices])))
        else:
            mae.append(float(-1))
    return mae


def macro_averaged_accuracy(y_true, y_pred):
    labels = [0, 1]
    acc = []
    for possible_class in labels:
        indices = np.flatnonzero(y_true == possible_class)

        if len(indices) > 0:
            acc.append(float(accuracy_score(y_true[indices], y_pred[indices])))
        else:
            acc.append(float(-1))
    return acc


def custom_confusion_matrix(y_true, y_pred, threshold=None):
    if threshold is not None:
        y_pred = y_pred > threshold

    pos_idx = np.flatnonzero(y_true == 1)
    tp = -1 if len(pos_idx) == 0 else len(np.flatnonzero(y_true[pos_idx] == y_pred[pos_idx]))
    fn = -1 if len(pos_idx) == 0 else len(y_true[pos_idx]) - tp

    neg_idx = np.flatnonzero(y_true == 0)
    tn = -1 if len(neg_idx) == 0 else len(np.flatnonzero(y_true[neg_idx] == y_pred[neg_idx]))
    fp = -1 if len(neg_idx) == 0 else len(y_true[neg_idx]) - tn

    return tn, fp, fn, tp


def custom_acc_p_r_f1(y_true, y_pred, threshold=None):
    if threshold is not None:
        y_pred = y_pred > threshold

    pos_idx = np.flatnonzero(y_true == 1)
    tp = 0 if len(pos_idx) == 0 else len(np.flatnonzero(y_true[pos_idx] == y_pred[pos_idx]))
    fn = 0 if len(pos_idx) == 0 else len(y_true[pos_idx]) - tp

    neg_idx = np.flatnonzero(y_true == 0)
    tn = 0 if len(neg_idx) == 0 else len(np.flatnonzero(y_true[neg_idx] == y_pred[neg_idx]))
    fp = 0 if len(neg_idx) == 0 else len(y_true[neg_idx]) - tn

    acc = (tn + tp) / len(y_true)
    P = tp / (tp + fp) if (tp + fp) > 0 else 0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0
    F1 = 2*P*R / (P+R) if (P+R) > 0 else 0

    return acc, P, R, F1


def custom_group_score(y_true, y_prob):
    num_fea = y_true.shape[1]
    score_zero, score_normal = [], []
    for i in range(num_fea):
        y_true_fea = y_true[:, i]
        y_prob_fea = y_prob[:, i]

        # score = macro_averaged_mean_absolute_error(y_true_fea.ravel(), y_prob_fea.ravel())
        # score = macro_averaged_accuracy(y_true_fea.ravel(), y_prob_fea.ravel())
        score = custom_confusion_matrix(y_true_fea.ravel(), y_prob_fea.ravel())

        if len(np.unique(y_true_fea)) == 1:
            score_zero.append(score)
        else:
            score_normal.append(score)

    return score_zero, score_normal


def custom_group_curve(y_true, y_prob):
    num_fea = y_true.shape[1]
    score_zero_on, score_zero_off, score_normal = [], [], []

    y_prob_flat = y_prob.flatten()
    desc_score_indices = np.argsort(y_prob_flat)[::-1]
    y_prob_flat = y_prob_flat[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_prob_flat))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # TODO:
    # threshold_idxs = threshold_idxs[-30:]

    for i in range(num_fea):
        y_true_fea = y_true[:, i]
        y_prob_fea = y_prob[:, i]

        score = [custom_acc_p_r_f1(y_true_fea.ravel(), y_prob_fea.ravel(), y_prob_flat[idx]) for idx in threshold_idxs]
        score = np.asarray(score)

        if len(np.unique(y_true_fea)) == 1 and y_true_fea[0] == 1:
            score_zero_on.append(score[:, 0])
        elif len(np.unique(y_true_fea)) == 1 and y_true_fea[0] == 0:
            score_zero_off.append(score[:, 0])
        else:
            score_normal.append(score[:, 1:4])

    score_zero_on, score_zero_off, score_normal = np.asarray(score_zero_on), np.asarray(score_zero_off), np.asarray(score_normal)
    score_zero_on, score_zero_off, score_normal = np.average(score_zero_on, axis=0), np.average(score_zero_off, axis=0), np.average(score_normal, axis=0)

    return score_zero_on, score_zero_off, score_normal, y_prob_flat[threshold_idxs]


def custom_sparsity(y_att_pred):
    sparsity = np.sum(y_att_pred) / (y_att_pred.shape[0]*y_att_pred.shape[1])
    return sparsity


def custom_att_metrics_dict(y_true, y_pred):
    num_fea = y_true.shape[1]
    normal_idx = [i for i in range(num_fea) if len(np.unique(y_true[:, i])) > 1]
    zero_on_idx = [i for i in range(num_fea) if len(np.unique(y_true[:, i])) == 1 and np.unique(y_true[:, i])[0] == 1]
    zero_off_idx = [i for i in range(num_fea) if len(np.unique(y_true[:, i])) == 1 and np.unique(y_true[:, i])[0] == 0]

    # Acc
    zero_on_acc = np.average([accuracy_score(y_true[:, i].ravel(), y_pred[:, i].ravel()) for i in zero_on_idx]) \
        if len(zero_on_idx) > 0 else np.nan
    zero_off_acc = np.average([accuracy_score(y_true[:, i].ravel(), y_pred[:, i].ravel()) for i in zero_off_idx]) \
        if len(zero_off_idx) > 0 else np.nan

    # P, R, F1
    normal_avg_P = precision_score(y_true[:, normal_idx], y_pred[:, normal_idx], zero_division=0, average='macro')
    normal_w_P = precision_score(y_true[:, normal_idx], y_pred[:, normal_idx], zero_division=0, average='weighted')

    normal_avg_R = recall_score(y_true[:, normal_idx], y_pred[:, normal_idx], zero_division=0, average='macro')
    normal_w_R = recall_score(y_true[:, normal_idx], y_pred[:, normal_idx], zero_division=0, average='weighted')

    normal_avg_F1 = f1_score(y_true[:, normal_idx], y_pred[:, normal_idx], zero_division=0, average='macro')
    normal_w_F1 = f1_score(y_true[:, normal_idx], y_pred[:, normal_idx], zero_division=0, average='weighted')

    # Sparsity
    sparsity = custom_sparsity(y_pred)

    return {'zero_on_acc': zero_on_acc, 'zero_off_acc': zero_off_acc,
            'normal_macro_P': normal_avg_P, 'normal_w_P': normal_w_P,
            'normal_macro_R': normal_avg_R, 'normal_w_R': normal_w_R,
            'normal_macro_F1': normal_avg_F1, 'normal_w_F1': normal_w_F1,
            'sparsity': sparsity}
