import numpy as np
from tqdm import tqdm


def calc_MCC(y_true, y_pred, thresh, epsilon=1e-7):
    # make it 0 or 1 of pos and neg
    y_pos_true = y_true
    y_pos_pred = y_pred > thresh
    y_neg_true = 1 - y_pos_true
    y_neg_pred = 1 - y_pos_pred

    # calc tp, tn, fp, fn
    tp = np.sum(y_pos_true * y_pos_pred)
    tn = np.sum(y_neg_true * y_neg_pred)
    fp = np.sum(y_neg_true * y_pos_pred)
    fn = np.sum(y_pos_true * y_neg_pred)

    # calc MCC
    MCC = (tp * tn - fp * fn) / (np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + epsilon)

    return MCC


def calc_best_MCC(y_trues, y_preds, bins=300):
    best_threshs = []
    # calc best MCC for each fold
    for y_true, y_pred in zip(y_trues, y_preds):
        _best_MCC = 0.
        best_thresh = 0.
        for thresh in tqdm(np.linspace(0.001, 0.999, bins)):
            MCC = calc_MCC(y_true, y_pred, thresh)
            if MCC > _best_MCC:
                _best_MCC = MCC
                best_thresh = thresh
        best_threshs.append(best_thresh)
    # calc best MCC for the whole data
    whole_y_true = np.concatenate(y_trues, axis=0)
    _y_preds = [(y_pred > best_thresh).astype(int)
                for y_pred, best_thresh in zip(y_preds, best_threshs)]
    whole_y_pred = np.concatenate(_y_preds, axis=0)
    best_MCC = calc_MCC(whole_y_true, whole_y_pred, 0.5)
    return best_MCC, best_threshs


def lgb_MCC(preds, train_data):
    y_true = train_data.get_label()
    y_pred = preds
    thresh = 0.5
    epsilon = 1e-7

    # make it 0 or 1 of pos and neg
    y_pos_true = y_true
    y_pos_pred = y_pred > thresh
    y_neg_true = 1 - y_pos_true
    y_neg_pred = 1 - y_pos_pred

    # calc tp, tn, fp, fn
    tp = np.sum(y_pos_true * y_pos_pred)
    tn = np.sum(y_neg_true * y_neg_pred)
    fp = np.sum(y_neg_true * y_pos_pred)
    fn = np.sum(y_pos_true * y_neg_pred)

    # calc MCC
    MCC = (tp * tn - fp * fn) / (np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + epsilon)

    return 'MCC', MCC, False
