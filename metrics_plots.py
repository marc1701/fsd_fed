from pytorch_lightning.metrics.functional import precision_recall_curve, roc, auc
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np

def pr_auc(y_pred, y_true):
    '''calculate PR-AUC metric - equivalent to mAP'''
    precision, recall, _ = precision_recall_curve(y_pred, y_true, pos_label=1)
    return auc(recall, precision)

def dprime(y_pred, y_true):
    fpr, tpr, _ = roc(y_pred, y_true, pos_label=1)
    auc_val = auc(fpr, tpr)
    return np.sqrt(2) * norm.ppf(auc_val)

def plot_pr_curve(y_pred, y_true):
    '''plots the precision-recall curve with AUC value'''
    precision, recall, _ = precision_recall_curve(y_pred, y_true, pos_label=1)
    auc_val = auc(recall, precision)
    n_classes = y_pred.shape[-1]

    plt.figure(figsize=(10,10))
    plt.plot(recall, precision, c='orange', lw=3)
    plt.hlines(1/n_classes, 0, 1, linestyles='--', lw=3)
    plt.text(0.9, 1, 'AUC = ' + "{:.2f}".format(float(auc_val)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')

def lwlrap(y_pred, y_true):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # adapted from code by Dan Ellis
    # https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8

    # convert y_true to boolean array
    y_true = np.array(y_true, dtype=bool)

    # skip samples with no labels so sklearn weighting is correct
    sample_weight = np.sum(y_true > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)

    overall_lwlrap = label_ranking_average_precision_score(
        y_true[nonzero_weight_sample_indices, :] > 0,
        y_pred[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices])

    return overall_lwlrap
