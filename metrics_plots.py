def pr_auc(y_pred, y_true):
    precision, recall, _ = precision_recall_curve(y_pred, y_true, pos_label=1)
    return auc(recall, precision)


def plot_pr_curve(y_pred, y_true):
    precision, recall, _ = precision_recall_curve(y_pred, y_true, pos_label=1)
    auc_val = auc(recall, precision)
    n_classes = y_pred.shape[-1]

    plt.figure(figsize=(10,10))
    plt.plot(recall, precision, c='orange', lw=3)
    plt.hlines(1/n_classes, 0, 1, linestyles='--', lw=3)
    plt.text(0.9, 1, 'AUC = ' + "{:.2f}".format(float(auc_val)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
