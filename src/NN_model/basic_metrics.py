from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def basic_metrics(y_true, y_preds, y_proba, average='binary'):
    multiclass = True if y_proba.shape[0] > 1 else False
    y_proba, average = (y_proba.T, average) if multiclass else (y_proba.flatten(), 'binary')

    y_true, y_preds, = y_true.flatten(), y_preds.flatten()
    metrics = dict()
    metrics["accuracy"] = accuracy_score(y_true, y_preds)
    metrics["f1"] = f1_score(y_true, y_preds, average=average)
    metrics["precision"] = precision_score(y_true, y_preds, zero_division=0, average=average)
    metrics["recall"] = recall_score(y_true, y_preds, average=average)
    if multiclass:
        metrics["AUC"] = roc_auc_score(y_true, y_proba, multi_class='ovo')
    else:
        metrics["AUC"] = roc_auc_score(y_true, y_proba)
    return metrics
