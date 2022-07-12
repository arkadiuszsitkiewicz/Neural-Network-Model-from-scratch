from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def basic_metrics(y_true, y_preds, y_proba):
    y_true, y_preds, y_proba = y_true.flatten(), y_preds.flatten(), y_proba.flatten()

    metrics = dict()
    metrics["accuracy"] = accuracy_score(y_true, y_preds)
    metrics["f1"] = f1_score(y_true, y_preds)
    metrics["precision"] = precision_score(y_true, y_preds, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_preds)
    metrics["AUC"] = roc_auc_score(y_true, y_proba)
    return metrics
