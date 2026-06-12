import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

def evaluate(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),    # sensitivity
        "f1":        f1_score(y_true, y_pred),
        "auc":       roc_auc_score(y_true, y_prob) if y_prob is not None else None
    }
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["specificity"] = tn / (tn + fp)
    return metrics, cm

def print_metrics(name, metrics):
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k:12s}: {v:.4f}")