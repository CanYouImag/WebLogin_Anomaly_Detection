import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


def binary_metrics(y_true, y_pred, y_prob=None):
    conf = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, zero_division=0),
        "tpr": tpr,
        "fpr": fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if y_prob is not None:
        try:
            out["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            out["auc"] = np.nan
    return out


def multiclass_metrics(y_true, y_pred, class_names=None, y_prob=None, average="macro"):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True,
        target_names=[str(i) for i in labels],
    )
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        f"precision_{average}": precision_score(y_true, y_pred, average=average, zero_division=0),
        f"recall_{average}": recall_score(y_true, y_pred, average=average, zero_division=0),
        f"f1_{average}": f1_score(y_true, y_pred, average=average, zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_prob is not None:
        try:
            n_classes = y_prob.shape[1]
            out["auc_ovr_macro"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro", labels=labels
            )
        except ValueError:
            out["auc_ovr_macro"] = np.nan
    per_class = []
    conf = confusion_matrix(y_true, y_pred, labels=labels)
    for i, c in enumerate(labels):
        name = class_names[c] if class_names is not None and c < len(class_names) else str(c)
        tp = conf[i, i]
        fn = conf[i, :].sum() - tp
        fp = conf[:, i].sum() - tp
        tn = conf.sum() - tp - fn - fp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append({
            "class": name,
            "support": int(conf[i, :].sum()),
            "tp": int(tp),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        })
    return out, per_class


def save_per_class(per_class, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(per_class).to_csv(path, index=False, float_format="%.4f")
    return path


def mcnemar(y_true, y_pred_a, y_pred_b):
    b_c = np.logical_and(y_pred_a != y_true, y_pred_b == y_true).sum()
    c_b = np.logical_and(y_pred_a == y_true, y_pred_b != y_true).sum()
    denom = b_c + c_b
    if denom == 0:
        return 1.0
    stat = (abs(b_c - c_b) - 1) ** 2 / denom
    from scipy.stats import chi2

    return 1 - chi2.cdf(stat, df=1)


def save_results_table(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4f")
    return path


def class_names_for(dataset):
    if dataset == "cicids2017":
        return [
            "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk",
            "DoS Slowhttptest", "DoS slowloris", "FTP-Patator", "Other_Attack",
            "PortScan", "SSH-Patator", "Web Attack Brute Force", "Web Attack XSS",
        ]
    if dataset == "unsw_nb15":
        return ["Normal", "Attack"]
    if dataset == "nsl_kdd":
        return ["Attack", "Normal"]
    return None