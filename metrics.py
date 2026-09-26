"""Comprehensive evaluation suite -- extends beyond accuracy/F1 because a fraud model
that only reports accuracy on a 95:5 class split is reporting almost nothing."""
import numpy as np
from sklearn import metrics as skm


def evaluate(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    out = {
        "threshold": threshold,
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
        "precision": skm.precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": skm.recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "npv": npv,
        "f1": skm.f1_score(y_true, y_pred, zero_division=0),
        "f2": skm.fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        "mcc": skm.matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0,
        "cohen_kappa": skm.cohen_kappa_score(y_true, y_pred),
        "roc_auc": skm.roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": skm.average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "log_loss": skm.log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "brier_score": skm.brier_score_loss(y_true, y_prob),
        "true_positives": int(tp), "false_positives": int(fp),
        "true_negatives": int(tn), "false_negatives": int(fn),
        "support_fraud": int(y_true.sum()), "support_legit": int((1 - y_true).sum()),
        "fraud_prevalence": float(y_true.mean()),
    }
    return out


def best_threshold_for_f1(y_true, y_prob):
    """Sweep thresholds and pick the one maximizing F1 -- reported alongside the default
    0.5 threshold since fraud triage usually wants a tuned operating point, not 0.5."""
    y_true = np.asarray(y_true).astype(int)
    prec, rec, thr = skm.precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_i = int(np.nanargmax(f1s[:-1])) if len(thr) else 0
    return float(thr[best_i]) if len(thr) else 0.5
