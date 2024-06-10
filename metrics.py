import evaluate
import numpy as np
from sklearn.metrics import cohen_kappa_score

accuracy = evaluate.load("accuracy")

def compute_metrics_accuracy(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    results = {
            'accuracy': accuracy.compute(predictions=predictions, references=labels)
        }
    return results

def compute_metrics_qwk(eval_pred):
    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.argmax(-1), weights='quadratic')
    results = {
        'qwk': qwk
    }
    return results

def compute_metrics_qwk_for_regression(eval_pred):
    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.clip(0,5).round(0), weights='quadratic')
    results = {
        'qwk': qwk
    }
    return results

def get_metric(_args):
    if _args.metric == "accuracy":
        return compute_metrics_accuracy
    elif _args.metric == "qwk":
        if _args.use_regression == "True":
            print("Using QWK regression metric")
            return compute_metrics_qwk_for_regression
        else:
            print("Using QWK classification metric")
            return compute_metrics_qwk
    else:
        raise ValueError("Undefined metric")