import evaluate
import numpy as np
from sklearn.metrics import cohen_kappa_score
import optuna
import pandas as pd
from numba import jit 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING) 

@jit
def qwk3(a1, a2, max_rat=5):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


class OptunaRounder:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = np.array([0, 1, 2, 3, 4, 5])

    def __call__(self, trial):
        thresholds = []
        for i in range(len(self.labels) - 1):
            low = max(thresholds) if i > 0 else min(self.labels)
            high = max(self.labels)
            t = trial.suggest_uniform(f't{i}', low, high)
            thresholds.append(t)
        try:
            opt_y_pred = self.adjust(self.y_pred, thresholds)
        except: return 0
        return qwk3(self.y_true, opt_y_pred)

    def adjust(self, y_pred, thresholds):
        opt_y_pred = pd.cut(y_pred, [-np.inf] + thresholds + [np.inf], labels=self.labels)
        return opt_y_pred


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


def compute_metrics_qwk_for_regression_with_optuna(eval_pred):
    predictions, labels = eval_pred
    objective = OptunaRounder(labels, predictions)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, timeout=180)
    best_thresholds = sorted(study.best_params.values())
    best_thresholds = [i for i in best_thresholds]
    preds_opt = objective.adjust(predictions, best_thresholds)
    preds_opt = preds_opt.astype(int)

    qwk = qwk3(labels, preds_opt)
    results = {
        'qwk': qwk,
        'threshold0' : best_thresholds[0],
        'threshold1' : best_thresholds[1],
        'threshold2' : best_thresholds[2],
        'threshold3' : best_thresholds[3],
        'threshold4' : best_thresholds[4],
    }
    return results

def get_metric(_args):
    if _args.metric == "accuracy":
        return compute_metrics_accuracy
    elif _args.metric == "qwk":
        if _args.use_regression == "True":
            if _args.optimize_threshold == "True":
                print("Using QWK regression metric with threshold optimization")
                return compute_metrics_qwk_for_regression_with_optuna
            print("Using QWK regression metric")
            return compute_metrics_qwk_for_regression
        else:
            print("Using QWK classification metric")
            return compute_metrics_qwk
    else:
        raise ValueError("Undefined metric")