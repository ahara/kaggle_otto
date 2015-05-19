import numpy as np
import os

from scipy.optimize import fmin_cobyla
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

import consts
import utils


def blended(c, x):
    result = None
    for i in range(len(c)):
        result = result + c[i] * x[i] if result is not None else c[i] * x[i]
    result /= sum(c)
    return result


def error(p, x, y):
    preds = blended(p, x)
    err = log_loss(y, preds)
    return err


def constraint(p, *args):
    return min(p) - .0


def get_weights():
    # Read validation labels
    _, labels, _, _, _ = utils.load_data()
    skf = StratifiedKFold(labels, n_folds=5, random_state=23)
    test_index = None
    for _, test_idx in skf:
        test_index = np.append(test_index, test_idx) if test_index is not None else test_idx
    val_labels = labels[test_index]
    # Read predictions on validation set
    val_predictions = []
    prediction_files = utils.get_prediction_files()
    for preds_file in prediction_files:
        vp = np.genfromtxt(os.path.join(consts.BLEND_PATH, preds_file), delimiter=',')
        val_predictions.append(vp)
    # Minimize blending function
    p0 = [1.] * len(prediction_files)
    p = fmin_cobyla(error, p0, args=(val_predictions, val_labels), cons=[constraint], rhoend=1e-5)

    return p


if __name__ == '__main__':
    weights = get_weights()
    print weights
    print weights / np.sum(weights)