"""
Code shared between all models for Otto competition.
"""

import gc
import numpy as np
import os
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import log_loss

import consts


def load_data(path_train=consts.DATA_TRAIN_PATH, path_test=consts.DATA_TEST_PATH):
    train = pd.read_csv(path_train)
    train_labels = [int(v[-1])-1 for v in train.target.values]
    train_ids = train.id.values
    train = train.drop('id', axis=1)
    train = train.drop('target', axis=1)

    test = pd.read_csv(path_test)
    test_ids = test.id.values
    test = test.drop('id', axis=1)

    return np.array(train, dtype=float), np.array(train_labels), np.array(test, dtype=float),\
        np.array(train_ids), np.array(test_ids)


def make_blender_cv(classifier, x, y, calibrate=False):
    skf = StratifiedKFold(y, n_folds=5, random_state=23)
    scores, predictions = [], None
    for train_index, test_index in skf:
        if calibrate:
            # Make training and calibration
            calibrated_classifier = CalibratedClassifierCV(classifier, method='isotonic', cv=get_cv(y[train_index]))
            fitted_classifier = calibrated_classifier.fit(x[train_index, :], y[train_index])
        else:
            fitted_classifier = classifier.fit(x[train_index, :], y[train_index])
        preds = fitted_classifier.predict_proba(x[test_index, :])

        # Free memory
        calibrated_classifier, fitted_classifier = None, None
        gc.collect()

        scores.append(log_loss(y[test_index], preds))
        predictions = np.append(predictions, preds, axis=0) if predictions is not None else preds
    return scores, predictions


def write_blender_data(path, file_name, predictions):
    file_path = os.path.join(path, file_name)
    np.savetxt(file_path, predictions, delimiter=',', fmt='%.5f')


def save_submission(path_sample_submission, output_file_path, predictions):
    sample = pd.read_csv(path_sample_submission)
    submission = pd.DataFrame(predictions, index=sample.id.values, columns=sample.columns[1:])
    submission.to_csv(output_file_path, index_label='id')


def stratified_split(x, y, test_size=0.2):
    strat_shuffled_split = StratifiedShuffleSplit(y, n_iter=1, test_size=test_size, random_state=23)
    train_index, valid_index = [s for s in strat_shuffled_split][0]

    x_train, y_train, x_valid, y_valid = x[train_index, :], y[train_index], x[valid_index, :], y[valid_index]

    return x_train, y_train, x_valid, y_valid


def hold_out_evaluation(classifier, x, y, test_size=0.2, calibrate=False):
    x_train, y_train, x_valid, y_valid = stratified_split(x, y, test_size)

    # Train
    if calibrate:
        # Make training and calibration
        calibrated_classifier = CalibratedClassifierCV(classifier, method='isotonic', cv=get_cv(y_train))
        fitted_classifier = calibrated_classifier.fit(x_train, y_train)
    else:
        fitted_classifier = classifier.fit(x_train, y_train)
    # Evaluate
    score = log_loss(y_valid, fitted_classifier.predict_proba(x_valid))

    return score


def get_prediction_files():
    return ['model_%s.csv' % f for f in consts.PREDICTION_FILES]


def get_cv(y, n_folds=5):
    return StratifiedKFold(y, n_folds=n_folds, random_state=23)
