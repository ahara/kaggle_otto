"""
5-fold CV - log loss 0.513778609795
"""
import numpy as np
import os

from hyperopt import fmin, hp, tpe

from sklearn import feature_extraction, preprocessing, svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier

from otto_utils import consts, utils


MODEL_NAME = 'model_03_svm'
MODE = 'cv'  # cv|submission|holdout|tune

# import data
train, labels, test, _, _ = utils.load_data()

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer(smooth_idf=False)
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()

# encode labels
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

# train classifier
clf = OneVsRestClassifier(svm.SVC(C=4.919646+2., kernel='rbf', tol=.001,
                                  verbose=True, probability=True, gamma=0.646508+.3, random_state=23))

if MODE == 'cv':
    scores, predictions = utils.make_blender_cv(clf, train, labels, calibrate=True)
    print 'CV:', scores, 'Mean log loss:', np.mean(scores)
    utils.write_blender_data(consts.BLEND_PATH, MODEL_NAME + '.csv', predictions)
elif MODE == 'submission':
    calibrated_classifier = CalibratedClassifierCV(clf, method='isotonic', cv=utils.get_cv(labels))
    fitted_classifier = calibrated_classifier.fit(train, labels)
    predictions = fitted_classifier.predict_proba(test)
    utils.save_submission(consts.DATA_SAMPLE_SUBMISSION_PATH,
                          os.path.join(consts.ENSEMBLE_PATH, MODEL_NAME + '.csv'),
                          predictions)
elif MODE == 'holdout':
    score = utils.hold_out_evaluation(clf, train, labels, calibrate=False, test_size=0.9)
    print 'Log loss:', score
elif MODE == 'tune':
    train, labels, valid, valid_labels = utils.stratified_split(train, labels, test_size=.8)
    from sklearn.metrics import log_loss
    # Objective function
    def objective(args):
        c, gamma = args
        clf = OneVsRestClassifier(svm.SVC(C=c, kernel='rbf', tol=.001, gamma=gamma,
                                  probability=True, random_state=23))
        score1 = 0
        score2 = utils.hold_out_evaluation(clf, train, labels, calibrate=False)
        score = log_loss(valid_labels, clf.predict_proba(valid))
        print 'C=%f, gamma=%f, score1=%f, score2=%f, score=%f' % (c, gamma, score1, score2, score)
        return score
    # Searching space
    space = (
        hp.uniform('c', 4, 10),
        hp.uniform('gamma', 0.3, 3)
    )

    best_sln = fmin(objective, space, algo=tpe.suggest, max_evals=200)
    print 'Best solution:', best_sln
else:
    print 'Unknown mode'
