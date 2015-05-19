"""
5-fold CV - log loss 0.4687
"""
import numpy as np
import os

from sklearn import ensemble, feature_extraction, preprocessing
from sklearn.calibration import CalibratedClassifierCV

from otto_utils import consts, utils


MODEL_NAME = 'model_08_random_forest_calibrated'
MODE = 'holdout'  # cv|submission|holdout

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
clf = ensemble.ExtraTreesClassifier(n_jobs=5, n_estimators=600, max_features=20, min_samples_split=3,
                                    bootstrap=False, verbose=3, random_state=23)

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
    score = utils.hold_out_evaluation(clf, train, labels, calibrate=True)
    print 'Log loss:', score
else:
    print 'Unknown mode'
