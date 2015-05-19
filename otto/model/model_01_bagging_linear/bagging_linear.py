"""
It achieves around 0.52914588084 log loss on holdout set
"""

import numpy as np
import os

from sklearn import ensemble, feature_extraction, linear_model, preprocessing
from sklearn.svm import LinearSVC

from otto_utils import consts, utils


MODEL_NAME = 'model_01_bagging_linear'
MODE = 'holdout'  # cv|submission|holdout

# import data
train, labels, test, _, _ = utils.load_data()

# polynomial features
poly_feat = preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
train = poly_feat.fit_transform(train, labels)
test = poly_feat.transform(test)

print train.shape

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer(smooth_idf=False)
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()

# feature selection
feat_selector = LinearSVC(C=0.3, penalty='l1', dual=False)
train = feat_selector.fit_transform(train, labels)
test = feat_selector.transform(test)

print train.shape

# encode labels
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

# train classifier
linear_clf = linear_model.LogisticRegression(C=1, penalty='l1',
                                             fit_intercept=True, random_state=23)

clf = ensemble.BaggingClassifier(base_estimator=linear_clf, n_estimators=40,
                                 max_samples=1., max_features=1., bootstrap=True,
                                 n_jobs=5, verbose=True, random_state=23)

if MODE == 'cv':
    scores, predictions = utils.make_blender_cv(clf, train, labels)
    print 'CV:', scores, 'Mean log loss:', np.mean(scores)
    utils.write_blender_data(consts.BLEND_PATH, MODEL_NAME + '.csv', predictions)
elif MODE == 'submission':
    clf.fit(train, labels)
    predictions = clf.predict_proba(test)
    utils.save_submission(consts.DATA_SAMPLE_SUBMISSION_PATH,
                          os.path.join(consts.ENSEMBLE_PATH, MODEL_NAME + '.csv'),
                          predictions)
elif MODE == 'holdout':
    score = utils.hold_out_evaluation(clf, train, labels)
    print 'Log loss:', score
else:
    print 'Unknown mode'


