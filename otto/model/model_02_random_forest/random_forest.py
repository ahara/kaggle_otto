import numpy as np
import os

from sklearn import ensemble, feature_extraction, preprocessing

from otto_utils import consts, utils


MODEL_NAME = 'model_02_random_forest'
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
clf = ensemble.ExtraTreesClassifier(n_jobs=4, n_estimators=2000, max_features=20, min_samples_split=3,
                                    bootstrap=False, verbose=3, random_state=23)

if MODE == 'cv':
    scores, predictions = utils.make_blender_cv(clf, train, labels, calibrate=False)
    print 'CV:', scores, 'Mean log loss:', np.mean(scores)
    utils.write_blender_data(consts.BLEND_PATH, MODEL_NAME + '.csv', predictions)
elif MODE == 'submission':
    clf.fit(train, labels)
    predictions = clf.predict_proba(test)
    utils.save_submission(consts.DATA_SAMPLE_SUBMISSION_PATH,
                          os.path.join(consts.ENSEMBLE_PATH, MODEL_NAME + '.csv'),
                          predictions)
elif MODE == 'holdout':
    score = utils.hold_out_evaluation(clf, train, labels, calibrate=False)
    print 'Log loss:', score
else:
    print 'Unknown mode'
