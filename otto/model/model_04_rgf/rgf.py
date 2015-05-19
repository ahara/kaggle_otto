import inspect
import numpy as np
import os
import subprocess
import shutil

from sklearn import feature_extraction
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss

from otto_utils import consts, utils


MODEL_NAME = 'model_04_rgf'
MODE = 'cv'  # cv|submission|holdout|tune


class RGF(BaseEstimator, ClassifierMixin):
    def __init__(self, verbose=True, random_state=None):
        self.n_classes_ = None
        self.rgf_path = None
        self.files_location_ = None
        self.files_location_data_ = None
        self.files_location_output_ = None
        self.models = None
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        self.n_classes_ = 9
        self.rgf_path = '/home/adam/Tools/rgf1.2/bin/rgf'
        self.files_location_ = '/home/adam/Projects/otto/model/model_04_rgf'
        self.files_location_data_ = os.path.join(self.files_location_, 'data')
        self.files_location_output_ = os.path.join(self.files_location_, 'output')
        self.models = dict()

        shutil.rmtree(self.files_location_output_)

        train_index = np.array(range(X.shape[0]))
        np.random.shuffle(train_index)
        x_train, y_train = X[train_index], y[train_index]

        self._train(x_train, y_train)

        return self

    def predict(self, X):
        preds = self.predict_proba(X)
        return np.argmax(preds, 1)

    def predict_proba(self, X):
        return self._predict(X)

    def score(self, X, y, sample_weight=None):
        return log_loss(y, self.predict_proba(X))

    def get_params(self, deep=True):
        return {'verbose': self.verbose, 'random_state': self.random_state}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.predict_proba(X)

    def transform(self, X):
        return self.predict_proba(X)

    # Private methods
    def write_into_files(self, prefix, x, y=None):
        if not os.path.exists(self.files_location_data_):
            os.makedirs(self.files_location_data_)
        # Write file with X
        data_location = os.path.join(self.files_location_data_, '%s.data.x' % prefix)
        np.savetxt(data_location, x, delimiter='\t', fmt='%.5f')

        paths = dict(x=data_location, y=[])

        if y is not None:
            for i in range(self.n_classes_):
                labels = map(lambda l: ['+1'] if i == l else ['-1'], y)
                labels_location = os.path.join(self.files_location_data_, '%s.data.y.%d' % (prefix, i))
                np.savetxt(labels_location, labels, delimiter='\t', fmt='%s')
                paths['y'].append(labels_location)

        return paths

    def get_params_string(self, train_x_fn=None, train_y_fn=None, test_x_fn=None, test_y_fn=None,
                          model_fn=None, model_fn_prefix=None, evaluation_fn=None, prediction_fn=None,
                          reg_L2=None, reg_sL2=None, algorithm=None, loss=None,
                          test_interval=None, max_tree=None, max_leaf_forest=None):
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        params_string = ''

        for arg in args:
            if values[arg] is not None and arg != 'self':
                params_string += '%s=%s,' % (arg, values[arg])

        return params_string

    def _train(self, x_train, y_train):
        prefix_train, prefix_model = 'train', 'model'
        cmd = self.rgf_path + ' train %s'

        if not os.path.exists(self.files_location_output_):
            os.makedirs(self.files_location_output_)

        # Write files in RGF format
        paths = dict()
        paths[prefix_train] = self.write_into_files(prefix_train, x_train, y_train)

        for i in range(self.n_classes_):
            # Train and test model
            params_string = self.get_params_string(train_x_fn=paths[prefix_train]['x'],
                                                   train_y_fn=paths[prefix_train]['y'][i],
                                                   model_fn_prefix=os.path.join(self.files_location_output_,
                                                                                '%s_class_%s' % (prefix_model, i)),
                                                   reg_L2=.01,  # Should be 0.001
                                                   loss='Log',  # Maybe LS will work better
                                                   test_interval=200,  # Should be 2000
                                                   max_tree=1200,  # Should be 1000
                                                   max_leaf_forest=6000  # Should be 10000
                                                   )
            print 'Running', cmd % params_string
            process = subprocess.Popen((cmd % params_string).split(), stdout=subprocess.PIPE)
            output = process.communicate()[0]
            if self.verbose:
                print output

            # Read list of generated models
            models = [m for m in os.listdir(self.files_location_output_) if ('%s_class_%s' % (prefix_model, i)) in m]
            models.sort()
            self.models[i] = models[-1]

    def _predict(self, x_test):
        prefix_test, prefix_preds = 'test', 'preds'
        cmd = self.rgf_path + ' predict %s'

        if not os.path.exists(self.files_location_output_):
            os.makedirs(self.files_location_output_)

        # Write files in RGF format
        paths = dict()
        paths[prefix_test] = self.write_into_files(prefix_test, x_test)

        all_predictions = []

        for i in range(self.n_classes_):
            # Make predictions and collect it
            preds_file = os.path.join(self.files_location_output_, '%s_class_%s' % (prefix_preds, i))

            params_string = self.get_params_string(test_x_fn=paths[prefix_test]['x'],
                                                   model_fn=os.path.join(self.files_location_output_,
                                                                         self.models[i]),
                                                   prediction_fn=preds_file
                                                   )
            print 'Running', cmd % params_string
            process = subprocess.Popen((cmd % params_string).split(), stdout=subprocess.PIPE)
            output = process.communicate()[0]
            if self.verbose:
                print output

            # Read generated predictions
            preds = np.loadtxt(preds_file)
            preds = 1. / (1. + np.exp(-preds))  # For Log, Expo

            all_predictions.append(preds)

        # Join all predictions
        all_predictions = np.array(all_predictions).T
        all_predictions /= np.sum(all_predictions, axis=1)[:, None]

        return all_predictions


if __name__ == '__main__':
    train, labels, test, _, _ = utils.load_data()

    # Preprocess data - transform counts to TFIDF features
    tfidf = feature_extraction.text.TfidfTransformer(smooth_idf=False)
    train = np.append(train, tfidf.fit_transform(train).toarray(), axis=1)
    test = np.append(test, tfidf.transform(test).toarray(), axis=1)

    clf = RGF(verbose=False, random_state=23)

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