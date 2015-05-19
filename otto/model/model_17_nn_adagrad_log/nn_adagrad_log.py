"""
Mean log loss from 5-fold CV: 0.488150595136
"""
import copy
import itertools
import numpy as np
import lasagne
import math
import os
import theano
import theano.tensor as T
import time

from lasagne.layers import DenseLayer, DropoutLayer, InputLayer, get_all_params
from lasagne.nonlinearities import rectify, softmax
from lasagne.objectives import categorical_crossentropy, Objective
from lasagne.updates import adagrad

from sklearn.base import BaseEstimator
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.utils import check_random_state

from otto_utils import consts, utils


MODEL_NAME = 'model_17_nn_adagrad_log'
MODE = 'submission'  # cv|submission|holdout|tune


class NeuralNetwork(BaseEstimator):
    def __init__(self, n_hidden=20, max_epochs=150, batch_size=200,
                 lr=0.01, rho=0.9, dropout=0.5, valid_ratio=0.0,
                 use_valid=False, verbose=0, random_state=None):
        self.n_hidden = n_hidden
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.rho = rho
        self.dropout = dropout
        self.valid_ratio = valid_ratio
        self.use_valid = use_valid
        self.verbose = verbose
        self.random_state = random_state
        # State
        self.score_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.model = None

    def fit(self, data, targets, sample_weight=None):
        self.classes_, indices = np.unique(targets, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]

        random_state = check_random_state(self.random_state)

        # Shuffle data and eventually split on train and validation sets
        if self.valid_ratio > 0:
            strat_shuffled_split = StratifiedShuffleSplit(targets, test_size=self.valid_ratio,
                                                          n_iter=1, random_state=self.random_state)
            train_index, valid_index = [s for s in strat_shuffled_split][0]
            X_train, y_train = data[train_index], targets[train_index]
            X_valid, y_valid = data[valid_index], targets[valid_index]
        else:
            X_train, y_train = data, targets
            X_valid, y_valid = np.array([]), np.array([])

        if self.verbose > 5:
            print 'X_train: %s, y_train: %s' % (X_train.shape, y_train.shape)
            if self.use_valid:
                print 'X_valid: %s, y_valid: %s' % (X_valid.shape, y_valid.shape)

        # Prepare theano variables
        dataset = dict(
            X_train=theano.shared(lasagne.utils.floatX(X_train)),
            y_train=T.cast(theano.shared(y_train), 'int32'),
            X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
            y_valid=T.cast(theano.shared(y_valid), 'int32'),
            num_examples_train=X_train.shape[0],
            num_examples_valid=X_valid.shape[0],
            input_dim=X_train.shape[1],
            output_dim=self.n_classes_,
        )

        if self.verbose > 0:
            print "Building model and compiling functions..."
        output_layer = self.build_model(dataset['input_dim'])
        iter_funcs = self.create_iter_functions(dataset, output_layer)

        if self.verbose > 0:
            print "Starting training..."
        now = time.time()
        results = []
        try:
            for epoch in self.train(iter_funcs, dataset, output_layer):
                if self.verbose > 1:
                    print "Epoch {} of {} took {:.3f}s".format(
                        epoch['number'], self.max_epochs, time.time() - now)
                now = time.time()
                results.append([epoch['number'], epoch['train_loss'], epoch['valid_loss']])
                if self.verbose > 1:
                    print "  training loss:\t\t{:.6f}".format(epoch['train_loss'])
                    print "  validation loss:\t\t{:.6f}".format(epoch['valid_loss'])
                    print "  validation accuracy:\t\t{:.2f} %%".format(
                        epoch['valid_accuracy'] * 100)

                if epoch['number'] >= self.max_epochs:
                    break

            if self.verbose > 0:
                print 'Minimum validation error: %f (epoch %d)' % \
                      (epoch['best_val_error'], epoch['best_val_iter'])

        except KeyboardInterrupt:
            pass

        return self

    def predict(self, data):
        preds, _ = self.make_predictions(data)

        return preds

    def predict_proba(self, data):
        _, proba = self.make_predictions(data)

        return proba

    def score(self):
        return self.score_

    # Private methods
    def build_model(self, input_dim):
        l_in = InputLayer(shape=(self.batch_size, input_dim))

        l_hidden1 = DenseLayer(l_in, num_units=self.n_hidden, nonlinearity=rectify)
        l_hidden1_dropout = DropoutLayer(l_hidden1, p=self.dropout)

        l_hidden2 = DenseLayer(l_hidden1_dropout, num_units=self.n_hidden / 2, nonlinearity=rectify)
        l_hidden2_dropout = DropoutLayer(l_hidden2, p=self.dropout)

        l_hidden3 = DenseLayer(l_hidden2_dropout, num_units=self.n_hidden / 4, nonlinearity=rectify)
        l_hidden3_dropout = DropoutLayer(l_hidden3, p=self.dropout)

        l_out = DenseLayer(l_hidden3_dropout, num_units=self.n_classes_, nonlinearity=softmax)

        return l_out

    def create_iter_functions(self, dataset, output_layer, X_tensor_type=T.matrix):
        batch_index = T.iscalar('batch_index')
        X_batch = X_tensor_type('x')
        y_batch = T.ivector('y')

        batch_slice = slice(batch_index * self.batch_size, (batch_index + 1) * self.batch_size)

        objective = Objective(output_layer, loss_function=categorical_crossentropy)

        loss_train = objective.get_loss(X_batch, target=y_batch)
        loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)

        pred = T.argmax(output_layer.get_output(X_batch, deterministic=True), axis=1)
        proba = output_layer.get_output(X_batch, deterministic=True)
        accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

        all_params = get_all_params(output_layer)
        updates = adagrad(loss_train, all_params, self.lr, self.rho)

        iter_train = theano.function(
            [batch_index], loss_train,
            updates=updates,
            givens={
                X_batch: dataset['X_train'][batch_slice],
                y_batch: dataset['y_train'][batch_slice],
            },
            on_unused_input='ignore',
        )

        iter_valid = None
        if self.use_valid:
            iter_valid = theano.function(
                [batch_index], [loss_eval, accuracy, proba],
                givens={
                    X_batch: dataset['X_valid'][batch_slice],
                    y_batch: dataset['y_valid'][batch_slice],
                },
            )

        return dict(train=iter_train, valid=iter_valid)

    def create_test_function(self, dataset, output_layer, X_tensor_type=T.matrix):
        batch_index = T.iscalar('batch_index')
        X_batch = X_tensor_type('x')

        batch_slice = slice(batch_index * self.batch_size, (batch_index + 1) * self.batch_size)

        pred = T.argmax(output_layer.get_output(X_batch, deterministic=True), axis=1)
        proba = output_layer.get_output(X_batch, deterministic=True)

        iter_test = theano.function(
            [batch_index], [pred, proba],
            givens={
                X_batch: dataset['X_test'][batch_slice],
            },
        )

        return dict(test=iter_test)

    def train(self, iter_funcs, dataset, output_layer):
        num_batches_train = dataset['num_examples_train'] // self.batch_size
        num_batches_valid = int(math.ceil(dataset['num_examples_valid'] / float(self.batch_size)))

        best_val_err = 100
        best_val_iter = -1

        for epoch in itertools.count(1):
            batch_train_losses = []
            for b in range(num_batches_train):
                batch_train_loss = iter_funcs['train'](b)
                batch_train_losses.append(batch_train_loss)

            avg_train_loss = np.mean(batch_train_losses)

            batch_valid_losses = []
            batch_valid_accuracies = []
            batch_valid_probas = []

            if self.use_valid:
                for b in range(num_batches_valid):
                    batch_valid_loss, batch_valid_accuracy, batch_valid_proba = iter_funcs['valid'](b)
                    batch_valid_losses.append(batch_valid_loss)
                    batch_valid_accuracies.append(batch_valid_accuracy)
                    batch_valid_probas.append(batch_valid_proba)

            avg_valid_loss = np.mean(batch_valid_losses)
            avg_valid_accuracy = np.mean(batch_valid_accuracies)

            if (best_val_err > avg_valid_loss and self.use_valid) or\
                    (epoch == self.max_epochs and not self.use_valid):
                best_val_err = avg_valid_loss
                best_val_iter = epoch
                # Save model
                self.score_ = best_val_err
                self.model = copy.deepcopy(output_layer)


            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
                'valid_loss': avg_valid_loss,
                'valid_accuracy': avg_valid_accuracy,
                'best_val_error': best_val_err,
                'best_val_iter': best_val_iter,
            }

    def make_predictions(self, data):
        dataset = dict(
            X_test=theano.shared(lasagne.utils.floatX(data)),
            num_examples_test=data.shape[0],
            input_dim=data.shape[1],
            output_dim=self.n_classes_,
        )

        iter_funcs = self.create_test_function(dataset, self.model)
        num_batches_test = int(math.ceil(dataset['num_examples_test'] / float(self.batch_size)))

        test_preds, test_probas = np.array([]), None

        for b in range(num_batches_test):
            batch_test_pred, batch_test_proba = iter_funcs['test'](b)
            test_preds = np.append(test_preds, batch_test_pred)
            test_probas = np.append(test_probas, batch_test_proba, axis=0) if test_probas is not None else batch_test_proba

        return test_preds, test_probas


if __name__ == '__main__':
    train, labels, test, _, _ = utils.load_data()

    train = np.log(train + 1.)
    test = np.log(test + 1.)

    clf = NeuralNetwork(1024, 110, 220, 0.0026294067059507813, 1.1141900388281156e-15, 0.26355646219340834,
                        .02, True, 10, random_state=23)

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