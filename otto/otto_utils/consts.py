import os


PROJECT_PATH = '/home/adam/Projects/otto'

# Data
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
DATA_TRAIN_PATH = os.path.join(DATA_PATH, 'train.csv')
DATA_TEST_PATH = os.path.join(DATA_PATH, 'test.csv')
DATA_SAMPLE_SUBMISSION_PATH = os.path.join(DATA_PATH, 'sampleSubmission.csv')

# Models
MODEL_PATH = os.path.join(PROJECT_PATH, 'model')

# Results
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'output')

# Blending
BLEND_PATH = os.path.join(PROJECT_PATH, 'blend')

# Ensembling
ENSEMBLE_PATH = os.path.join(PROJECT_PATH, 'ensemble')

# Names of prediction files
PREDICTION_FILES = ['03_svm', '05_bagging_nn_rmsprop',
                    '06_xgboost', '09_nn_adagrad', '11_xgboost_poly',
                    '12_nn_rmsprop_pca', '13_nn_rmsprop_features',
                    '16_random_forest_calibrated_feature_selection']
