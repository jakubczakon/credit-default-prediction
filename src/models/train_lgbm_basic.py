import os

import lightgbm as lgb
import neptune
from neptunecontrib.monitoring.lightgbm import neptune_monitor
from neptunecontrib.monitoring.reporting import plot_prediction_distribution
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
import scikitplot.metrics as sk_metrics
import matplotlib.pyplot as plt


TRAIN_IDX_PATH = 'data/processed/train_idx.csv'
VALID_IDX_PATH = 'data/processed/valid_idx.csv'
FEATURES_PATH = 'data/processed/features_joined_v1.csv'
NUM_BOOST_ROUND=1000
EARLY_STOPPING_ROUNDS=100

def train_evaluate(train, valid, params, callbacks=None):
    X_train = train[NUMERICAL_COLUMNS+CATEGORICAL_COLUMNS]
    y_train = train['TARGET']
    X_valid = valid[NUMERICAL_COLUMNS+CATEGORICAL_COLUMNS]
    y_valid = valid['TARGET']

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    model = lgb.train(params,
                          train_data,
                          feature_name=NUMERICAL_COLUMNS+CATEGORICAL_COLUMNS,
                          categorical_feature=CATEGORICAL_COLUMNS,
                          num_boost_round=NUM_BOOST_ROUND,
                          valid_sets = [train_data, valid_data],
                          valid_names=['train_iter', 'valid_iter'],
                          early_stopping_rounds = EARLY_STOPPING_ROUNDS,
                          callbacks=callbacks)

    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    train_score = roc_auc_score(y_train, y_train_pred)
    train_preds = train[['SK_ID_CURR','TARGET']]
    train_preds['preds_neg'] = 1.0 - y_train_pred
    train_preds['preds_pos'] = y_train_pred

    y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    valid_score = roc_auc_score(y_valid, y_valid_pred)
    valid_preds = valid[['SK_ID_CURR','TARGET']]
    valid_preds['preds_neg'] = 1.0 - y_valid_pred
    valid_preds['preds_pos'] = y_valid_pred

    return {'train_score':train_score,
            'valid_score':valid_score,
            'train_preds':train_preds,
            'valid_preds':valid_preds,
            'model':model}

from src.features.utils import md5_hash, get_filepaths
from consts import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS

import neptune

neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'), 
             project_qualified_name=os.getenv('NEPTUNE_PROJECT'))

LGBM_PARAMS = {'metric':'auc',
               'seed': SEED,
                'num_threads': 2,
               'learning_rate':0.014,
               'max_depth': 19,
               'num_leaves': 100}

with neptune.create_experiment(name='model training',
                               params=LGBM_PARAMS, # log hyperparameters
                               tags=['lgbm'], # organize with tags
                               upload_source_files=get_filepaths(), # log source code
                               ):
    
    # log data versions
    neptune.set_property('features_path', FEATURES_PATH)
    neptune.set_property('features_version', md5_hash(FEATURES_PATH))
        
    # run training and evaluation
    valid_score, valid_preds, model = train_evaluate(train, valid, LGBM_PARAMS, callbacks=[neptune_monitor()])

    # log metrics
    neptune.send_metric('valid_auc', valid_score)

    # log artifacts
    valid_preds.to_csv(valid_pred_path, index=None)
    neptune.send_artifact(valid_pred_path)
    
    # log model versions
    neptune.set_property('model_version', md5_hash(model_path))
    neptune.send_artifact(model_path)

    # log diagnostic charts and images
    sk_metrics.plot_roc(valid_preds['TARGET'], valid_preds[['preds_neg','preds_pos']], ax=ax)
    fig.savefig(plot_path)
    neptune.send_image('diagnostics', plot_path)