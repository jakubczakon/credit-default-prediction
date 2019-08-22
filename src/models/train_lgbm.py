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

from src.features.utils import md5_hash, get_filepaths
from src.models.consts import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS
from src.serve import CreditDefaultClassifier

TRAIN_IDX_PATH = 'data/processed/train_idx.csv'
VALID_IDX_PATH = 'data/processed/valid_idx.csv'
FEATURES_PATH = 'data/processed/features_joined_v1.csv'
MODEL_DIRPATH = 'models/weights'
PRODUCTION_DIRPATH = 'models/production'
PREDICTION_DIRPATH = 'models/predictions'
REPORTS_DIRPATH = 'reports'
NROWS = None
PACKAGE_TO_PROD = True

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 100
SEED = 1234

LGBM_PARAMS = {'metric': 'auc',
               'seed': SEED,
               'num_threads': 2,
               'learning_rate': 0.014,
               'max_depth': 19,
               'num_leaves': 100,
               'min_data_in_leaf': 102,
               'feature_fraction': 0.27,
               'subsample': 0.912
               }


def main():
    neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'), project_qualified_name=os.getenv('NEPTUNE_PROJECT'))

    train_idx = pd.read_csv(TRAIN_IDX_PATH, nrows=NROWS)
    valid_idx = pd.read_csv(VALID_IDX_PATH, nrows=NROWS)
    features = pd.read_csv(FEATURES_PATH, nrows=NROWS)

    train = pd.merge(train_idx, features, on='SK_ID_CURR')
    valid = pd.merge(valid_idx, features, on='SK_ID_CURR')

    all_params = {'num_boost_round': NUM_BOOST_ROUND,
                  'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
                  **LGBM_PARAMS
                  }

    with neptune.create_experiment(name='model training',
                                   params=all_params,
                                   tags=['lgbm'],
                                   upload_source_files=get_filepaths(),
                                   properties={'features_path': FEATURES_PATH,
                                               'features_version': md5_hash(FEATURES_PATH),
                                               'train_split_version': md5_hash(TRAIN_IDX_PATH),
                                               'valid_split_version': md5_hash(VALID_IDX_PATH),
                                               }):
        results = train_evaluate(train, valid, LGBM_PARAMS, callbacks=[neptune_monitor()])
        train_score, valid_score = results['train_score'], results['valid_score']
        train_preds, valid_preds = results['train_preds'], results['valid_preds']

        neptune.send_metric('train_auc', train_score)
        neptune.send_metric('valid_auc', valid_score)

        train_pred_path = os.path.join(PREDICTION_DIRPATH, 'train_preds.csv')
        train_preds.to_csv(train_pred_path, index=None)
        neptune.send_artifact(train_pred_path)

        valid_pred_path = os.path.join(PREDICTION_DIRPATH, 'valid_preds.csv')
        valid_preds.to_csv(valid_pred_path, index=None)
        neptune.send_artifact(valid_pred_path)

        model_path = os.path.join(MODEL_DIRPATH, 'model.pkl')
        joblib.dump(results['model'], model_path)
        neptune.set_property('model_path', model_path)
        neptune.set_property('model_version', md5_hash(model_path))
        neptune.send_artifact(model_path)

        if PACKAGE_TO_PROD:
            saved_path = CreditDefaultClassifier.pack(model=results['model']).save(PRODUCTION_DIRPATH)
            neptune.set_property('production_model_path', saved_path)

        fig, ax = plt.subplots(figsize=(16, 12))
        sk_metrics.plot_confusion_matrix(valid_preds['TARGET'], valid_preds['preds_pos'] > 0.5, ax=ax)
        plot_path = os.path.join(REPORTS_DIRPATH, 'conf_matrix.png')
        fig.savefig(plot_path)
        neptune.send_image('diagnostics', plot_path)

        fig, ax = plt.subplots(figsize=(16, 12))
        sk_metrics.plot_roc(valid_preds['TARGET'], valid_preds[['preds_neg', 'preds_pos']], ax=ax)
        plot_path = os.path.join(REPORTS_DIRPATH, 'roc_auc.png')
        fig.savefig(plot_path)
        neptune.send_image('diagnostics', plot_path)

        fig, ax = plt.subplots(figsize=(16, 12))
        sk_metrics.plot_precision_recall(valid_preds['TARGET'], valid_preds[['preds_neg', 'preds_pos']], ax=ax)
        plot_path = os.path.join(REPORTS_DIRPATH, 'prec_recall.png')
        fig.savefig(plot_path)
        neptune.send_image('diagnostics', plot_path)

        fig, ax = plt.subplots(figsize=(16, 12))
        plot_prediction_distribution(valid_preds['TARGET'], valid_preds['preds_pos'], ax=ax)
        plot_path = os.path.join(REPORTS_DIRPATH, 'preds_dist.png')
        fig.savefig(plot_path)
        neptune.send_image('diagnostics', plot_path)


def train_evaluate(train, valid, params, callbacks=None):
    X_train = train[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS]
    y_train = train['TARGET']
    X_valid = valid[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS]
    y_valid = valid['TARGET']

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    model = lgb.train(params,
                      train_data,
                      feature_name=NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS,
                      categorical_feature=CATEGORICAL_COLUMNS,
                      num_boost_round=NUM_BOOST_ROUND,
                      valid_sets=[train_data, valid_data],
                      valid_names=['train_iter', 'valid_iter'],
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      callbacks=callbacks)

    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    train_score = roc_auc_score(y_train, y_train_pred)
    train_preds = train[['SK_ID_CURR', 'TARGET']]
    train_preds['preds_neg'] = 1.0 - y_train_pred
    train_preds['preds_pos'] = y_train_pred

    y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    valid_score = roc_auc_score(y_valid, y_valid_pred)
    valid_preds = valid[['SK_ID_CURR', 'TARGET']]
    valid_preds['preds_neg'] = 1.0 - y_valid_pred
    valid_preds['preds_pos'] = y_valid_pred

    return {'train_score': train_score,
            'valid_score': valid_score,
            'train_preds': train_preds,
            'valid_preds': valid_preds,
            'model': model}


if __name__ == '__main__':
    main()
