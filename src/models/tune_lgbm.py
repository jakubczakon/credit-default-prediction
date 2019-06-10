import os

import neptune
from neptunecontrib.monitoring.utils import axes2fig
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skopt
import skopt.plots
from sklearn.externals import joblib

from src.features.utils import md5_hash, get_filepaths
from train_lgbm import train_evaluate

TRAIN_IDX_PATH = 'data/processed/train_idx.csv'
VALID_IDX_PATH = 'data/processed/valid_idx.csv'
FEATURES_PATH = 'data/processed/features_joined_v1.csv'
REPORTS_DIRPATH = 'reports'
NROWS=None
SEED=1234

STATIC_PARAMS = {'metric':'auc',
                 'seed': SEED,
                 'num_threads': 2,
                }

HPO_PARAMS = {'n_calls':100,
              'n_random_starts':10,
              'base_estimator':'ET',
              'acq_func':'EI',
              'xi':0.01,
             }

SPACE = [skopt.space.Real(0.01, 0.5, name='learning_rate', prior='log-uniform'),
         skopt.space.Integer(1, 30, name='max_depth'),
         skopt.space.Integer(2, 100, name='num_leaves'),
         skopt.space.Integer(10, 1000, name='min_data_in_leaf'),
         skopt.space.Real(0.1, 1.0, name='feature_fraction', prior='uniform'),
         skopt.space.Real(0.1, 1.0, name='subsample', prior='uniform'),
         ]

def to_named_params(params):
    return([(dimension.name, param) for dimension, param in zip(SPACE, params)])

def monitor(res):
    neptune.send_metric('run_score', res.func_vals[-1])
    neptune.send_text('run_parameters', str(to_named_params(res.x_iters[-1])))
        
        
if __name__ == '__main__':
    
    neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'), project_qualified_name=os.getenv('NEPTUNE_PROJECT'))
    
    train_idx = pd.read_csv(TRAIN_IDX_PATH, nrows=NROWS)
    valid_idx = pd.read_csv(VALID_IDX_PATH, nrows=NROWS)
    features = pd.read_csv(FEATURES_PATH, nrows=NROWS)

    train = pd.merge(train_idx, features, on='SK_ID_CURR')
    valid = pd.merge(valid_idx, features, on='SK_ID_CURR')

    experiment_params = {**HPO_PARAMS,**STATIC_PARAMS}

    @skopt.utils.use_named_args(SPACE)
    def objective(**params):
        all_params = {**params, **STATIC_PARAMS}
        results = train_evaluate(train, valid, all_params)
        return -1.0 * results['valid_score']

    with neptune.create_experiment(name='model training',
                                   params=experiment_params,
                                   tags=['hpo', 'lgbm'],
                                   upload_source_files=get_filepaths(),
                                   properties={'features_path':FEATURES_PATH,
                                               'features_version':md5_hash(FEATURES_PATH),
                                               'train_split_version': md5_hash(TRAIN_IDX_PATH),  
                                               'valid_split_version': md5_hash(VALID_IDX_PATH)}):

        results = skopt.forest_minimize(objective, SPACE, callback=[monitor], **HPO_PARAMS)

        best_auc = -1.0 * results.fun
        best_params = results.x

        neptune.send_metric('valid_auc', best_auc)
        neptune.set_property('best_params', str(to_named_params(best_params)))

        # log results
        skopt.dump(results, os.path.join(REPORTS_DIRPATH, 'skopt_results.pkl'))
        neptune.send_artifact(os.path.join(REPORTS_DIRPATH, 'skopt_results.pkl'))

        # log diagnostic plots
        fig, ax = plt.subplots(figsize=(16,12))
        skopt.plots.plot_convergence(results, ax=ax)
        fig.savefig(os.path.join(REPORTS_DIRPATH, 'convergence.png'))
        neptune.send_image('diagnostics', os.path.join(REPORTS_DIRPATH, 'convergence.png'))

        axes = skopt.plots.plot_evaluations(results)
        fig = plt.figure(figsize=(16,12))
        fig = axes2fig(axes, fig)
        fig.savefig(os.path.join(REPORTS_DIRPATH, 'evaluations.png'))
        neptune.send_image('diagnostics', os.path.join(REPORTS_DIRPATH, 'evaluations.png'))

        axes = skopt.plots.plot_objective(results)   
        fig = plt.figure(figsize=(16,12))
        fig = axes2fig(axes, fig)
        fig.savefig(os.path.join(REPORTS_DIRPATH, 'objective.png'))
        neptune.send_image('diagnostics', os.path.join(REPORTS_DIRPATH, 'objective.png'))
