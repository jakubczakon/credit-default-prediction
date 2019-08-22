import os

import neptune
import pandas as pd

from utils import md5_hash, get_filepaths

APPLICATION_FEATURES_PATH = 'data/interim/application_features.csv'
BUREAU_FEATURES_PATH = 'data/interim/bureau_features.csv'
PROCESSED_FEATURES_FILEPATH = 'data/processed/features_joined_v1.csv'
NROWS=None


def main():
    neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'), project_qualified_name=os.getenv('NEPTUNE_PROJECT'))

    interim_feature_paths = [APPLICATION_FEATURES_PATH, BUREAU_FEATURES_PATH]

    with neptune.create_experiment(name='feature_extraction',
                                   tags=['processed', 'feature_extraction','joined_features'],
                                   upload_source_files=get_filepaths()):

        features = pd.read_csv(interim_feature_paths[0],usecols=['SK_ID_CURR'], nrows=NROWS)
        for path in interim_feature_paths:
            df = pd.read_csv(path, nrows=NROWS)
            features = features.merge(df, on='SK_ID_CURR')

        features.to_csv(PROCESSED_FEATURES_FILEPATH, index=None)
        neptune.set_property('features_version', md5_hash(PROCESSED_FEATURES_FILEPATH))
        neptune.set_property('features_path', PROCESSED_FEATURES_FILEPATH)

if __name__ == '__main__':
    main()
