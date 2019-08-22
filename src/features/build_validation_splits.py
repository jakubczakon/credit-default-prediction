import os

import neptune
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.utils import md5_hash, get_filepaths

RAW_DATA_DIRPATH = 'data/raw'
INTERIM_FEATURES_DIRPATH = 'data/processed'
TEST_SIZE=0.2
SEED=1234
NROWS=None

def main():
    neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'),
                 project_qualified_name=os.getenv('NEPTUNE_PROJECT'))

    application_table_path = os.path.join(RAW_DATA_DIRPATH,'application_train.csv.zip')
    application_table = pd.read_csv(application_table_path, nrows=NROWS)

    index_table = application_table[['SK_ID_CURR', 'TARGET']]

    with neptune.create_experiment(name='validation schema',
                                   tags=['processed', 'validation'],
                                   upload_source_files=get_filepaths()):

        train_idx, valid_idx = train_test_split(index_table, test_size=TEST_SIZE, random_state=SEED)
        train_idx_path = os.path.join(INTERIM_FEATURES_DIRPATH,'train_idx.csv')
        train_idx.to_csv(train_idx_path, index=None)
        neptune.send_artifact(train_idx_path)
        neptune.set_property('train_split_version', md5_hash(train_idx_path))

        valid_idx_path = os.path.join(INTERIM_FEATURES_DIRPATH,'valid_idx.csv')
        valid_idx.to_csv(valid_idx_path, index=None)
        neptune.send_artifact(valid_idx_path)
        neptune.set_property('valid_split_version', md5_hash(valid_idx_path))


if __name__ == '__main__':
    main()
