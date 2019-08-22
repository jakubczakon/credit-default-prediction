import os

import neptune
import pandas as pd

from utils import md5_hash, get_filepaths, encode_categoricals

RAW_DATA_DIRPATH = 'data/raw'
INTERIM_FEATURES_FILEPATH = 'data/interim/bureau_features.csv'
NROWS=None

def main():
    neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'), project_qualified_name=os.getenv('NEPTUNE_PROJECT'))

    bureau_raw_path = os.path.join(RAW_DATA_DIRPATH,'bureau.csv.zip')
    bureau_raw = pd.read_csv(bureau_raw_path, nrows=NROWS)

    with neptune.create_experiment(name='feature_extraction',
                                   tags=['interim',
                                         'bureau',
                                         'feature_extraction'],
                                   upload_source_files=get_filepaths()):

        bureau_features, numeric_cols = extract(bureau_raw)
        bureau_features.to_csv(INTERIM_FEATURES_FILEPATH, index=None)

        neptune.set_property('numeric_features', str(numeric_cols))
        neptune.set_property('features_version', md5_hash(INTERIM_FEATURES_FILEPATH))
        neptune.set_property('features_path', INTERIM_FEATURES_FILEPATH)


def extract(bureau):
    groupby_SK_ID_CURR = bureau.groupby(by=['SK_ID_CURR'])
    groupby_features = []
    features = pd.DataFrame({'SK_ID_CURR':bureau['SK_ID_CURR'].unique()})

    group_obj = groupby_SK_ID_CURR['DAYS_CREDIT'].agg('count').\
      reset_index().rename(index=str, columns={'DAYS_CREDIT': 'bureau_number_of_past_loans'})
    groupby_features.append(group_obj)

    group_obj = groupby_SK_ID_CURR['CREDIT_TYPE'].agg('nunique').reset_index().\
      rename(index=str, columns={'CREDIT_TYPE': 'bureau_number_of_loan_types'})
    groupby_features.append(group_obj)

    group_obj = groupby_SK_ID_CURR['AMT_CREDIT_SUM_DEBT'].agg('sum').reset_index().\
      rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'bureau_total_customer_debt'})
    groupby_features.append(group_obj)

    group_obj = groupby_SK_ID_CURR['AMT_CREDIT_SUM'].agg('sum').reset_index().\
      rename(index=str, columns={'AMT_CREDIT_SUM': 'bureau_total_customer_credit'})
    groupby_features.append(group_obj)

    group_obj = groupby_SK_ID_CURR['AMT_CREDIT_SUM_OVERDUE'].agg('sum').reset_index().\
       rename(index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'bureau_total_customer_overdue'})
    groupby_features.append(group_obj)

    for group_obj in groupby_features:
        features = features.merge(group_obj, on=['SK_ID_CURR'], how='left')

    numerical_cols = [col for col in features.columns if col!='SK_ID_CURR']

    return features, numerical_cols


if __name__ == '__main__':
    main()
