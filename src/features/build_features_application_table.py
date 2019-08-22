import os

import neptune
import numpy as np
import pandas as pd
import swifter

from src.features.utils import md5_hash, get_filepaths, encode_categoricals

RAW_DATA_DIRPATH = 'data/raw'
INTERIM_FEATURES_DIRPATH = 'data/interim/application_features.csv'
NROWS=None

def main():
    neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'),
                 project_qualified_name=os.getenv('NEPTUNE_PROJECT'))

    application_raw_path = os.path.join(RAW_DATA_DIRPATH,'application_train.csv.zip')
    application_raw = pd.read_csv(application_raw_path, nrows=NROWS)

    with neptune.create_experiment(name='feature_extraction',
                                   tags=['interim',
                                         'application',
                                         'feature_extraction'],
                                   upload_source_files=get_filepaths()):

        application_features, (numeric_cols, categorical_cols) = extract(application_raw)
        application_features.to_csv(INTERIM_FEATURES_DIRPATH, index=None)

        neptune.set_property('numeric_features', str(numeric_cols))
        neptune.set_property('categorical_features', str(categorical_cols))
        neptune.set_property('features_version', md5_hash(INTERIM_FEATURES_DIRPATH))
        neptune.set_property('features_path', INTERIM_FEATURES_DIRPATH)


def extract(X):
    categorical_cols= ['NAME_CONTRACT_TYPE',
                      'CODE_GENDER',
                      'NAME_EDUCATION_TYPE',
                      'NAME_FAMILY_STATUS',
                      'NAME_HOUSING_TYPE',
                      'FLAG_OWN_CAR',
                      'FLAG_OWN_REALTY',
                      'NAME_INCOME_TYPE',
                      'OCCUPATION_TYPE',
                      'ORGANIZATION_TYPE',
                     ]

    numerical_cols = ['AMT_REQ_CREDIT_BUREAU_WEEK',
                     'AMT_REQ_CREDIT_BUREAU_QRT',
                     'AMT_REQ_CREDIT_BUREAU_DAY',
                     'EXT_SOURCE_2',
                     'CNT_FAM_MEMBERS',
                     'AMT_CREDIT',
                     'CNT_CHILDREN',
                     'EXT_SOURCE_3',
                     'AMT_REQ_CREDIT_BUREAU_YEAR',
                     'DAYS_EMPLOYED',
                     'AMT_ANNUITY',
                     'EXT_SOURCE_1',
                     'AMT_REQ_CREDIT_BUREAU_HOUR',
                     'AMT_INCOME_TOTAL',
                     'DAYS_BIRTH']

    def clean_table(X, numerical_columns):
        X[numerical_columns] = X[numerical_columns].astype(float)
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)
        X['CODE_GENDER'] = X['CODE_GENDER'].replace('XNA', np.nan)
        X['ORGANIZATION_TYPE'] = X['ORGANIZATION_TYPE'].replace('XNA', np.nan)
        return X

    def split_organization_type(X):
        organization_cols = ['ORGANIZATION_TYPE_main','ORGANIZATION_TYPE_subtype']

        def _split_org(x):
            x = str(x).replace(':','')
            split_types = x.lower().split('type')
            if len(split_types) == 1:
                return pd.Series({'ORGANIZATION_TYPE_main':split_types[0],
                                  'ORGANIZATION_TYPE_subtype':np.nan})
            else:
                return pd.Series({'ORGANIZATION_TYPE_main':split_types[0],
                                  'ORGANIZATION_TYPE_subtype':split_types[1]})

        X[organization_cols] = X['ORGANIZATION_TYPE'].swifter.apply(_split_org)
        return X, organization_cols

    def hand_crafted_features(X):
        X['annuity_income_percentage'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['children_ratio'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS']
        X['credit_to_annuity_ratio'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
        X['credit_to_income_ratio'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
        X['days_employed_percentage'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
        X['income_credit_percentage'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
        X['income_per_child'] = X['AMT_INCOME_TOTAL'] / (1 + X['CNT_CHILDREN'])
        X['income_per_person'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS']
        X['payment_rate'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']

        hand_crafted_columns = ['annuity_income_percentage',
                               'children_ratio',
                               'credit_to_annuity_ratio',
                               'credit_to_income_ratio',
                               'days_employed_percentage',
                               'income_credit_percentage',
                               'income_per_child',
                               'income_per_person',
                               'payment_rate',
                              ]

        return X, hand_crafted_columns

    X = clean_table(X, numerical_cols)
    X, hand_crafted_columns = hand_crafted_features(X)
    numerical_cols = numerical_cols + hand_crafted_columns

    X, organization_cols = split_organization_type(X)
    categorical_cols = categorical_cols + organization_cols

    X = encode_categoricals(X, categorical_cols)

    X = X[['SK_ID_CURR'] + numerical_cols + categorical_cols]

    return X, (numerical_cols, categorical_cols)


if __name__ == '__main__':
    main()
