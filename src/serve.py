import bentoml
from bentoml.artifact import PickleArtifact
from bentoml.handlers import JsonHandler
import pandas as pd

import src.features.build_features_application_table as application
import src.features.build_features_bureau_table as bureau


def preprocess_data(application_df, bureau_df):
    application_df, _ = application.extract(application_df)
    bureau_df, _ = bureau.extract(bureau_df)
    features = pd.merge(application_df, bureau_df, on=['SK_ID_CURR'])
    return features


@bentoml.artifacts([PickleArtifact('model')])
class CreditDefaultClassifier(bentoml.BentoService):

    @bentoml.api(JsonHandler)
    def predict(self, json):
        print('loading data...')
        application_df = pd.read_csv(json['application'])
        bureau_df = pd.read_csv(json['bureau'])

        print('preprocessing data...')
        features = preprocess_data(application_df, bureau_df)

        print('predicting on data...')
        return self.artifacts.model.predict(features)
