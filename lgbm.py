from typing import List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, Dataset, cv as cv_score
from optuna.integration.lightgbm import LightGBMTunerCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from config import Config
from utils.params import task_to_id_event_type


class LightGBM(object):

    def __init__(self, target: str, objective, metric, cv_metric, league='nba',
                 n_estimators=5000, learning_rate=0.005, num_leaves=31, feature_fraction=1.0,
                 lambda_l1=0.0, lambda_l2=0.0, bagging_fraction=1.0, bagging_freq=1, min_data_in_leaf=20,
                 early_stopping_round=50, cv=5, seed=7):

        self.train_path = Config().project_path + 'data/teams_train_' + target + '_' + league + '.pkl'
        self.test_path = Config().project_path + 'data/teams_test_' + target + '_' + league + '.pkl'
        self.model_path = Config().project_path + 'teams/saved_models/lgbm_' + target + '_' + league + '.pkl'

        self.target = target

        self.objective = objective
        self.metric = metric
        self.cv_metric = cv_metric

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_round = early_stopping_round
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.min_data_in_leaf = min_data_in_leaf

        self.cv = cv
        self.seed = seed

        self.features = []
        self.cat_features = []

    def params(self) -> dict:
        """"""

        params = {
            'objective': self.objective,
            'metric': self.metric,

            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'early_stopping_round': self.early_stopping_round,
            'random_state': self.seed,
            'verbose': -1,

            'feature_fraction': self.feature_fraction,
            'num_leaves': self.num_leaves,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'min_data_in_leaf': self.min_data_in_leaf,
        }

        return params

    def classifier(self):
        """"""
        return LGBMClassifier(objective=self.objective,
                              n_estimators=self.n_estimators,
                              learning_rate=self.learning_rate,
                              early_stopping_round=self.early_stopping_round,
                              feature_fraction=self.feature_fraction,
                              num_leaves=self.num_leaves,
                              lambda_l1=self.lambda_l1,
                              lambda_l2=self.lambda_l2,
                              bagging_fraction=self.bagging_fraction,
                              bagging_freq=self.bagging_freq,
                              min_data_in_leaf=self.min_data_in_leaf,
                              random_state=self.seed)

    def regression(self):
        """"""
        return LGBMRegressor(objective=self.objective,
                             n_estimators=self.n_estimators,
                             learning_rate=self.learning_rate,
                             early_stopping_round=self.early_stopping_round,
                             feature_fraction=self.feature_fraction,
                             num_leaves=self.num_leaves,
                             lambda_l1=self.lambda_l1,
                             lambda_l2=self.lambda_l2,
                             bagging_fraction=self.bagging_fraction,
                             bagging_freq=self.bagging_freq,
                             min_data_in_leaf=self.min_data_in_leaf,
                             random_state=self.seed)

    def classifier_cv_score(self, params: dict, train_set: Dataset, cat_features: List[str] = None) -> float:
        """"""
        if cat_features is None:
            cat_features = self.cat_features

        return cv_score(params=params,
                        train_set=train_set,
                        categorical_feature=cat_features,
                        stratified=True,
                        show_stdv=False,
                        nfold=self.cv,
                        seed=self.seed)[self.cv_metric][-1]

    def regression_cv_score(self, params: dict, train_set: Dataset, cat_features: List[str] = None) -> float:
        """"""
        if cat_features is None:
            cat_features = self.cat_features

        return cv_score(params=params,
                        train_set=train_set,
                        categorical_feature=cat_features,
                        stratified=False,
                        show_stdv=False,
                        nfold=self.cv,
                        seed=self.seed)[self.cv_metric][-1]

    def cross_val_score(self):
        """"""
        train = joblib.load(self.train_path)

        x, y = train.loc[:, self.features], train[self.target]

        train_set = lgb.Dataset(x, y, categorical_feature=self.cat_features)

        return self.regression_cv_score(self.params(), train_set)

    def validation(self):
        """
            Check quality on validation set
        """

        validation = joblib.load(self.test_path)

        x = validation.loc[:, self.features]
        y = validation[self.target]

        lgbm_list = joblib.load(self.model_path)

        validation['prediction'] = np.mean([lgbm.predict(x) for lgbm in lgbm_list], axis=0, dtype=np.float64)

        print('MSE: ', mean_squared_error(y, validation['prediction']))

        return validation

    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
            Load train model and make predictions on new data (or test).
        """

        models_list = joblib.load(self.model_path)

        data['value'] = np.mean([model.predict(data.loc[:, self.features]) for model in models_list], axis=0, dtype=np.float64)

        data['id_event_type'] = task_to_id_event_type(self.target)
        data['id_player'] = 27

        data = (data
                .loc[:, ['id_match', 'ishome', 'id_event_type', 'id_player', 'value']]
                .sort_values(['id_match', 'ishome']))

        return data

    def optuna_stepwise_optimization(self, train_set: Dataset, time_budget_minutes: int, is_regression: bool):
        """Hyperparameter tuner for LightGBM.

            It optimizes the following hyperparameters in a stepwise manner:
            lambda_l1, lambda_l2, num_leaves, feature_fraction, bagging_fraction, bagging_freq and min_data_in_leaf.

            https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.lightgbm.LightGBMTuner.html
        """

        if is_regression:
            stratified = False
        else:
            stratified = True

        tuner = LightGBMTunerCV(params=self.params(),
                                train_set=train_set,
                                early_stopping_rounds=self.early_stopping_round,
                                stratified=stratified,
                                nfold=self.cv,
                                seed=self.seed,
                                show_stdv=False,
                                verbose_eval=False,
                                time_budget=time_budget_minutes * 60,
                                show_progress_bar=True)

        tuner.run()

        print("Best score:", tuner.best_score)
        best_params = tuner.best_params
        print("Best params:", best_params)

        tuned_features = ['lambda_l1', 'lambda_l2', 'num_leaves', 'feature_fraction',
                          'bagging_fraction', 'bagging_freq', 'min_data_in_leaf']

        print("  Tuned Features: ")
        for key, value in best_params.items():
            if key in tuned_features:
                print("    {}: {}".format(key, round(value, 8)))

    def feature_importance(self, is_regression: bool) -> pd.DataFrame:
        """
        """

        train = joblib.load(self.train_path)
        x_train, y_train = train.loc[:, self.features], train[self.target]

        test = joblib.load(self.test_path)
        x_test, y_test = test.loc[:, self.features], test[self.target]

        if is_regression:
            light_gbm = self.regression()
        else:
            light_gbm = self.classifier()

        light_gbm.fit(x_train, y_train,
                      categorical_feature=self.cat_features,
                      early_stopping_rounds=self.early_stopping_round,
                      eval_set=(x_test, y_test),
                      verbose=False)

        feature_importance = (pd.DataFrame(zip(x_train.columns, light_gbm.feature_importances_),
                                           columns=['feature', 'gain'])
                              .sort_values(['gain'], ascending=False)
                              .reset_index(drop=True))

        feature_importance['gain'] = round(100 * feature_importance['gain'] / sum(feature_importance['gain']), 3)

        return feature_importance

    def save_model(self, is_regression: bool, is_with_validation=True):
        """
            Fit and save trained model.
        """

        train_set = joblib.load(self.train_path)

        if is_regression:
            light_gbm = self.regression()
        else:
            light_gbm = self.classifier()

        if is_with_validation:
            validation = joblib.load(self.test_path)
            train_set = pd.concat([train_set, validation])

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)

        models = []
        for train_index, test_index in kf.split(train_set):
            fold_train, fold_test = train_set.iloc[train_index], train_set.iloc[test_index]

            x_train, y_train = fold_train.loc[:, self.features], fold_train[self.target]
            x_test, y_test = fold_test.loc[:, self.features], fold_test[self.target]

            light_gbm.fit(x_train, y_train,
                          early_stopping_rounds=self.early_stopping_round,
                          categorical_feature=self.cat_features,
                          eval_set=(x_test, y_test),
                          verbose=False)

            models.append(light_gbm)
        joblib.dump(models, self.model_path)

    def cv_stepwise_optimization(self, time_budget_minutes: int):
        """"""
        train = joblib.load(self.train_path)
        x = train.loc[:, self.features]
        y = train[self.target]

        train_set = lgb.Dataset(x, y, categorical_feature=self.cat_features, free_raw_data=False, params={'verbose': -1})

        self.optuna_stepwise_optimization(train_set, time_budget_minutes, is_regression=True)

    def feature_selector(self, n_iterations: int, fixed_features: List[str], n_fixed_features_iterations: int,
                         is_regression: bool):
        """
        :param n_iterations: number of features for removing, some of them can be returned
        :param fixed_features: list of features are ignored for removing, while current iteration < n_fixed_features_iterations
        :param n_fixed_features_iterations:
        :param is_regression:
        :return:
        """
        train = joblib.load(self.train_path)

        x, y = train.loc[:, self.features], train[self.target]

        train_set = Dataset(x, y, categorical_feature=self.cat_features)

        params = self.params()

        if is_regression:
            init_loss = self.regression_cv_score(params, train_set)
        else:
            init_loss = self.classifier_cv_score(params, train_set)

        features = self.features
        cat_features = self.cat_features
        weakest_features = []

        results = [{'best_features': self.features, 'weakest_features': weakest_features, 'loss': init_loss}]

        # remove features
        for i in range(1, n_iterations + 1):

            if i < n_fixed_features_iterations:
                features_for_remove = [c for c in features if c not in fixed_features]
            else:
                features_for_remove = features

            losses = dict()
            for feature in features_for_remove:

                features_sample = [f for f in features if f != feature]
                cat_features_sample = [f for f in cat_features if f != feature]

                train_set = Dataset(x.loc[:, features_sample], y, categorical_feature=cat_features_sample)

                if is_regression:
                    losses[feature] = self.regression_cv_score(params, train_set, cat_features_sample)
                else:
                    losses[feature] = self.classifier_cv_score(params, train_set, cat_features_sample)

            weakest_feature = min(losses, key=losses.get)
            best_loss = losses[weakest_feature]

            weakest_features.append(weakest_feature)
            features = [f for f in features if f != weakest_feature]
            cat_features = [f for f in cat_features if f != weakest_feature]

            results.append({'best_features': features, 'weakest_features': weakest_features, 'loss': best_loss})

            print('weakest_features:', weakest_features, 'loss:', best_loss)

        # return features
        for i in range(len(weakest_features)):

            losses = dict()
            for feature in weakest_features:

                features_sample = features + [feature]

                if feature in self.cat_features:
                    cat_features_sample = cat_features + [feature]
                else:
                    cat_features_sample = cat_features

                train_set = Dataset(x.loc[:, features_sample], y, categorical_feature=cat_features_sample)

                if is_regression:
                    losses[feature] = self.regression_cv_score(params, train_set, cat_features_sample)
                else:
                    losses[feature] = self.classifier_cv_score(params, train_set, cat_features_sample)

            best_feature = min(losses, key=losses.get)
            best_loss = losses[best_feature]

            weakest_features = [f for f in weakest_features if f != best_feature]
            features.append(best_feature)

            if best_feature in self.cat_features:
                cat_features.append(best_feature)

            results.append({'best_features': features, 'weakest_features': weakest_features, 'loss': best_loss})

        results = pd.DataFrame(results)

        results = results.sort_values(['loss'])

        return results
