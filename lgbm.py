import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import Dataset, LGBMClassifier, LGBMRegressor, cv as cv_score
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, mean_absolute_error


class LightGBM(object):

    def __init__(self, target: str, objective, metric, cv_metric, is_regression=True,
                 n_estimators=5000, learning_rate=0.005, num_leaves=31, feature_fraction=1.0,
                 lambda_l1=0.0, lambda_l2=0.0, bagging_fraction=1.0, bagging_freq=1, min_data_in_leaf=20,
                 early_stopping_round=50, cv=5, seed=7):

        self.is_regression = is_regression

        self.train_path = ''
        self.validation_path = ''
        self.test_path = ''

        self.model_path = ''

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
        self.columns = []

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

    def classifier_cv_score(self, params: dict, train_set: Dataset, cat_features: list[str] = None) -> float:
        """"""
        if cat_features is None:
            cat_features = self.cat_features

        return cv_score(params=params,
                        train_set=train_set,
                        categorical_feature=cat_features,
                        stratified=True,
                        nfold=self.cv,
                        seed=self.seed)[self.cv_metric][-1]

    def regression_cv_score(self, params: dict, train_set: Dataset, cat_features: list[str] = None) -> float:
        """"""
        if cat_features is None:
            cat_features = self.cat_features

        return cv_score(params=params,
                        train_set=train_set,
                        categorical_feature=cat_features,
                        stratified=False,
                        nfold=self.cv,
                        seed=self.seed)[self.cv_metric][-1]

    def cross_val_score(self) -> float:
        """"""
        train = pd.read_feather(self.train_path, columns=self.columns)

        x, y = train.loc[:, self.features], train[self.target]

        train_set = lgb.Dataset(x, y, categorical_feature=self.cat_features)

        if self.is_regression:
            return self.regression_cv_score(self.params(), train_set)
        else:
            return self.classifier_cv_score(self.params(), train_set)

    def validation(self) -> pd.DataFrame:
        """
            Check quality on validation set
        """

        validation = pd.read_feather(self.test_path, columns=self.columns)

        x = validation.loc[:, self.features]
        y = validation[self.target]

        lgbm = joblib.load(self.model_path)

        if self.is_regression:
            validation['prediction'] = lgbm.predict(x)
            print('MSE: ', mean_squared_error(y, validation['prediction']))
            print('MAE: ', mean_absolute_error(y, validation['prediction']))
        else:
            validation['prediction'] = lgbm.predict_proba(x)[:, 1]
            print('LogLoss: ', log_loss(y, validation['prediction']))
            print('Accuracy: ', accuracy_score(y, validation['prediction'].round()))

        return validation

    def feature_importance(self) -> pd.DataFrame:
        """
        """

        train = pd.read_feather(self.train_path)
        x_train, y_train = train.loc[:, self.features], train[self.target]

        test = pd.read_feather(self.test_path)
        x_test, y_test = test.loc[:, self.features], test[self.target]

        if self.is_regression:
            light_gbm = self.regression()
        else:
            light_gbm = self.classifier()

        light_gbm.fit(x_train, y_train,
                      categorical_feature=self.cat_features,
                      eval_set=[(x_test, y_test)])

        feature_importance = (pd.DataFrame(zip(x_train.columns, light_gbm.feature_importances_),
                                           columns=['feature', 'gain'])
                              .sort_values(['gain'], ascending=False)
                              .reset_index(drop=True))

        feature_importance['gain'] = round(100 * feature_importance['gain'] / sum(feature_importance['gain']), 3)

        return feature_importance

    def save_model(self):
        """
            Fit and save trained model.
        """

        train_set = pd.read_feather(self.train_path, columns=self.columns)
        test_set = pd.read_feather(self.test_path, columns=self.columns)

        if self.is_regression:
            light_gbm = self.regression()
        else:
            light_gbm = self.classifier()

        x_train = train_set.loc[:, self.features]
        y_train = train_set[self.target]

        x_test = test_set.loc[:, self.features]
        y_test = test_set[self.target]

        light_gbm.fit(x_train, y_train,
                      categorical_feature=self.cat_features,
                      eval_set=[(x_test, y_test)])

        joblib.dump(light_gbm, self.model_path)
