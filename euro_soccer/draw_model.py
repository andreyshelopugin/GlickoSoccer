import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from lightgbm import LGBMRegressor
import lightgbm as lgb
from optuna.integration import LightGBMTunerCV
from scipy.stats import skellam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from euro_soccer.draw_probability_train_creator import TrainCreator


class DrawLightGBM(object):

    def __init__(self, objective='poisson', metric='poisson', cv_metric='poisson-mean',
                 n_estimators=5000, learning_rate=0.04, num_leaves=31, feature_fraction=0.88,
                 lambda_l1=0.0, lambda_l2=0.0, bagging_fraction=1.0, bagging_freq=1, min_data_in_leaf=20,
                 early_stopping_round=70, cv=5, seed=7):
        #
        # def __init__(self, objective='poisson', metric='poisson-mean', n_estimators=1500, learning_rate=0.006,
        #              num_leaves=6, feature_fraction=0.224, min_data_in_leaf=71, early_stopping_round=50, cv=5, seed=7):
        self.objective = objective
        self.metric = metric
        self.cv_metric = cv_metric

        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.feature_fraction = feature_fraction
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.min_data_in_leaf = min_data_in_leaf
        self.early_stopping_round = early_stopping_round

        self.cv = cv
        self.seed = seed

        self.train_path = 'data/train.pkl'
        self.test_path = 'data/test.pkl'
        self.model_path = 'data/model.pkl'
        self.features = ['is_home',
                         'avg_scoring_10',
                         'avg_scoring_20',
                         'is_pandemic',
                         'mean_score_10',
                         'median_score_10',
                         'mean_score_20',
                         'mean_score_10_against',
                         'median_score_10_against',
                         'mean_score_20_against']
        self.target = 'score'
        self.categorical_features = []

    def _params(self) -> dict:
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

    def model(self):
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

    def cross_val_score(self, params=None):
        """"""
        if params is None:
            params = self._params()

        train = joblib.load(self.train_path)

        x = train.loc[:, self.features]
        y = train[self.target].to_numpy('int')

        train_set = lgb.Dataset(x, y)

        categorical_features = self.categorical_features

        return lgb.cv(
            params=params,
            train_set=train_set,
            categorical_feature=categorical_features,
            stratified=False,
            show_stdv=False,
            nfold=self.cv,
            seed=self.seed)[self.cv_metric][-1]

    def save_model(self):
        """
            Fit and save trained model.
        """
        train_set = joblib.load(self.train_path)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)

        light_gbm = self.model()

        models = []
        for train_index, test_index in kf.split(train_set):
            fold_train, fold_test = train_set.iloc[train_index], train_set.iloc[test_index]

            x_train, y_train = fold_train.loc[:, self.features], fold_train[self.target]
            x_test, y_test = fold_test.loc[:, self.features], fold_test[self.target]

            light_gbm.fit(x_train, y_train,
                          early_stopping_rounds=self.early_stopping_round,
                          categorical_feature=self.categorical_features,
                          eval_set=(x_test, y_test),
                          verbose=False)

            models.append(light_gbm)

        joblib.dump(models, self.model_path)

    def validation(self):
        """
            Check quality on validation set
        """

        validation = joblib.load(self.test_path)

        x = validation.loc[:, self.features]
        y = validation[self.target].to_numpy('int')

        light_gbm = joblib.load('models/lgbm_draw.pkl')

        validation['prediction'] = light_gbm.predict(x)

        print(mean_squared_error(validation['prediction'], y))
        print(mean_absolute_error(validation['prediction'], y))

        return validation

    def actual_predictions(self, results: pd.DataFrame):
        """"""
        TrainCreator().train_validation(results)

        self.save_model()

        new_data = TrainCreator().for_predictions(results)

        self.predict(new_data)

    def predict(self, new_data=None) -> pd.DataFrame:
        """"""

        if new_data is None:
            new_data = joblib.load(self.test_path)

        light_gbm_list = joblib.load(self.model_path)

        new_data['prediction'] = np.mean([model.predict(new_data.loc[:, self.features]) for model in light_gbm_list],
                                         axis=0, dtype=np.float64)

        home = (new_data
                .loc[(new_data['is_home'] == 1), ['index', 'prediction']]
                .dropna()
                .rename(columns={'prediction': 'home_prediction'}))

        away = (new_data
                .loc[(new_data['is_home'] == 0), ['index', 'prediction']]
                .dropna()
                .rename(columns={'opponent': 'team', 'prediction': 'away_prediction'}))

        home['index'] = home['index'].to_numpy('int')
        away['index'] = away['index'].to_numpy('int')

        predictions = home.merge(away, how='inner', on=['index'])

        predictions = (predictions
                       .sort_values(['index'])
                       .reset_index(drop=True))

        predictions['home_win'] = (predictions
                                   .loc[:, ['home_prediction', 'away_prediction']]
                                   .apply(lambda df: skellam.pmf([range(1, 30)], df[0], df[1]).sum(), axis=1))

        predictions['draw'] = (predictions
                               .loc[:, ['home_prediction', 'away_prediction']]
                               .apply(lambda df: skellam.pmf(0, df[0], df[1]).sum(), axis=1))

        predictions['away_win'] = (1 - predictions['home_win'] - predictions['draw'])

        joblib.dump(predictions, 'data/skellam_predictions.pkl')

        return predictions

    def shap_feature_importance(self, max_display=10):
        """"""

        train = joblib.load(self.train_path)

        x_train, y_train = train.loc[:, self.features], train[self.target].to_numpy('int')

        light_gbm = self.model()

        model = light_gbm.fit(x_train, y_train,
                              categorical_feature=self.categorical_features,
                              verbose=False)

        shap_values = shap.TreeExplainer(model).shap_values(x_train)

        return shap.summary_plot(shap_values, x_train, plot_type="bar", max_display=max_display, show=False)

    def feature_importance(self) -> pd.DataFrame:
        """
        """

        train = joblib.load(self.train_path)
        x_train, y_train = train.loc[:, self.features], train['goals'].to_numpy('int')

        light_gbm = self.model()
        light_gbm.fit(x_train, y_train,
                      categorical_feature=self.categorical_features,
                      verbose=False)

        feature_importance = (pd.DataFrame(zip(x_train.columns, light_gbm.feature_importances_),
                                           columns=['feature', 'gain'])
                              .sort_values(['gain'], ascending=False)
                              .reset_index(drop=True))

        feature_importance['gain'] = round(100 * feature_importance['gain'] / sum(feature_importance['gain']), 3)

        return feature_importance

    def optuna_stepwise_optimization(self, time_budget_minutes: int, is_regression=True):
        """Hyperparameter tuner for LightGBM.

            It optimizes the following hyperparameters in a stepwise manner:
            lambda_l1, lambda_l2, num_leaves, feature_fraction, bagging_fraction, bagging_freq and min_data_in_leaf.

            https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.lightgbm.LightGBMTuner.html
        """

        if is_regression:
            stratified = False
        else:
            stratified = True

        train = joblib.load(self.train_path)

        x = train.loc[:, self.features]
        y = train[self.target].to_numpy('int')

        train_set = lgb.Dataset(x, y)

        tuner = LightGBMTunerCV(params=self._params(),
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

    def optuna_optimization(self, n_trials: int) -> pd.DataFrame:
        """"""

        study = optuna.create_study(direction="minimize")

        def objective(trial):
            params = {
                'objective': trial.suggest_categorical('objective', [self.objective]),
                'n_estimators': trial.suggest_categorical('n_estimators', [5000]),
                'early_stopping_round': trial.suggest_categorical('early_stopping_round',
                                                                  [10, 20, 30, 40, 50, 70, 100]),
                'learning_rate': trial.suggest_uniform('learning_rate', 0.0001, 0.05),

                'feature_fraction': trial.suggest_categorical('feature_fraction', [self.feature_fraction]),
                'num_leaves': trial.suggest_categorical('num_leaves', [self.num_leaves]),
                'lambda_l1': trial.suggest_categorical('lambda_l1', [self.lambda_l1]),
                'lambda_l2': trial.suggest_categorical('lambda_l2', [self.lambda_l2]),
                'bagging_fraction': trial.suggest_categorical('bagging_fraction', [self.bagging_fraction]),
                'bagging_freq': trial.suggest_categorical('bagging_freq', [self.bagging_freq]),
                'min_data_in_leaf': trial.suggest_categorical('min_data_in_leaf', [self.min_data_in_leaf]),

                'verbose': -1
            }

            return self.cross_val_score(params)

        study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True, show_progress_bar=True)

        experiments = study.trials_dataframe()

        experiments = (experiments
                       .drop(columns=['datetime_start', 'datetime_complete', 'state'])
                       .sort_values(['value'], ascending=True))

        experiments.columns = [col.replace('params_', '') for col in experiments.columns]

        return experiments




