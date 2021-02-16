import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from scipy.stats import skellam
from sklearn.metrics import mean_squared_error, mean_absolute_error


class PoissonLightGBM(object):

    def __init__(self, objective='poisson', metric='poisson-mean', n_estimators=1500, learning_rate=0.006,
                 num_leaves=6, feature_fraction=0.224, min_data_in_leaf=71, cv=5, seed=7):
        self.objective = objective
        self.metric = metric

        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.feature_fraction = feature_fraction
        self.min_data_in_leaf = min_data_in_leaf

        self.cv = cv
        self.seed = seed

        self.train_path = 'data/train.pkl'
        self.test_path = 'data/test.pkl'
        self.features = ['is_home',
                         'team_id',
                         'opponent_id',
                         'avg_goals',
                         'avg_goals_against',
                         'avg_xg',
                         'avg_xg_against',
                         'opp_avg_goals',
                         'opp_avg_goals_against',
                         'opp_avg_xg',
                         'opp_avg_xg_against',
                         'avg_goals_difference',
                         'avg_xg_difference']
        self.categorical_features = ['team_id', 'opponent_id']
        self.seasons = [2017, 2018, 2019]

    def _params(self):
        """"""
        params = {
            'objective': self.objective,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'feature_fraction': self.feature_fraction,
            'min_data_in_leaf': self.min_data_in_leaf,
            'random_state': self.seed,
            'verbose': -1
        }

        return params

    def model(self):
        """"""
        light_gbm = LGBMRegressor(objective=self.objective,
                                  n_estimators=self.n_estimators,
                                  num_leaves=self.num_leaves,
                                  learning_rate=self.learning_rate,
                                  feature_fraction=self.feature_fraction,
                                  min_data_in_leaf=self.min_data_in_leaf,
                                  random_state=self.seed,
                                  verbose=-1)

        return light_gbm

    def cross_val_score(self):
        """"""
        train = joblib.load(self.train_path)
        test = joblib.load(self.test_path)

        light_gbm = self.model()

        scores = []
        for season in self.seasons:
            train_season = train.loc[train['season'] == season]
            test_season = test.loc[test['season'] == season]

            x_train, y_train = train_season.loc[:, self.features], train_season['goals'].to_numpy('int')
            x_test, y_test = test_season.loc[:, self.features], test_season['goals'].to_numpy('int')

            light_gbm.fit(x_train, y_train,
                          categorical_feature=self.categorical_features,
                          verbose=False)

            test_season['prediction'] = light_gbm.predict(x_test)

            scores.append(mean_squared_error(y_test, test_season['prediction']))

        return np.mean(scores)

    def save_model(self, season: int):
        """
            Fit and save trained model.
        """

        train_set = joblib.load(self.train_path)
        train_season = train_set.loc[train_set['season'] == season]

        x_train, y_train = train_season.loc[:, self.features], train_season['goals'].to_numpy('int')

        light_gbm = self.model()

        light_gbm.fit(x_train, y_train,
                      categorical_feature=self.categorical_features,
                      verbose=False)

        joblib.dump(light_gbm, 'models/lgbm' + str(season) + '.pkl')

    def validation(self, season: int):
        """
            Check quality on validation set
        """

        validation = joblib.load(self.test_path)

        validation = validation[validation['season'] == season]

        x = validation.loc[:, self.features]
        y = validation['goals'].to_numpy('int')

        light_gbm = joblib.load('models/lgbm' + str(season) + '.pkl')

        validation['prediction'] = light_gbm.predict(x)

        print(mean_squared_error(validation['prediction'], y))
        print(mean_absolute_error(validation['prediction'], y))

        return validation

    def predict(self, season: int) -> pd.DataFrame:
        """"""

        new_data = joblib.load(self.test_path)
        new_data = new_data.loc[new_data['season'] == season]

        light_gbm = joblib.load('models/lgbm' + str(season) + '.pkl')

        new_data['prediction'] = light_gbm.predict(new_data.loc[:, self.features])

        home = (new_data
                .loc[(new_data['is_home'] == 1), ['tournament', 'round', 'team', 'opponent', 'prediction']]
                .rename(columns={'prediction': 'home_prediction'}))

        away = (new_data
                .loc[(new_data['is_home'] == 0), ['round', 'opponent', 'prediction']]
                .rename(columns={'opponent': 'team', 'prediction': 'away_prediction'}))

        predictions = home.merge(away, how='inner', on=['round', 'team'])

        predictions = (predictions
                       .sort_values(['tournament', 'round'])
                       .rename(columns={'team': 'home_team', 'opponent': 'away_team'})
                       .reset_index(drop=True))

        predictions['home_win'] = (predictions
                                   .loc[:, ['home_prediction', 'away_prediction']]
                                   .apply(lambda df: skellam.pmf([range(1, 30)], df[0], df[1]).sum(), axis=1))

        predictions['draw'] = (predictions
                               .loc[:, ['home_prediction', 'away_prediction']]
                               .apply(lambda df: skellam.pmf(0, df[0], df[1]).sum(), axis=1))

        predictions['away_win'] = (1 - predictions['home_win'] - predictions['draw'])

        predictions['season'] = season

        return predictions

    def predict_all(self):
        """"""
        predictions_list = []
        for season in self.seasons:
            predictions = self.predict(season)
            predictions['season'] = season
            predictions_list.append(predictions)

        predictions = pd.concat(predictions_list)

        return predictions

    def shap_feature_importance(self, season: int, max_display=10):
        """"""

        train = joblib.load(self.train_path)

        train = train.loc[train['season'] == season]
        x_train, y_train = train.loc[:, self.features], train['goals'].to_numpy('int')

        light_gbm = self.model()

        model = light_gbm.fit(x_train, y_train,
                              categorical_feature=self.categorical_features,
                              verbose=False)

        shap_values = shap.TreeExplainer(model).shap_values(x_train)

        return shap.summary_plot(shap_values, x_train, plot_type="bar", max_display=max_display, show=False)

    def feature_importance(self, season: int) -> pd.DataFrame:
        """
        """

        train = joblib.load(self.train_path)
        train = train.loc[train['season'] == season]
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

    def optuna_optimization(self, n_trials: int) -> pd.DataFrame:
        """"""
        study = optuna.create_study(direction="minimize")

        train = joblib.load(self.train_path)
        test = joblib.load(self.test_path)

        useless_columns = ['round', 'xg', 'xg_against', 'goals_against', 'team', 'opponent', 'tournament']
        train = train.drop(columns=useless_columns)
        test = test.drop(columns=useless_columns)

        model = self.model()

        def objective(trial):
            params = {
                'objective': trial.suggest_categorical('objective', ['poisson']),
                'n_estimators': trial.suggest_int('n_estimators', 1500, 2000, step=1),
                'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.01),
                'num_leaves': trial.suggest_int('num_leaves', 4, 12, step=1),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.2, 0.3),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 90, step=1),
            }

            model.set_params(**params)

            scores = []
            for season in self.seasons:
                train_season = train.loc[train['season'] == season]
                test_season = test.loc[test['season'] == season]

                x_train, y_train = train_season.loc[:, self.features], train_season['goals'].to_numpy('int')
                x_test, y_test = test_season.loc[:, self.features], test_season['goals'].to_numpy('int')

                model.fit(x_train, y_train,
                          categorical_feature=self.categorical_features,
                          verbose=False)

                scores.append(mean_squared_error(y_test, model.predict(x_test)))

            return np.mean(scores)

        study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)

        experiments = study.trials_dataframe()

        experiments = (experiments
                       .drop(columns=['datetime_start', 'datetime_complete', 'state'])
                       .sort_values(['value'], ascending=True))

        experiments.columns = [col.replace('params_', '') for col in experiments.columns]

        return experiments
