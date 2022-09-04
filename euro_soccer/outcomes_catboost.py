import joblib
import optuna
import pandas as pd
from scipy.stats import skellam

from catboost_model import CatBoost, CatBoostRegressor
from config import Config


class OutcomesCatBoost(CatBoost):

    def __init__(self, target='score'):
        super().__init__(target=target, iterations=500, learning_rate=0.02, colsample_bylevel=0.8,
                         depth=8, l2_leaf_reg=3, bagging_temperature=1,
                         random_strength=1, od_wait=20)

        self.train_path = Config().project_path + Config().outcomes_paths['train']
        self.validation_path = Config().project_path + Config().outcomes_paths['validation']
        self.test_path = Config().project_path + Config().outcomes_paths['test']
        self.predictions_path = Config().project_path + Config().outcomes_paths['predictions']

        self.model_path = Config().project_path + Config().outcomes_paths['model']

        self.features = ['is_home',
                         'is_pandemic',
                         'avg_scoring_5',
                         'avg_scoring_10',
                         'avg_scoring_20',
                         'avg_scoring_30',
                         'tournament_type',
                         'tournament',
                         'league',
                         'opp_league',
                         'location_mean_score_5',
                         'location_mean_score_10',
                         'location_mean_score_20',
                         'location_mean_score_30',
                         'location_median_score_5',
                         'location_median_score_10',
                         'location_median_score_20',
                         'location_median_score_30',
                         'location_mean_score_5_against',
                         'location_mean_score_10_against',
                         'location_mean_score_20_against',
                         'location_median_score_10_against',
                         'location_max_score_10']

        self.cat_features = ['tournament_type', 'tournament', 'league', 'opp_league']

    def optuna_optimization(self, n_trials: int):
        """"""

        study = optuna.create_study(direction="minimize")

        train = joblib.load(self.train_path)
        x = train.loc[:, self.features]
        y = train[self.target]

        def objective(trial):
            params = {
                'loss_function': trial.suggest_categorical('loss_function', [self.loss_function]),
                'od_type': trial.suggest_categorical('iterations', [self.od_type]),
                'iterations': trial.suggest_categorical('iterations', [self.iterations]),
                'od_wait': trial.suggest_categorical('od_wait', [10, 20, 30, 40, 50, 70, 100]),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 0.05),

                "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-2, 1e0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1),
                "depth": trial.suggest_int("depth", 4, 10),
                "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
                'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 2, 20),

                'verbose': False
            }

            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
            elif params["bootstrap_type"] == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

            return self.regressor_cv_score(params, x, y, self.cat_features)

        study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True, show_progress_bar=True)

        experiments = study.trials_dataframe()

        experiments = (experiments
                       .drop(columns=['datetime_start', 'datetime_complete', 'state'])
                       .sort_values(['value'], ascending=True))

        experiments.columns = [col.replace('params_', '') for col in experiments.columns]

        return experiments

    def predict(self, new_data=None) -> pd.DataFrame:
        """"""

        if new_data is None:
            new_data = joblib.load(self.test_path)

        model = CatBoostRegressor()

        model.load_model(self.model_path)

        new_data['prediction'] = model.predict(new_data.loc[:, self.features])

        home = (new_data
                .loc[(new_data['is_home'] == 1), ['match_id', 'prediction']]
                .dropna()
                .rename(columns={'prediction': 'home_goals'}))

        away = (new_data
                .loc[(new_data['is_home'] == 0), ['match_id', 'prediction']]
                .dropna()
                .rename(columns={'opponent': 'team', 'prediction': 'away_goals'}))

        home['match_id'] = home['match_id'].to_numpy('int')
        away['match_id'] = away['match_id'].to_numpy('int')

        predictions = home.merge(away, how='inner', on=['match_id'])

        predictions = (predictions
                       .sort_values(['match_id'])
                       .reset_index(drop=True))

        predictions['home_win'] = (predictions
                                   .loc[:, ['home_goals', 'away_goals']]
                                   .apply(lambda df: skellam.pmf([range(1, 30)], df[0], df[1]).sum(), axis=1))

        predictions['draw'] = (predictions
                               .loc[:, ['home_goals', 'away_goals']]
                               .apply(lambda df: skellam.pmf(0, df[0], df[1]).sum(), axis=1))

        predictions['away_win'] = (1 - predictions['home_win'] - predictions['draw'])

        joblib.dump(predictions, self.predictions_path)

        return predictions
