import joblib
import optuna
from scipy.stats import skellam
import numpy as np
import pandas as pd
from catboost_model import CatBoost


class OutcomesCatBoost(CatBoost):

    def __init__(self, target='score'):
        super().__init__(target=target, iterations=5000, learning_rate=0.002, colsample_bylevel=0.8,
                         depth=8, l2_leaf_reg=3, bagging_temperature=1,
                         random_strength=1, od_wait=20)

        self.features = [
                         'is_home',
                         'avg_scoring_5',
                         'avg_scoring_10',
                         'avg_scoring_20',
                         'avg_scoring_30',
                         'is_pandemic',
                         'mean_score_5',
                         'mean_score_10',
                         'mean_score_20',
                         'mean_score_30',
                         'median_score_5',
                         'median_score_10',
                         'median_score_20',
                         'median_score_30',
                         'mean_score_5_against',
                         'mean_score_10_against',
                         'mean_score_20_against',
                         'median_score_10_against',
                         'max_score_10']
        self.cat_features = []
        self.predictions_path = 'data/skellam_predictions.pkl'

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

        predictions['draw'] = (predictions
                               .loc[:, ['home_prediction', 'away_prediction']]
                               .apply(lambda df: skellam.pmf(0, df[0], df[1]).sum(), axis=1))

        joblib.dump(predictions, self.predictions_path)

        return predictions
