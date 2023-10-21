import optuna
import pandas as pd
from scipy.stats import skellam

from catboost_model import CatBoost, CatBoostRegressor, Pool
from config import Config


class OutcomesCatBoost(CatBoost):
    """The CatBoost model predicts the number of goals for each team.
    Afterward, we make the assumption that goals in football follow a Poisson distribution.
    This assumption enables us to utilize the probability mass function for a Skellam distribution."""

    def __init__(self, target='score'):
        super().__init__(target=target,
                         fold_count=3,
                         iterations=1000, learning_rate=0.05090175934770157, colsample_bylevel=0.9092111469100367,
                         depth=7, l2_leaf_reg=3.4802473839385946, bagging_temperature=1,
                         random_strength=4.356977387977597, od_wait=100, min_data_in_leaf=18)

        self.loss_function = 'Poisson'
        self.cv_metric = 'test-Poisson-mean'
        self.bootstrap_type = 'MVS'
        self.boosting_type = 'Plain'

        self.train_path = Config().outcomes_paths['train']
        self.validation_path = Config().outcomes_paths['validation']
        self.test_path = Config().outcomes_paths['test']

        self.predictions_path = Config().outcomes_paths['catboost_predictions']

        self.model_path = Config().outcomes_paths['catboost_model']

        self.features = ['is_home',
                         'is_pandemic',
                         'avg_scoring_5',
                         'avg_scoring_10',
                         'avg_scoring_20',
                         'avg_scoring_30',
                         'avg_scoring_5_against',
                         'avg_scoring_10_against',
                         'avg_scoring_20_against',
                         'avg_scoring_30_against',
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

    def optuna_optimization(self, n_trials: int) -> pd.DataFrame:
        """"""

        study = optuna.create_study(direction="minimize")

        train = pd.read_feather(self.train_path)

        cv_dataset = Pool(data=train.loc[:, self.features],
                          label=train[self.target],
                          cat_features=self.cat_features)

        def objective(trial):
            """boosting_type:
            Ordered — Usually provides better quality on small datasets (< 50k), but it may be slower than the Plain scheme.
            Plain — The classic gradient boosting scheme."""

            params = {
                'loss_function': self.loss_function,
                'od_type': self.od_type,
                'iterations': self.iterations,
                "boosting_type": self.boosting_type,
                'verbose': False,
                "allow_writing_files": False,

                'od_wait': self.od_wait,
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.08),

                "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 0.01, 5),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1),
                "depth": trial.suggest_int("depth", 6, 12),

                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
                'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 5, 50),
                "random_strength": trial.suggest_float("random_strength", 0.1, 5),

            }

            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
            elif params["bootstrap_type"] == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

            return self.optuna_cv_score(params, cv_dataset)

        study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True, show_progress_bar=True)

        experiments = study.trials_dataframe()

        experiments = (experiments
                       .drop(columns=['datetime_start', 'datetime_complete', 'state'])
                       .sort_values(['value'], ascending=True))

        experiments.columns = [col.replace('params_', '') for col in experiments.columns]

        return experiments

    def predict(self, new_data=None) -> pd.DataFrame:
        """Predicts the outcomes of the football matches."""

        if new_data is None:
            new_data = pd.read_feather(self.test_path)

        model = CatBoostRegressor()

        model.load_model(self.model_path)

        new_data['prediction'] = model.predict(new_data.loc[:, self.features])

        home = (new_data
                .loc[(new_data['is_home'] == True), ['match_id', 'prediction']]
                .dropna()
                .rename(columns={'prediction': 'home_goals'}))

        away = (new_data
                .loc[(new_data['is_home'] == False), ['match_id', 'prediction']]
                .dropna()
                .rename(columns={'opponent': 'team', 'prediction': 'away_goals'}))

        home['match_id'] = home['match_id'].to_numpy('int')
        away['match_id'] = away['match_id'].to_numpy('int')

        predictions = home.merge(away, how='inner', on='match_id')

        # use the assumption that goals are Poisson distributed
        predictions['home_win'] = (predictions
                                   .loc[:, ['home_goals', 'away_goals']]
                                   .apply(lambda df: skellam.pmf([range(1, 30)], df[0], df[1]).sum(), axis=1))

        predictions['draw'] = (predictions
                               .loc[:, ['home_goals', 'away_goals']]
                               .apply(lambda df: skellam.pmf(0, df[0], df[1]).sum(), axis=1))

        predictions['away_win'] = (1 - predictions['home_win'] - predictions['draw'])

        predictions.reset_index().to_feather(self.predictions_path)

        return predictions
