import joblib
import lightgbm as lgb
import optuna
import pandas as pd
from scipy.stats import skellam

from config import Config
from lgbm import LightGBM


class OutcomesLGBM(LightGBM):
    """The Light GBM model predicts the number of goals for each team.
    Afterward, we make the assumption that goals in football follow a Poisson distribution.
    This assumption enables us to utilize the probability mass function for a Skellam distribution."""

    def __init__(self):
        super().__init__(target='score', objective='poisson', metric='poisson', cv_metric='valid poisson-mean',
                         n_estimators=5000, learning_rate=0.011217,
                         num_leaves=20, feature_fraction=0.45, lambda_l1=1.26258, lambda_l2=9.50238,
                         bagging_fraction=0.844, bagging_freq=8, min_data_in_leaf=20, early_stopping_round=100,
                         cv=5)

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

        # potential data leak here: there's a wrong assumption that the strengths of the leagues remain stable over time.
        # on the one hand, we want to take into account information about opponents' leagues.
        # on the other hand, we're looking into the future when using this information.
        self.cat_features = ['tournament_type', 'tournament', 'league', 'opp_league']
        self.columns = self.features + [self.target]

        self.train_path = Config().outcomes_paths['train']
        self.validation_path = Config().outcomes_paths['validation']
        self.test_path = Config().outcomes_paths['test']

        self.predictions_path = Config().outcomes_paths['lgbm_predictions']

        self.model_path = Config().outcomes_paths['lgbm_model']

    def optuna_optimization(self, n_trials: int) -> pd.DataFrame:
        """"""

        study = optuna.create_study(direction="minimize")

        train = pd.read_feather(self.train_path, columns=self.columns)
        x = train.loc[:, self.features]
        y = train[self.target]

        train_set = lgb.Dataset(x, y, categorical_feature=self.cat_features)

        fraction_step = round((1 / len(self.features)) - 0.001, 4)

        def objective(trial):
            params = {
                'objective': self.objective,
                'n_estimators': self.n_estimators,
                'verbose': -1,

                'early_stopping_round': self.early_stopping_round,
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, step=0.000001),

                'feature_fraction': trial.suggest_float('feature_fraction', 0.45, 0.75, step=fraction_step),
                'num_leaves': trial.suggest_int('num_leaves', 10, 30),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.5, 4, step=0.00001),
                'lambda_l2': trial.suggest_float('lambda_l2', 1, 10, step=0.00001),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.95, step=fraction_step),
                'bagging_freq': trial.suggest_int('bagging_freq', 2, 200, step=2),
                # 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),

            }

            return self.regression_cv_score(params, train_set)

        study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)

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

        model = joblib.load(self.model_path)

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
                                   .apply(lambda df: skellam.pmf(list(range(1, 20)), df[0], df[1]).sum(), axis=1))

        predictions['draw'] = (predictions
                               .loc[:, ['home_goals', 'away_goals']]
                               .apply(lambda df: skellam.pmf(0, df[0], df[1]).sum(), axis=1))

        predictions['away_win'] = (1 - predictions['home_win'] - predictions['draw'])

        predictions.reset_index().to_feather(self.predictions_path)

        return predictions
