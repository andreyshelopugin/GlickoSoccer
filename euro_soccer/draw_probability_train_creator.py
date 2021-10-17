import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class TrainCreator(object):

    def __init__(self, test_size=0.2, random_state=7):
        self.test_size = test_size
        self.random_state = random_state

    def features(self, results: pd.DataFrame) -> pd.DataFrame:
        """"""
        home_mean_score_5 = (results
                             .sort_values(['home_team', 'date'], ascending=[True, True])
                             .groupby(['home_team'], sort=False)
                             ['home_score']
                             .shift()
                             .rolling(5, min_periods=3)
                             .mean()
                             .to_dict())

        away_mean_score_5 = (results
                             .sort_values(['away_team', 'date'], ascending=[True, True])
                             .groupby(['away_team'], sort=False)
                             ['away_score']
                             .shift()
                             .rolling(5, min_periods=3)
                             .mean()
                             .to_dict())

        home_mean_score_10 = (results
                              .sort_values(['home_team', 'date'], ascending=[True, True])
                              .groupby(['home_team'], sort=False)
                              ['home_score']
                              .shift()
                              .rolling(10, min_periods=5)
                              .mean()
                              .to_dict())

        away_mean_score_10 = (results
                              .sort_values(['away_team', 'date'], ascending=[True, True])
                              .groupby(['away_team'], sort=False)
                              ['away_score']
                              .shift()
                              .rolling(10, min_periods=5)
                              .mean()
                              .to_dict())

        home_median_score_10 = (results
                                .sort_values(['home_team', 'date'], ascending=[True, True])
                                .groupby(['home_team'], sort=False)
                                ['home_score']
                                .shift()
                                .rolling(10, min_periods=5)
                                .median()
                                .to_dict())

        away_median_score_10 = (results
                                .sort_values(['away_team', 'date'], ascending=[True, True])
                                .groupby(['away_team'], sort=False)
                                ['away_score']
                                .shift()
                                .rolling(10, min_periods=5)
                                .median()
                                .to_dict())

        home_mean_score_20 = (results
                              .sort_values(['home_team', 'date'], ascending=[True, True])
                              .groupby(['home_team'], sort=False)
                              ['home_score']
                              .shift()
                              .rolling(20, min_periods=10)
                              .median()
                              .to_dict())

        away_mean_score_20 = (results
                              .sort_values(['away_team', 'date'], ascending=[True, True])
                              .groupby(['away_team'], sort=False)
                              ['away_score']
                              .shift()
                              .rolling(20, min_periods=10)
                              .median()
                              .to_dict())

        home_mean_score_10_against = (results
                                      .sort_values(['home_team', 'date'], ascending=[True, True])
                                      .groupby(['home_team'], sort=False)
                                      ['away_score']
                                      .shift()
                                      .rolling(10, min_periods=5)
                                      .mean()
                                      .to_dict())

        away_mean_score_10_against = (results
                                      .sort_values(['away_team', 'date'], ascending=[True, True])
                                      .groupby(['away_team'], sort=False)
                                      ['home_score']
                                      .shift()
                                      .rolling(10, min_periods=5)
                                      .mean()
                                      .to_dict())

        home_median_score_10_against = (results
                                        .sort_values(['home_team', 'date'], ascending=[True, True])
                                        .groupby(['home_team'], sort=False)
                                        ['away_score']
                                        .shift()
                                        .rolling(10, min_periods=5)
                                        .median()
                                        .to_dict())

        away_median_score_10_against = (results
                                        .sort_values(['away_team', 'date'], ascending=[True, True])
                                        .groupby(['away_team'], sort=False)
                                        ['home_score']
                                        .shift()
                                        .rolling(10, min_periods=5)
                                        .median()
                                        .to_dict())

        home_mean_score_20_against = (results
                                      .sort_values(['home_team', 'date'], ascending=[True, True])
                                      .groupby(['home_team'], sort=False)
                                      ['away_score']
                                      .shift()
                                      .rolling(20, min_periods=10)
                                      .median()
                                      .to_dict())

        away_mean_score_20_against = (results
                                      .sort_values(['away_team', 'date'], ascending=[True, True])
                                      .groupby(['away_team'], sort=False)
                                      ['home_score']
                                      .shift()
                                      .rolling(20, min_periods=10)
                                      .median()
                                      .to_dict())

        results = results.reset_index()

        results['home_mean_score_5'] = results['level_0'].map(home_mean_score_5)
        results['away_mean_score_5'] = results['level_0'].map(away_mean_score_5)

        results['home_mean_score_10'] = results['level_0'].map(home_mean_score_10)
        results['away_mean_score_10'] = results['level_0'].map(away_mean_score_10)

        results['home_median_score_10'] = results['level_0'].map(home_median_score_10)
        results['away_median_score_10'] = results['level_0'].map(away_median_score_10)

        results['home_mean_score_20'] = results['level_0'].map(home_mean_score_20)
        results['away_mean_score_20'] = results['level_0'].map(away_mean_score_20)

        results['home_mean_score_10_against'] = results['level_0'].map(home_mean_score_10_against)
        results['away_mean_score_10_against'] = results['level_0'].map(away_mean_score_10_against)

        results['home_median_score_10_against'] = results['level_0'].map(home_median_score_10_against)
        results['away_median_score_10_against'] = results['level_0'].map(away_median_score_10_against)

        results['home_mean_score_20_against'] = results['level_0'].map(home_mean_score_20_against)
        results['away_mean_score_20_against'] = results['level_0'].map(away_mean_score_20_against)

        results = results.drop(columns=['level_0'])

        home_scoring = (results
                        .loc[:, ['date', 'home_team', 'home_score', 'index']]
                        .rename(columns={'home_team': 'team', 'home_score': 'score'}))

        away_scoring = (results
                        .loc[:, ['date', 'away_team', 'away_score', 'index']]
                        .rename(columns={'away_team': 'team', 'away_score': 'score'}))

        home_scoring['is_home'] = 1
        away_scoring['is_home'] = 0

        scoring = pd.concat([home_scoring, away_scoring])

        avg_scoring_5 = (scoring
                         .sort_values(['team', 'date'], ascending=[True, True])
                         .groupby(['team'], sort=False)
                         ['score']
                         .shift()
                         .rolling(5, min_periods=3)
                         .mean()
                         .to_dict())

        avg_scoring_10 = (scoring
                          .sort_values(['team', 'date'], ascending=[True, True])
                          .groupby(['team'], sort=False)
                          ['score']
                          .shift()
                          .rolling(10, min_periods=5)
                          .mean()
                          .to_dict())

        avg_scoring_20 = (scoring
                          .sort_values(['team', 'date'], ascending=[True, True])
                          .groupby(['team'], sort=False)
                          ['score']
                          .shift()
                          .rolling(20, min_periods=10)
                          .mean()
                          .to_dict())

        scoring = scoring.reset_index()

        scoring['avg_scoring_5'] = scoring['level_0'].map(avg_scoring_5)
        scoring['avg_scoring_10'] = scoring['level_0'].map(avg_scoring_10)
        scoring['avg_scoring_20'] = scoring['level_0'].map(avg_scoring_20)

        results_columns = ['index', 'is_pandemic', 'home_mean_score_5', 'away_mean_score_5', 'home_mean_score_10',
                           'away_mean_score_10', 'home_median_score_10', 'away_median_score_10',
                           'home_mean_score_20', 'away_mean_score_20',
                           'home_mean_score_10_against', 'away_mean_score_10_against',
                           'home_median_score_10_against', 'away_median_score_10_against',
                           'home_mean_score_20_against', 'away_mean_score_20_against']

        scoring = (scoring
                   .drop(columns=['date', 'level_0'])
                   .merge(results.loc[:, results_columns], how='left', on=['index']))

        is_home = (scoring['is_home'] == 1)
        scoring['mean_score_10'] = np.where(is_home, scoring['home_mean_score_10'], scoring['away_mean_score_10'])
        scoring['median_score_10'] = np.where(is_home, scoring['home_median_score_10'], scoring['away_median_score_10'])
        scoring['mean_score_20'] = np.where(is_home, scoring['home_mean_score_20'], scoring['away_mean_score_20'])

        scoring['mean_score_10_against'] = np.where(is_home, scoring['home_mean_score_10_against'],
                                                    scoring['away_mean_score_10_against'])

        scoring['median_score_10_against'] = np.where(is_home, scoring['home_median_score_10_against'],
                                                      scoring['away_median_score_10_against'])

        scoring['mean_score_20_against'] = np.where(is_home, scoring['home_mean_score_20_against'],
                                                    scoring['away_mean_score_20_against'])

        drop_columns = ['home_mean_score_10', 'away_mean_score_10', 'home_median_score_10', 'away_median_score_10',
                        'home_mean_score_20', 'away_mean_score_20', 'home_mean_score_10_against',
                        'away_mean_score_10_against', 'home_median_score_10_against', 'away_median_score_10_against',
                        'home_mean_score_20_against', 'away_mean_score_20_against']

        scoring = scoring.drop(columns=drop_columns)

        return scoring

    def train_validation(self, results: pd.DataFrame):
        """"""
        scoring = self.features(results)
        train, validation = train_test_split(scoring, test_size=self.test_size, shuffle=True,
                                             random_state=self.random_state)

        joblib.dump(train, 'data/train.pkl')
        joblib.dump(validation, 'data/validation.pkl')

    def for_predictions(self, results: pd.DataFrame):
        """"""
        scoring = self.features(results)
        return scoring
