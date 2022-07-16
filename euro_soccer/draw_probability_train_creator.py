from typing import Dict

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class TrainCreator(object):

    def __init__(self, test_size=0.1, random_state=7):
        self.test_size = test_size
        self.random_state = random_state

    @staticmethod
    def calculate_stats(results: pd.DataFrame, is_home: bool, stats_type: str, window: int, min_periods: int,
                        is_against=False) -> Dict:
        """"""
        if is_home:
            team = 'home_team'
            if is_against:
                stats = 'away_score'
            else:
                stats = 'home_score'

        else:
            team = 'away_team'
            if is_against:
                stats = 'home_score'
            else:
                stats = 'away_score'

        group_by_object = (results
                           .sort_values([team, 'date'], ascending=[True, True])
                           .groupby([team], sort=False)
                           [stats]
                           .shift()
                           .rolling(window, min_periods=min_periods))

        if stats_type == 'mean':
            return group_by_object.mean().to_dict()

        elif stats_type == 'median':
            return group_by_object.median().to_dict()

        elif stats_type == 'min':
            return group_by_object.min().to_dict()

        elif stats_type == 'max':
            return group_by_object.max().to_dict()

        else:
            raise Exception("Wrong stats_type. Choose from [mean, median, min, max]")

    def features(self, results: pd.DataFrame) -> pd.DataFrame:
        """"""
        home_mean_score_5 = self.calculate_stats(results, True, 'mean', 5, 3)
        away_mean_score_5 = self.calculate_stats(results, False, 'mean', 5, 3)

        home_mean_score_10 = self.calculate_stats(results, True, 'mean', 10, 5)
        away_mean_score_10 = self.calculate_stats(results, False, 'mean', 10, 5)

        home_mean_score_20 = self.calculate_stats(results, True, 'mean', 20, 10)
        away_mean_score_20 = self.calculate_stats(results, False, 'mean', 20, 10)

        home_mean_score_30 = self.calculate_stats(results, True, 'mean', 30, 20)
        away_mean_score_30 = self.calculate_stats(results, False, 'mean', 30, 20)

        home_median_score_5 = self.calculate_stats(results, True, 'median', 5, 3)
        away_median_score_5 = self.calculate_stats(results, False, 'median', 5, 3)

        home_median_score_10 = self.calculate_stats(results, True, 'median', 10, 5)
        away_median_score_10 = self.calculate_stats(results, False, 'median', 10, 5)

        home_median_score_20 = self.calculate_stats(results, True, 'median', 20, 10)
        away_median_score_20 = self.calculate_stats(results, False, 'median', 20, 10)

        home_median_score_30 = self.calculate_stats(results, True, 'median', 30, 20)
        away_median_score_30 = self.calculate_stats(results, False, 'median', 30, 20)

        home_mean_score_5_against = self.calculate_stats(results, True, 'mean', 5, 3, True)
        away_mean_score_5_against = self.calculate_stats(results, False, 'mean', 5, 3, True)

        home_mean_score_10_against = self.calculate_stats(results, True, 'mean', 10, 5, True)
        away_mean_score_10_against = self.calculate_stats(results, False, 'mean', 10, 5, True)

        home_median_score_10_against = self.calculate_stats(results, True, 'median', 10, 5, True)
        away_median_score_10_against = self.calculate_stats(results, False, 'median', 10, 5, True)

        home_mean_score_20_against = self.calculate_stats(results, True, 'mean', 20, 10, True)
        away_mean_score_20_against = self.calculate_stats(results, False, 'mean', 20, 10, True)

        home_max_score_10 = self.calculate_stats(results, True, 'max', 10, 5)
        away_max_score_10 = self.calculate_stats(results, False, 'max', 10, 5)

        results = results.reset_index()

        results['home_mean_score_5'] = results['level_0'].map(home_mean_score_5)
        results['away_mean_score_5'] = results['level_0'].map(away_mean_score_5)

        results['home_mean_score_10'] = results['level_0'].map(home_mean_score_10)
        results['away_mean_score_10'] = results['level_0'].map(away_mean_score_10)

        results['home_mean_score_20'] = results['level_0'].map(home_mean_score_20)
        results['away_mean_score_20'] = results['level_0'].map(away_mean_score_20)

        results['home_mean_score_30'] = results['level_0'].map(home_mean_score_30)
        results['away_mean_score_30'] = results['level_0'].map(away_mean_score_30)

        results['home_median_score_5'] = results['level_0'].map(home_median_score_5)
        results['away_median_score_5'] = results['level_0'].map(away_median_score_5)

        results['home_median_score_10'] = results['level_0'].map(home_median_score_10)
        results['away_median_score_10'] = results['level_0'].map(away_median_score_10)

        results['home_median_score_20'] = results['level_0'].map(home_median_score_20)
        results['away_median_score_20'] = results['level_0'].map(away_median_score_20)

        results['home_median_score_30'] = results['level_0'].map(home_median_score_30)
        results['away_median_score_30'] = results['level_0'].map(away_median_score_30)

        results['home_mean_score_5_against'] = results['level_0'].map(home_mean_score_5_against)
        results['away_mean_score_5_against'] = results['level_0'].map(away_mean_score_5_against)

        results['home_mean_score_10_against'] = results['level_0'].map(home_mean_score_10_against)
        results['away_mean_score_10_against'] = results['level_0'].map(away_mean_score_10_against)

        results['home_mean_score_20_against'] = results['level_0'].map(home_mean_score_20_against)
        results['away_mean_score_20_against'] = results['level_0'].map(away_mean_score_20_against)

        results['home_median_score_10_against'] = results['level_0'].map(home_median_score_10_against)
        results['away_median_score_10_against'] = results['level_0'].map(away_median_score_10_against)

        results['home_max_score_10'] = results['level_0'].map(home_max_score_10)
        results['away_max_score_10'] = results['level_0'].map(away_max_score_10)

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

        avg_scoring_30 = (scoring
                          .sort_values(['team', 'date'], ascending=[True, True])
                          .groupby(['team'], sort=False)
                          ['score']
                          .shift()
                          .rolling(30, min_periods=20)
                          .mean()
                          .to_dict())

        scoring = scoring.reset_index()

        scoring['avg_scoring_5'] = scoring['level_0'].map(avg_scoring_5)
        scoring['avg_scoring_10'] = scoring['level_0'].map(avg_scoring_10)
        scoring['avg_scoring_20'] = scoring['level_0'].map(avg_scoring_20)
        scoring['avg_scoring_30'] = scoring['level_0'].map(avg_scoring_30)

        results_columns = ['index', 'is_pandemic',
                           'home_mean_score_5', 'away_mean_score_5',
                           'home_mean_score_10', 'away_mean_score_10',
                           'home_mean_score_20', 'away_mean_score_20',
                           'home_mean_score_30', 'away_mean_score_30',
                           'home_median_score_5', 'away_median_score_5',
                           'home_median_score_10', 'away_median_score_10',
                           'home_median_score_20', 'away_median_score_20',
                           'home_median_score_30', 'away_median_score_30',
                           'home_mean_score_5_against', 'away_mean_score_5_against',
                           'home_mean_score_10_against', 'away_mean_score_10_against',
                           'home_mean_score_20_against', 'away_mean_score_20_against',
                           'home_median_score_10_against', 'away_median_score_10_against',
                           'home_max_score_10', 'away_max_score_10']

        scoring = (scoring
                   .drop(columns=['date', 'level_0'])
                   .merge(results.loc[:, results_columns], how='left', on=['index']))

        is_home = (scoring['is_home'] == 1)
        scoring['location_mean_score_5'] = np.where(is_home, scoring['home_mean_score_5'], scoring['away_mean_score_5'])
        scoring['location_mean_score_10'] = np.where(is_home, scoring['home_mean_score_10'], scoring['away_mean_score_10'])
        scoring['location_mean_score_20'] = np.where(is_home, scoring['home_mean_score_20'], scoring['away_mean_score_20'])
        scoring['location_mean_score_30'] = np.where(is_home, scoring['home_mean_score_30'], scoring['away_mean_score_30'])

        scoring['location_median_score_5'] = np.where(is_home, scoring['home_median_score_5'], scoring['away_median_score_5'])
        scoring['location_median_score_10'] = np.where(is_home, scoring['home_median_score_10'], scoring['away_median_score_10'])
        scoring['location_median_score_20'] = np.where(is_home, scoring['home_median_score_20'], scoring['away_median_score_20'])
        scoring['location_median_score_30'] = np.where(is_home, scoring['home_median_score_30'], scoring['away_median_score_30'])

        scoring['location_mean_score_5_against'] = np.where(is_home,
                                                            scoring['home_mean_score_5_against'],
                                                            scoring['away_mean_score_5_against'])

        scoring['location_mean_score_10_against'] = np.where(is_home,
                                                             scoring['home_mean_score_10_against'],
                                                             scoring['away_mean_score_10_against'])

        scoring['location_mean_score_20_against'] = np.where(is_home,
                                                             scoring['home_mean_score_20_against'],
                                                             scoring['away_mean_score_20_against'])

        scoring['location_median_score_10_against'] = np.where(is_home,
                                                               scoring['home_median_score_10_against'],
                                                               scoring['away_median_score_10_against'])

        scoring['location_max_score_10'] = np.where(is_home, scoring['home_max_score_10'], scoring['away_max_score_10'])

        drop_columns = ['home_mean_score_5', 'away_mean_score_5',
                        'home_mean_score_10', 'away_mean_score_10',
                        'home_mean_score_20', 'away_mean_score_20',
                        'home_mean_score_30', 'away_mean_score_30',
                        'home_median_score_5', 'away_median_score_5',
                        'home_median_score_10', 'away_median_score_10',
                        'home_median_score_20', 'away_median_score_20',
                        'home_median_score_30', 'away_median_score_30',
                        'home_mean_score_5_against', 'away_mean_score_5_against',
                        'home_mean_score_10_against', 'away_mean_score_10_against',
                        'home_mean_score_20_against', 'away_mean_score_20_against',
                        'home_median_score_10_against', 'away_median_score_10_against',
                        'home_max_score_10', 'away_max_score_10']

        scoring = scoring.drop(columns=drop_columns)

        return scoring

    def train_validation(self, results: pd.DataFrame) -> tuple:
        """"""
        scoring = self.features(results)
        train, validation = train_test_split(scoring, test_size=self.test_size, shuffle=True,
                                             random_state=self.random_state)

        joblib.dump(train, 'data/train.pkl')
        joblib.dump(validation, 'data/validation.pkl')

        return train, validation

    def for_predictions(self, results: pd.DataFrame):
        """"""
        scoring = self.features(results)
        return scoring
