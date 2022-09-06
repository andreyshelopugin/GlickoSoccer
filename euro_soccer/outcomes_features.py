import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config


class TrainCreator(object):

    def __init__(self, test_size=0.1, random_state=7):
        self.test_size = test_size
        self.random_state = random_state

    @staticmethod
    def calculate_rolling_stats(matches: pd.DataFrame, is_home: bool, stats_type: str, window: int, min_periods: int,
                                is_against=False) -> dict:
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

        group_by_object = (matches
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

    @staticmethod
    def _team_leagues(matches: pd.DataFrame) -> pd.DataFrame:
        """Each team match with her league for specific season."""

        team_leagues = (matches
                        .loc[matches['tournament_type'].isin({1, 2}), ['home_team', 'tournament', 'season']]
                        .drop_duplicates(['home_team', 'season'], keep='first')
                        .rename(columns={'home_team': 'team', 'tournament': 'league'}))

        return team_leagues

    def features(self, matches: pd.DataFrame) -> pd.DataFrame:
        """"""
        home_mean_score_5 = self.calculate_rolling_stats(matches, True, 'mean', 5, 3)
        away_mean_score_5 = self.calculate_rolling_stats(matches, False, 'mean', 5, 3)

        home_mean_score_10 = self.calculate_rolling_stats(matches, True, 'mean', 10, 5)
        away_mean_score_10 = self.calculate_rolling_stats(matches, False, 'mean', 10, 5)

        home_mean_score_20 = self.calculate_rolling_stats(matches, True, 'mean', 20, 10)
        away_mean_score_20 = self.calculate_rolling_stats(matches, False, 'mean', 20, 10)

        home_mean_score_30 = self.calculate_rolling_stats(matches, True, 'mean', 30, 20)
        away_mean_score_30 = self.calculate_rolling_stats(matches, False, 'mean', 30, 20)

        home_median_score_5 = self.calculate_rolling_stats(matches, True, 'median', 5, 3)
        away_median_score_5 = self.calculate_rolling_stats(matches, False, 'median', 5, 3)

        home_median_score_10 = self.calculate_rolling_stats(matches, True, 'median', 10, 5)
        away_median_score_10 = self.calculate_rolling_stats(matches, False, 'median', 10, 5)

        home_median_score_20 = self.calculate_rolling_stats(matches, True, 'median', 20, 10)
        away_median_score_20 = self.calculate_rolling_stats(matches, False, 'median', 20, 10)

        home_median_score_30 = self.calculate_rolling_stats(matches, True, 'median', 30, 20)
        away_median_score_30 = self.calculate_rolling_stats(matches, False, 'median', 30, 20)

        home_mean_score_5_against = self.calculate_rolling_stats(matches, True, 'mean', 5, 3, True)
        away_mean_score_5_against = self.calculate_rolling_stats(matches, False, 'mean', 5, 3, True)

        home_mean_score_10_against = self.calculate_rolling_stats(matches, True, 'mean', 10, 5, True)
        away_mean_score_10_against = self.calculate_rolling_stats(matches, False, 'mean', 10, 5, True)

        home_median_score_10_against = self.calculate_rolling_stats(matches, True, 'median', 10, 5, True)
        away_median_score_10_against = self.calculate_rolling_stats(matches, False, 'median', 10, 5, True)

        home_mean_score_20_against = self.calculate_rolling_stats(matches, True, 'mean', 20, 10, True)
        away_mean_score_20_against = self.calculate_rolling_stats(matches, False, 'mean', 20, 10, True)

        home_max_score_10 = self.calculate_rolling_stats(matches, True, 'max', 10, 5)
        away_max_score_10 = self.calculate_rolling_stats(matches, False, 'max', 10, 5)

        # for get "index"
        matches = matches.reset_index()

        team_leagues = self._team_leagues(matches)

        matches = (matches
                   .merge(team_leagues.rename(columns={'league': 'home_league'}),
                          how='left', left_on=['home_team', 'season'], right_on=['team', 'season'])
                   .merge(team_leagues.rename(columns={'league': 'away_league'}),
                          how='left', left_on=['away_team', 'season'], right_on=['team', 'season']))

        matches['home_mean_score_5'] = matches['index'].map(home_mean_score_5)
        matches['away_mean_score_5'] = matches['index'].map(away_mean_score_5)

        matches['home_mean_score_10'] = matches['index'].map(home_mean_score_10)
        matches['away_mean_score_10'] = matches['index'].map(away_mean_score_10)

        matches['home_mean_score_20'] = matches['index'].map(home_mean_score_20)
        matches['away_mean_score_20'] = matches['index'].map(away_mean_score_20)

        matches['home_mean_score_30'] = matches['index'].map(home_mean_score_30)
        matches['away_mean_score_30'] = matches['index'].map(away_mean_score_30)

        matches['home_median_score_5'] = matches['index'].map(home_median_score_5)
        matches['away_median_score_5'] = matches['index'].map(away_median_score_5)

        matches['home_median_score_10'] = matches['index'].map(home_median_score_10)
        matches['away_median_score_10'] = matches['index'].map(away_median_score_10)

        matches['home_median_score_20'] = matches['index'].map(home_median_score_20)
        matches['away_median_score_20'] = matches['index'].map(away_median_score_20)

        matches['home_median_score_30'] = matches['index'].map(home_median_score_30)
        matches['away_median_score_30'] = matches['index'].map(away_median_score_30)

        matches['home_mean_score_5_against'] = matches['index'].map(home_mean_score_5_against)
        matches['away_mean_score_5_against'] = matches['index'].map(away_mean_score_5_against)

        matches['home_mean_score_10_against'] = matches['index'].map(home_mean_score_10_against)
        matches['away_mean_score_10_against'] = matches['index'].map(away_mean_score_10_against)

        matches['home_mean_score_20_against'] = matches['index'].map(home_mean_score_20_against)
        matches['away_mean_score_20_against'] = matches['index'].map(away_mean_score_20_against)

        matches['home_median_score_10_against'] = matches['index'].map(home_median_score_10_against)
        matches['away_median_score_10_against'] = matches['index'].map(away_median_score_10_against)

        matches['home_max_score_10'] = matches['index'].map(home_max_score_10)
        matches['away_max_score_10'] = matches['index'].map(away_max_score_10)

        matches = matches.drop(columns=['index'])

        home_scoring = (matches
                        .loc[:, ['date', 'home_team', 'away_team', 'home_score', 'match_id']]
                        .rename(columns={'home_team': 'team', 'away_team': 'opp_team', 'home_score': 'score'}))

        away_scoring = (matches
                        .loc[:, ['date', 'away_team', 'home_team', 'away_score', 'match_id']]
                        .rename(columns={'away_team': 'team', 'home_team': 'opp_team', 'away_score': 'score'}))

        home_scoring['is_home'] = True
        away_scoring['is_home'] = False

        scoring = pd.concat([home_scoring, away_scoring])

        scoring = scoring.reset_index(drop=True).reset_index()

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

        avg_scoring_5_against = (scoring
                                 .sort_values(['opp_team', 'date'], ascending=[True, True])
                                 .groupby(['opp_team'], sort=False)
                                 ['score']
                                 .shift()
                                 .rolling(5, min_periods=3)
                                 .mean()
                                 .to_dict())

        avg_scoring_10_against = (scoring
                                  .sort_values(['opp_team', 'date'], ascending=[True, True])
                                  .groupby(['opp_team'], sort=False)
                                  ['score']
                                  .shift()
                                  .rolling(10, min_periods=5)
                                  .mean()
                                  .to_dict())

        avg_scoring_20_against = (scoring
                                  .sort_values(['opp_team', 'date'], ascending=[True, True])
                                  .groupby(['opp_team'], sort=False)
                                  ['score']
                                  .shift()
                                  .rolling(20, min_periods=10)
                                  .mean()
                                  .to_dict())

        avg_scoring_30_against = (scoring
                                  .sort_values(['opp_team', 'date'], ascending=[True, True])
                                  .groupby(['opp_team'], sort=False)
                                  ['score']
                                  .shift()
                                  .rolling(30, min_periods=20)
                                  .mean()
                                  .to_dict())

        scoring['avg_scoring_5'] = scoring['index'].map(avg_scoring_5)
        scoring['avg_scoring_10'] = scoring['index'].map(avg_scoring_10)
        scoring['avg_scoring_20'] = scoring['index'].map(avg_scoring_20)
        scoring['avg_scoring_30'] = scoring['index'].map(avg_scoring_30)

        scoring['avg_scoring_5_against'] = scoring['index'].map(avg_scoring_5_against)
        scoring['avg_scoring_10_against'] = scoring['index'].map(avg_scoring_10_against)
        scoring['avg_scoring_20_against'] = scoring['index'].map(avg_scoring_20_against)
        scoring['avg_scoring_30_against'] = scoring['index'].map(avg_scoring_30_against)

        matches_columns = ['match_id',
                           'season',
                           'tournament_type', 'tournament',
                           'home_league', 'away_league',
                           'is_pandemic',
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
                   .drop(columns=['date', 'index'])
                   .merge(matches.loc[:, matches_columns], how='left', on=['match_id']))

        is_home = (scoring['is_home'] == 1)
        scoring['league'] = np.where(is_home, scoring['home_league'], scoring['away_league'])
        scoring['opp_league'] = np.where(is_home, scoring['away_league'], scoring['home_league'])

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

        drop_columns = ['home_league', 'away_league',
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

        scoring = scoring.drop(columns=drop_columns)

        cat_features = ['tournament_type', 'tournament', 'league', 'opp_league']

        scoring[cat_features] = scoring[cat_features].fillna('NaN')

        return scoring

    def train_validation_test(self, matches: pd.DataFrame) -> tuple:
        """Leave two seasons for test set."""
        matches = self.features(matches)

        train_validation = matches.loc[matches['season'] < 2020]
        test = matches.loc[matches['season'] >= 2020]

        train, validation = train_test_split(train_validation, test_size=self.test_size, shuffle=True,
                                             random_state=self.random_state)

        joblib.dump(train, Config().project_path + Config().outcomes_paths['train'])
        joblib.dump(validation, Config().project_path + Config().outcomes_paths['validation'])
        joblib.dump(test, Config().project_path + Config().outcomes_paths['test'])

        return train, validation, test

    def for_predictions(self, matches: pd.DataFrame):
        """"""
        matches = self.features(matches)
        return matches
