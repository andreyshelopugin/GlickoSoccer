from typing import Tuple, List

import joblib
import pandas as pd

from lgbm_glicko.features import Features
from lgbm_glicko.preprocessing import DataPreprocessor


class TrainTestCreator(object):

    def __init__(self, last_train_week=19, current_season=2020):
        self.last_train_week = last_train_week
        self.current_season = current_season

    def train_test_one_season(self, matches: pd.DataFrame, season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """"""
        if season == self.current_season:
            train = matches.loc[matches['goals'].notna()]
            test = matches.loc[matches['goals'].isna()]

        else:
            train = matches.loc[matches['round'] <= self.last_train_week]
            test = matches.loc[matches['round'] > self.last_train_week]

            # leave stats before last_train_week
            team_stats = (test
                              .sort_values(['date', 'round'])
                              .drop_duplicates(['team_id'], keep='first')
                              .loc[:, ['team_id', 'avg_goals', 'avg_goals_against', 'avg_xg', 'avg_xg_against']])

            opp_stats = (team_stats
                         .rename(columns={'team_id': 'opponent_id',
                                          'avg_goals': 'opp_avg_goals',
                                          'avg_goals_against': 'opp_avg_goals_against',
                                          'avg_xg': 'opp_avg_xg',
                                          'avg_xg_against': 'opp_avg_xg_against'}))

            test = (test
                    .loc[:, ['date', 'round', 'team', 'opponent', 'goals', 'is_home', 'team_id', 'opponent_id']]
                    .merge(team_stats, how='left', on=['team_id'])
                    .merge(opp_stats, how='left', on=['opponent_id']))

        train['avg_goals_difference'] = (train['avg_goals'] + train['opp_avg_goals_against']) / 2
        train['avg_xg_difference'] = (train['avg_xg'] + train['opp_avg_xg_against']) / 2

        test['avg_goals_difference'] = (test['avg_goals'] + test['opp_avg_goals_against']) / 2
        test['avg_xg_difference'] = (test['avg_xg'] + test['opp_avg_xg_against']) / 2

        return train, test

    def train_test(self, tournaments: List[str], seasons: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
            Create train, test sets for LightGBM model
        """
        train_list = []
        test_list = []
        for tournament in tournaments:
            for season in seasons:
                matches = pd.read_excel('data/xg_' + tournament + '.xlsx', sheet_name=str(season))
                matches = DataPreprocessor().preprocessing_one_season(matches)
                matches = DataPreprocessor().preprocessing_for_lgbm(matches)
                matches = Features().features(matches)

                train, test = self.train_test_one_season(matches, season)

                train['tournament'] = tournament
                test['tournament'] = tournament

                train['season'] = season
                test['season'] = season

                train_list.append(train)
                test_list.append(test)

        train = pd.concat(train_list)
        test = pd.concat(test_list)

        joblib.dump(train, 'data/train.pkl')
        joblib.dump(test, 'data/test.pkl')

        return train, test


