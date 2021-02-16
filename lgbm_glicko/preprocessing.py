from typing import List

import numpy as np
import pandas as pd


class DataPreprocessor(object):

    @staticmethod
    def preprocessing_one_season(matches: pd.DataFrame) -> pd.DataFrame:
        """"""

        matches = (matches
                   .loc[matches['Wk'].notna(), ['Date', 'Wk', 'Home', 'Away', 'xG', 'Score', 'xG.1']]
                   .rename(columns={'Date': 'date', 'xG': 'home_xg', 'xG.1': 'away_xg', 'Wk': 'round',
                                    'Home': 'home_team', 'Away': 'away_team'}))

        matches['home_score'], matches['away_score'] = matches['Score'].fillna('–').str.split('–', 1).str

        matches['home_score'] = matches['home_score'].replace('', np.nan)
        matches['away_score'] = matches['away_score'].replace('', np.nan)

        # outcome, score is nan means future match
        conditions = [(matches['home_score'] > matches['away_score']),
                      (matches['home_score'] == matches['away_score']),
                      (matches['home_score'] < matches['away_score'])]

        outcomes = ['H', 'D', 'A']
        matches['outcome'] = np.select(conditions, outcomes, default='F')

        # xG outcome
        conditions = [(matches['home_xg'] > matches['away_xg'] + 0.5),
                      (matches['home_xg'] - matches['away_xg']).abs() <= 0.5,
                      (matches['home_xg'] + 0.5 < matches['away_xg'])]

        matches['xg_outcome'] = np.select(conditions, outcomes, default='F')

        matches['round'] = matches['round'].to_numpy('int')

        matches = matches.drop(columns=['Score']).sort_values(['date', 'round']).reset_index(drop=True)

        return matches

    @staticmethod
    def preprocessing_for_lgbm(matches: pd.DataFrame) -> pd.DataFrame:
        """"""

        home = (matches
                .loc[:, ['date', 'round', 'home_team', 'away_team', 'home_xg', 'away_xg', 'home_score', 'away_score']]
                .rename(columns={'home_team': 'team', 'away_team': 'opponent', 'home_xg': 'xg', 'home_score': 'goals',
                                 'away_score': 'goals_against', 'away_xg': 'xg_against'}))

        home['is_home'] = 1

        away = (matches
                .loc[:, ['date', 'round', 'away_team', 'home_team', 'away_xg', 'home_xg', 'away_score', 'home_score']]
                .rename(columns={'away_team': 'team', 'home_team': 'opponent', 'away_xg': 'xg', 'away_score': 'goals',
                                 'home_score': 'goals_against', 'home_xg': 'xg_against'}))

        away['is_home'] = 0

        matches = pd.concat([home, away])

        matches = matches.sort_values(['date', 'round'])

        return matches

    @staticmethod
    def glicko_preprocessing(tournaments: List[str], seasons: List[int]) -> pd.DataFrame:
        """"""
        matches_list = []
        for tournament in tournaments:
            for season in seasons:
                matches = pd.read_excel('data/xg_' + tournament + '.xlsx', sheet_name=str(season))
                matches = DataPreprocessor().preprocessing_one_season(matches)
                matches['tournament'] = tournament
                matches['season'] = season
                matches_list.append(matches)

        matches = pd.concat(matches_list)

        return matches

