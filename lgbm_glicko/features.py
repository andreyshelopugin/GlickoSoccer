import numpy as np
import pandas as pd


class Features(object):

    @staticmethod
    def features(matches: pd.DataFrame) -> pd.DataFrame:
        """
            Features for LightGBM model for one season
        """
        teams = matches['team'].unique()

        teams_to_index = dict()
        i = 1
        for team in teams:
            teams_to_index[team] = i
            i += 1

        matches['team_id'] = matches['team'].map(teams_to_index)
        matches['opponent_id'] = matches['opponent'].map(teams_to_index)

        matches = matches.sort_values(['date', 'round']).reset_index(drop=True).reset_index()

        avg_goals = (matches
                     .groupby(['team_id'])
                     ['goals']
                     .expanding()
                     .mean()
                     .reset_index()
                     .rename(columns={'level_1': 'index', 'goals': 'avg_goals'}))

        avg_goals_against = (matches
                             .groupby(['team_id'])
                             ['goals_against']
                             .expanding()
                             .mean()
                             .reset_index()
                             .rename(columns={'level_1': 'index', 'goals_against': 'avg_goals_against'}))

        avg_xg = (matches
                  .groupby(['team_id'])
                  ['xg']
                  .expanding()
                  .mean()
                  .reset_index()
                  .rename(columns={'level_1': 'index', 'xg': 'avg_xg'}))

        avg_xg_against = (matches
                          .groupby(['team_id'])
                          ['xg_against']
                          .expanding()
                          .mean()
                          .reset_index()
                          .rename(columns={'level_1': 'index', 'xg_against': 'avg_xg_against'}))

        matches = (matches
                   .merge(avg_goals, on=['team_id', 'index'])
                   .merge(avg_goals_against, on=['team_id', 'index'])
                   .merge(avg_xg, on=['team_id', 'index'])
                   .merge(avg_xg_against, on=['team_id', 'index'])
                   .drop(columns=['index'])
                   .sort_values(['team_id', 'date', 'round']))

        features = ['avg_goals', 'avg_goals_against', 'avg_xg', 'avg_xg_against']
        for column in features:
            matches[column] = np.where(matches['round'] == 1, 0, matches[column].shift())

        opponents = (matches
                     .loc[:, ['round', 'team_id', 'avg_goals', 'avg_goals_against', 'avg_xg', 'avg_xg_against']]
                     .rename(columns={'team_id': 'opponent_id',
                                      'avg_goals': 'opp_avg_goals',
                                      'avg_goals_against': 'opp_avg_goals_against',
                                      'avg_xg': 'opp_avg_xg',
                                      'avg_xg_against': 'opp_avg_xg_against'}))

        matches = matches.merge(opponents, how='left', on=['round', 'opponent_id'])

        return matches
