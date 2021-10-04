import pandas as pd
import numpy as np


class DataPreprocessor(object):

    def __init__(self):
        self.euro_cups = ['Europa League', 'Champions League', 'Europa Conference League']

    def _rename_teams(self, matches: pd.DataFrame):
        """"""

        uefa_matches = matches.loc[matches['country'].isin(self.euro_cups)]

        bracket_teams = (set(uefa_matches.loc[(matches['home_team'].map(lambda x: 1 if '(' in x else 0) == 1), 'home_team'])
                         .union(set(uefa_matches.loc[(matches['away_team'].map(lambda x: 1 if '(' in x else 0) == 1), 'away_team'])))

        renaming_teams = {}
        for team in bracket_teams:
            renaming_teams[team] = team.split('(')[0].strip()

        matches['home_team'] = matches['home_team'].map(lambda x: renaming_teams[x] if x in renaming_teams else x)
        matches['away_team'] = matches['away_team'].map(lambda x: renaming_teams[x] if x in renaming_teams else x)

        # rename the same name teams
        # count_leagues = (matches
        #                  .loc[~matches['league'].isin(self.euro_cups)]
        #                  .drop_duplicates(['home_team', 'league'])
        #                  .groupby(['home_team'])
        #                  ['league']
        #                  .count()
        #                  .reset_index())
        #
        # same_name_teams = count_leagues.loc[count_leagues['league'] > 1, 'home_team'].to_list()

        same_name_teams = ['Aris', 'Benfica', 'Bohemians', 'Concordia', 'Drita', 'Flamurtari', 'Rudar', 'Sloboda']

        matches['home_team'] = np.where(matches['home_team'].isin(same_name_teams) & (~matches['country'].isin(self.euro_cups)),
                                        (matches['home_team'] + ' ' + matches['country']),
                                        matches['home_team'])

        matches['away_team'] = np.where(matches['away_team'].isin(same_name_teams) & (~matches['country'].isin(self.euro_cups)),
                                        (matches['away_team'] + ' ' + matches['country']),
                                        matches['away_team'])

        return matches

    def preprocessing(self, matches: pd.DataFrame) -> pd.DataFrame:
        """"""
        matches = matches.reset_index()

        matches['date'] = pd.to_datetime(matches['date'].str.replace('29.02', '28.02'), format='%d.%m.%Y', dayfirst=True)

        matches['season'] = matches['season'].map(lambda x: x.split('-')[0]).to_numpy('int')

        matches = matches.sort_values(['date']).reset_index(drop=True)

        matches = self._rename_teams(matches)

        matches = matches.rename(columns={'league': 'tournament'})

        matches['tournament_type'] = matches['tournament_type'].map({'first': 1,
                                                                     'second': 2,
                                                                     'cups': 3})

        return matches
