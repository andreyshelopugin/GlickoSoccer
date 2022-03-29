import joblib
import pandas as pd
import numpy as np
from euro_soccer.draw_model import DrawLightGBM


class DataPreprocessor(object):

    def __init__(self, min_season=2017, is_actual_draw_predictions=False):
        self.min_season = min_season
        self.is_actual_draw_predictions = is_actual_draw_predictions
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

        # # rename the same name teams
        # for season in matches['season'].unique():
        #     count_leagues = (matches
        #                      .loc[(matches['season'] == season)
        #                           & matches['tournament_type'].isin([1, 2])]
        #                      .drop_duplicates(['home_team', 'league'])
        #                      .groupby(['home_team'])
        #                      ['league']
        #                      .count()
        #                      .reset_index())
        #
        #     same_name_teams = count_leagues.loc[count_leagues['league'] > 1, 'home_team'].to_list()
        #
        #     print(same_name_teams)

        same_name_teams = ['Aris', 'Benfica', 'Bohemians', 'Concordia', 'Drita', 'Flamurtari', 'Rudar', 'Sloboda']

        matches['home_team'] = np.where(matches['home_team'].isin(same_name_teams) & (~matches['country'].isin(self.euro_cups)),
                                        (matches['home_team'] + ' ' + matches['country']),
                                        matches['home_team'])

        matches['away_team'] = np.where(matches['away_team'].isin(same_name_teams) & (~matches['country'].isin(self.euro_cups)),
                                        (matches['away_team'] + ' ' + matches['country']),
                                        matches['away_team'])

        matches['home_team'] = matches['home_team'].map(lambda x: x.replace("'", "").strip())
        matches['away_team'] = matches['away_team'].map(lambda x: x.replace("'", "").strip())

        return matches

    def draw_probability(self, matches: pd.DataFrame) -> pd.DataFrame:
        """"""
        if self.is_actual_draw_predictions:
            DrawLightGBM().actual_predictions(matches)

        draw_predictions = joblib.load('data/skellam_predictions.pkl')

        draw_predictions = dict(zip(draw_predictions['index'], draw_predictions['draw']))

        matches['draw_probability'] = matches['index'].map(draw_predictions)

        # we can not apply skellam model, just use mean as draw probability
        mean_draw = matches.loc[matches['outcome'] == 'D'].shape[0] / matches.shape[0]

        matches['draw_probability'] = matches['draw_probability'].fillna(mean_draw)

        return matches

    def _remove_matches_with_unknown_team(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Remove matches between teams from leagues we don't know anything about."""
        for season in matches['season'].unique():
            team_leagues = self._team_leagues(matches, season)

            known_teams = team_leagues.keys()

            is_both_teams_known = ((matches['season'] == season)
                                   & matches['home_team'].isin(known_teams)
                                   & matches['away_team'].isin(known_teams))

            matches = matches.loc[(matches['season'] != season) | is_both_teams_known]

        return matches

    @staticmethod
    def _team_leagues(results: pd.DataFrame, season: int) -> dict:
        """"""
        no_cups = (results.loc[results['tournament_type'].isin([1, 2])
                               & (results['season'] == season), ['home_team', 'away_team', 'tournament']])

        team_leagues = dict(zip(no_cups['home_team'], no_cups['tournament']))

        team_leagues.update(dict(zip(no_cups['away_team'], no_cups['tournament'])))

        return team_leagues

    @staticmethod
    def _get_finals(matches: pd.DataFrame) -> pd.DataFrame:
        """Get final matches for neutral field games detection"""

        max_season = matches['season'].max()

        finals = (matches
                  .loc[(matches['season'] != max_season) & (matches['tournament_type'] == 3)]
                  .sort_values(['tournament', 'season', 'date'])
                  .drop_duplicates(['season', 'tournament'], keep='last')
                  ['index']
                  .unique())

        matches['tournament_type'] = np.where(matches['index'].isin(finals) | (matches['notes'] == 'Neutral'),
                                              4,
                                              matches['tournament_type'])

        return matches

    def preprocessing(self, matches: pd.DataFrame) -> pd.DataFrame:
        """"""
        matches = matches.reset_index()

        matches['date'] = pd.to_datetime(matches['date'].str.replace('29.02', '28.02'), format='%d.%m.%Y', dayfirst=True)

        matches['season'] = matches['season'].map(lambda x: x.split('-')[0]).to_numpy('int')

        matches = matches.loc[matches['season'] >= self.min_season]

        matches = matches.sort_values(['date']).reset_index(drop=True)

        # only ratings, not further predictions
        matches = matches.loc[matches['home_score'] != '-']

        matches = matches.loc[matches['notes'] != 'Awrd']

        matches = self._rename_teams(matches)

        matches = matches.rename(columns={'league': 'tournament'})

        matches['tournament_type'] = matches['tournament_type'].map({'first': 1,
                                                                     'second': 2,
                                                                     'cups': 3,
                                                                     'super_cups': 4})

        matches[['home_score', 'away_score']] = matches[['home_score', 'away_score']].to_numpy('int')

        conditions = [(matches['home_score'] > matches['away_score']),
                      (matches['home_score'] == matches['away_score']),
                      (matches['home_score'] < matches['away_score'])]

        outcomes = ['H', 'D', 'A']
        matches['outcome'] = np.select(conditions, outcomes)

        # pandemic
        is_pandemic = (matches['date'] > '2020-03-03') & (matches['date'] < '2021-06-06')
        matches['is_pandemic'] = np.where(is_pandemic, 1, 0)

        short_tournaments = [
            'Azerbaijan. First Division',
            'Faroe Islands. 1. Deild',
            'Armenia. First League',
            # 'Uzbekistan. Super League',
            # 'Uzbekistan. Super League',
            # 'Uzbekistan. Uzbekistan Cup',
            # 'Uzbekistan. Super Cup'
        ]

        matches = matches.loc[~matches['tournament'].isin(short_tournaments)]

        matches = self._remove_matches_with_unknown_team(matches)

        matches = self.draw_probability(matches)

        matches = self._get_finals(matches)

        # technical wins
        # print(matches['notes'].value_counts())

        matches = (matches
                   .drop(columns=['home_score', 'away_score', 'notes'])
                   .sort_values(['date']))

        return matches

    @staticmethod
    def test_data(matches: pd.DataFrame):
        """"""
        count_matches = (matches
                         .loc[matches['tournament_type'].isin([1, 2]) & (matches['season'] != matches['season'].max())]
                         .groupby(['country', 'tournament_type', 'season'])
                         ['index']
                         .count()
                         .reset_index()
                         .rename(columns={'index': 'number_matches'}))

        count_matches = count_matches.loc[count_matches['number_matches'] < 50]

        count_seasons = (matches
                         .groupby(['country', 'tournament_type'])
                         ['season']
                         .nunique()
                         .reset_index()
                         .rename(columns={'season': 'number_seasons'}))

        count_seasons = count_seasons.loc[count_seasons['number_seasons'] < 3]

        count_tournaments = (matches
                             .loc[matches['tournament_type'] != 4]
                             .groupby(['country', 'season'])
                             ['tournament_type']
                             .nunique()
                             .reset_index()
                             .rename(columns={'season': 'number_tournaments'}))

        count_tournaments = count_tournaments.loc[count_tournaments['number_tournaments'] < 3]

        return count_matches, count_seasons, count_tournaments
