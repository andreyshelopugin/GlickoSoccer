import joblib
import pandas as pd
import numpy as np
from euro_soccer.draw_model import DrawLightGBM


class DataPreprocessor(object):

    def __init__(self, min_season=2017, is_actual_draw_predictions=False):
        self.min_season = min_season
        self.is_actual_draw_predictions = is_actual_draw_predictions
        self.euro_cups = {'Europa League', 'Champions League', 'Europa Conference League'}

    def _rename_teams(self, matches: pd.DataFrame) -> pd.DataFrame:
        """"""

        uefa_matches = matches.loc[matches['country'].isin(self.euro_cups)]

        # country name in brackets
        bracket_teams = (set(uefa_matches.loc[(matches['home_team'].map(lambda x: 1 if '(' in x else 0) == 1), 'home_team'])
                         .union(set(uefa_matches.loc[(matches['away_team'].map(lambda x: 1 if '(' in x else 0) == 1), 'away_team'])))

        renaming_teams = {}
        for team in bracket_teams:
            renaming_teams[team] = team.split('(')[0].strip()

        matches['home_team'] = matches['home_team'].map(lambda x: renaming_teams[x] if x in renaming_teams else x)
        matches['away_team'] = matches['away_team'].map(lambda x: renaming_teams[x] if x in renaming_teams else x)

        duplicates_names = [('Бенфика', 'Luxembourg'),
                            ('Арис', 'Cyprus'),
                            ('Богемианс', 'Ireland'),
                            ('Горица', 'Croatia'),
                            ('Дрита', 'North Macedonia'),
                            ('Ноа', 'Latvia'),
                            ('Рудар', 'Montenegro')]

        for team, country in duplicates_names:
            for location_team in ['home_team', 'away_team']:
                is_duplicate_team = ((matches[location_team] == team) & (matches['country'] == country))

                matches[team] = np.where(is_duplicate_team,
                                         (matches[location_team] + ' ' + matches['country']),
                                         matches[location_team])

        same_name_teams = ['Concordia', 'Flamurtari', 'Sloboda', 'Звезда', 'Флямуртари']

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
        no_cups = (results.loc[results['tournament_type'].isin({1, 2})
                               & (results['season'] == season), ['home_team', 'away_team', 'tournament']])

        team_leagues = dict(zip(no_cups['home_team'], no_cups['tournament']))

        team_leagues.update(dict(zip(no_cups['away_team'], no_cups['tournament'])))

        return team_leagues

    @staticmethod
    def _get_finals(matches: pd.DataFrame, is_end_of_season=True) -> pd.DataFrame:
        """Get final matches for neutral field games detection"""

        if is_end_of_season:
            condition = (matches['tournament_type'] == 3)
        else:
            condition = (matches['season'] != matches['season'].max()) & (matches['tournament_type'] == 3)

        finals = (matches
                  .loc[condition]
                  .sort_values(['tournament', 'season', 'date'])
                  .drop_duplicates(['season', 'tournament'], keep='last')
                  ['index']
                  .unique())

        matches['tournament_type'] = np.where(matches['index'].isin(finals) | (matches['notes'] == 'Neutral'),
                                              4,
                                              matches['tournament_type'])

        matches['outcome'] = np.where((matches['tournament_type'] == 4) & matches['notes'].isin({'Послесп'}),
                                      'D',
                                      matches['outcome'])

        return matches

    def preprocessing(self, matches: pd.DataFrame) -> pd.DataFrame:
        """"""
        matches = matches.reset_index()

        matches['date'] = pd.to_datetime(matches['date'].str.replace('29.02', '28.02'), format='%d.%m.%Y', dayfirst=True)

        matches['season'] = matches['season'].map(lambda x: x.split('-')[0]).to_numpy('int')

        matches = (matches
                   .loc[matches['season'] >= self.min_season]
                   .sort_values(['date'])
                   .reset_index(drop=True))

        # drop future matches: need only ratings, not further predictions
        matches = matches.loc[matches['home_score'] != '-']

        matches = matches.loc[~matches['notes'].isin({"Awrd", "Техпоражение", "Неявка"})]

        matches = self._rename_teams(matches)

        matches["tournament"] = np.where(matches["league"].isin(self.euro_cups),
                                         matches["league"],
                                         matches["country"] + '. ' + matches["tournament_type"].str.title())

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

        # matches_copy = matches.copy()

        matches = self._remove_matches_with_unknown_team(matches)

        # print(matches_copy.loc[(~matches_copy['index'].isin(matches['index']))
        #                        & (matches_copy['country'].isin(self.euro_cups))].shape)
        #
        # print(matches_copy.loc[(~matches_copy['index'].isin(matches['index']))
        #                        & (matches_copy['country'].isin(self.euro_cups))].head())

        matches = self.draw_probability(matches)

        matches = self._get_finals(matches)

        # technical wins
        # print(matches['notes'].value_counts())

        matches = (matches
                   # .drop(columns=['home_score', 'away_score', 'notes'])
                   .sort_values(['date']))

        return matches

    def test_data(self, matches: pd.DataFrame):
        """"""
        # short season tournaments
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

        # find teams were participated in different countries
        teams_countries = (matches
                           .loc[matches['tournament_type'].isin({1, 2})]
                           .groupby(["home_team"])
                           ["country"]
                           .nunique()
                           .reset_index()
                           .rename(columns={'country': 'teams_countries'}))

        teams_countries = teams_countries.loc[teams_countries["teams_countries"] > 1]

        # teams in eurocups without nation league
        uefa_teams = (set(matches.loc[matches['country'].isin(self.euro_cups), 'home_team'])
                      .union(set(matches.loc[matches['country'].isin(self.euro_cups), 'away_team'])))

        national_teams = set(matches.loc[matches['tournament_type'].isin({1, 2}), 'home_team'])

        uefa_teams_without_national = sorted([t for t in uefa_teams if t not in national_teams])

        return count_matches, count_seasons, count_tournaments, teams_countries, uefa_teams_without_national
