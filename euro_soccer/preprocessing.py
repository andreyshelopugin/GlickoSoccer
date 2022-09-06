import joblib
import pandas as pd
import numpy as np
from euro_soccer.outcomes_features import TrainCreator
from euro_soccer.outcomes_catboost import OutcomesCatBoost
from config import Config


class DataPreprocessor(object):

    def __init__(self, min_season=2010, max_season=2021, is_actual_draw_predictions=False):
        self.min_season = min_season
        self.max_season = max_season
        self.is_actual_draw_predictions = is_actual_draw_predictions
        self.international_cups = {'Europa League', 'Champions League', 'Europa Conference League',
                                   'Copa Libertadores', 'Copa Sudamericana'}

    @staticmethod
    def _duplicated_teams(matches: pd.DataFrame) -> pd.DataFrame:
        """Find teams from different countries with the same names and rename them."""
        # find teams were participated in different countries
        teams_countries = (matches
                           .loc[matches['tournament_type'].isin({1, 2})]
                           .groupby(["home_team"])
                           ["country"]
                           .nunique()
                           .reset_index())

        # for detection teams with the same names
        teams_countries = set(teams_countries.loc[teams_countries["country"] > 1, 'home_team'])

        duplicated_teams = (matches
                            .loc[matches['home_team'].isin(teams_countries) & matches['tournament_type'].isin({1, 2})]
                            .drop_duplicates(['home_team', 'country'])
                            .loc[:, ['home_team', 'country']]
                            .sort_values(['home_team']))

        duplicated_teams_for_rename = duplicated_teams.loc[duplicated_teams['home_team'] == duplicated_teams['home_team'].shift(-1)]

        duplicated_teams_for_rename = list(zip(duplicated_teams_for_rename['home_team'],
                                               duplicated_teams_for_rename['country']))

        for team, country in duplicated_teams_for_rename:
            for location_team in ['home_team', 'away_team']:
                is_duplicate_team = ((matches[location_team] == team) & (matches['country'] == country))

                matches[location_team] = np.where(is_duplicate_team,
                                                  (matches[location_team] + ' ' + matches['country']),
                                                  matches[location_team])

        return matches

    def _rename_teams(self, matches: pd.DataFrame) -> pd.DataFrame:
        """"""

        international = matches.loc[matches['country'].isin(self.international_cups)]

        # country name in brackets
        bracket_teams = (set(international.loc[(matches['home_team'].map(lambda x: 1 if '(' in x else 0) == 1), 'home_team'])
                         .union(set(international.loc[(matches['away_team'].map(lambda x: 1 if '(' in x else 0) == 1), 'away_team'])))

        renaming_teams = {}
        for team in bracket_teams:
            renaming_teams[team] = team.split('(')[0].strip()

        matches['home_team'] = matches['home_team'].map(lambda x: renaming_teams[x] if x in renaming_teams else x)
        matches['away_team'] = matches['away_team'].map(lambda x: renaming_teams[x] if x in renaming_teams else x)

        matches = self._duplicated_teams(matches)

        matches['home_team'] = matches['home_team'].map(lambda x: x.replace("'", "").strip())
        matches['away_team'] = matches['away_team'].map(lambda x: x.replace("'", "").strip())

        return matches

    def draw_probability(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Use catboost model which calculate draw probability.
        Use this probability as a parameter in glicko model."""
        if self.is_actual_draw_predictions:
            features = TrainCreator().for_predictions(matches)
            OutcomesCatBoost().predict(features)

        draw_predictions = joblib.load(Config().project_path + Config().outcomes_paths['predictions'])

        draw_predictions = dict(zip(draw_predictions['match_id'], draw_predictions['draw']))

        matches['draw_probability'] = matches['match_id'].map(draw_predictions)

        # we can not apply skellam model, just use mean as draw probability
        mean_draw = matches.loc[matches['outcome'] == 'D'].shape[0] / matches.shape[0]

        matches['draw_probability'] = matches['draw_probability'].fillna(mean_draw)

        return matches

    @staticmethod
    def _remove_matches_with_unknown_team(matches: pd.DataFrame) -> pd.DataFrame:
        """Remove matches between teams from leagues we don't know anything about for preventing overfitting.
        For example, match of national cup with team from third league."""
        for season in matches['season'].unique():
            no_cups = (matches.loc[matches['tournament_type'].isin({1, 2})
                                   & (matches['season'] == season), ['home_team', 'away_team']])

            known_teams = set(no_cups['home_team']).union(no_cups['away_team'])

            is_both_teams_known = ((matches['season'] == season)
                                   & matches['home_team'].isin(known_teams)
                                   & matches['away_team'].isin(known_teams))

            matches = matches.loc[(matches['season'] != season) | is_both_teams_known]

        return matches

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

        matches['outcome'] = np.where((matches['tournament_type'] == 4) & (matches['notes'] == 'Pen'),
                                      'D',
                                      matches['outcome'])

        return matches

    def preprocessing(self, matches: pd.DataFrame = None) -> pd.DataFrame:
        """"""
        if matches is None:
            matches = pd.read_csv(Config().project_path + Config().matches_path)

        matches = matches.reset_index()

        matches['date'] = pd.to_datetime(matches['date'].str.replace('29.02', '28.02'), format='%d.%m.%Y', dayfirst=True)

        matches['season'] = matches['season'].map(lambda x: x.split('-')[0]).to_numpy('int')

        matches = (matches
                   .loc[matches['season'].between(self.min_season, self.max_season)]
                   .sort_values(['date'])
                   .reset_index(drop=True))

        # drop future matches: need only ratings, not further predictions
        matches = matches.loc[matches['home_score'] != '-']

        matches = matches.loc[~matches['notes'].isin({"Awrd"})]

        matches["tournament"] = np.where(matches["league"].isin(self.international_cups),
                                         matches["league"],
                                         matches["country"] + '. ' + matches["tournament_type"].str.title())

        matches['tournament_type'] = matches['tournament_type'].map({'first': 1,
                                                                     'second': 2,
                                                                     'cups': 3,
                                                                     'super_cups': 4})

        matches = self._rename_teams(matches)

        # outcome
        matches[['home_score', 'away_score']] = matches[['home_score', 'away_score']].to_numpy('int')

        conditions = [(matches['home_score'] > matches['away_score']),
                      (matches['home_score'] == matches['away_score']),
                      (matches['home_score'] < matches['away_score'])]

        outcomes = ['H', 'D', 'A']
        matches['outcome'] = np.select(conditions, outcomes)

        # pandemic
        is_pandemic = (matches['date'] > '2020-03-03') & (matches['date'] < '2021-06-06')
        matches['is_pandemic'] = np.where(is_pandemic, 1, 0)

        matches = self._remove_matches_with_unknown_team(matches)
        matches = self._get_finals(matches)

        matches = (matches
                   .drop(columns=['notes'])
                   .sort_values(['date'])
                   .rename(columns={'index': 'match_id'}))

        matches = self.draw_probability(matches)

        return matches

    def test_data(self, matches: pd.DataFrame):
        """Use this function for detection mistakes in the data."""
        # short season tournaments
        count_matches = (matches
                         .loc[matches['tournament_type'].isin({1, 2}) & (matches['season'] != matches['season'].max())]
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

        # teams in eurocups without nation league
        international_teams = (set(matches.loc[matches['country'].isin(self.international_cups), 'home_team'])
                               .union(set(matches.loc[matches['country'].isin(self.international_cups), 'away_team'])))

        national_teams = set(matches.loc[matches['tournament_type'].isin({1, 2}), 'home_team'])

        international_teams_without_national = sorted([t for t in international_teams if t not in national_teams])

        # find teams were participated in different countries
        teams_countries = (matches
                           .loc[matches['tournament_type'].isin({1, 2})]
                           .groupby(["home_team"])
                           ["country"]
                           .nunique()
                           .reset_index())

        # for detection teams with the same names
        teams_countries = set(teams_countries.loc[teams_countries["country"] > 1, 'home_team'])

        duplicated_teams = (matches
                            .loc[matches['home_team'].isin(teams_countries) & matches['tournament_type'].isin({1, 2})]
                            .drop_duplicates(['home_team', 'country'])
                            .loc[:, ['home_team', 'country']]
                            .sort_values(['home_team']))

        return count_matches, count_seasons, count_tournaments, international_teams_without_national, duplicated_teams
