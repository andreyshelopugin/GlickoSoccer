import numpy as np
import pandas as pd

from config import Config
from soccer.outcomes_features import TrainCreator
from soccer.outcomes_lgbm import OutcomesLGBM


class DataPreprocessor(object):

    def __init__(self, min_season=2010, max_season=2022, is_actual_draw_predictions=False, is_boosting_train=True):
        self.min_season = min_season
        self.max_season = max_season
        self.is_actual_draw_predictions = is_actual_draw_predictions
        self.is_boosting_train = is_boosting_train
        self.international_cups = {'Europa League', 'Champions League', 'Europa Conference League',
                                   'Copa Libertadores', 'Copa Sudamericana'}

        self.small_tournaments = {'Faroe Islands. Second',
                                  'Azerbaijan. Second'}

    @staticmethod
    def _duplicated_teams(matches: pd.DataFrame) -> pd.DataFrame:
        """Finds teams from different countries with the same names and renames them."""
        # find teams from different countries
        teams_countries = (matches
                           .loc[matches['tournament_type'].isin({1, 2})]
                           .groupby(["home_team"])
                           ["country"]
                           .nunique()
                           .reset_index())

        # teams with the same names
        teams_countries = set(teams_countries.loc[teams_countries["country"] > 1, 'home_team'])

        duplicated_teams = (matches
                            .loc[matches['home_team'].isin(teams_countries) & matches['tournament_type'].isin({1, 2})]
                            .drop_duplicates(['home_team', 'country'])
                            .loc[:, ['home_team', 'country']]
                            .sort_values(['home_team']))

        rename_international = {'Aris (Gre)': 'Aris Greece',
                                'Benfica (Por)': 'Benfica Portugal',
                                'Bohemians (Irl)': 'Bohemians Ireland',
                                'Drita (Kos)': 'Drita Kosovo',
                                'Everton (Eng)': 'Everton England',
                                'Everton (Chi)': 'Everton Chile',
                                'Flamurtari (Alb)': 'Flamurtari Albania',
                                'Fortaleza (Bra)': 'Fortaleza Brazil',
                                'Guarani (Par)': 'Guarani Paraguay',
                                'Nacional (Por)': 'Nacional Portugal',
                                'Nacional (Uru)': 'Nacional Uruguay',
                                'Noah (Arm)': 'Noah Armenia',
                                'Rangers (Sco)': 'Rangers Scotland',
                                'River Plate (Par)': 'River Plate Paraguay',
                                'River Plate (Uru)': 'River Plate Uruguay',
                                'River Plate (Arg)': 'River Plate Argentina',
                                'Rudar (Mne)': 'Rudar Montenegro',
                                'Rudar (Slo)': 'Rudar Slovenia',
                                'San Lorenzo (Arg)': 'San Lorenzo Argentina',
                                'Santa Cruz (Bra)': 'Santa Cruz Brazil',
                                'Sloboda (Bih)': 'Sloboda Bosnia And Herzegovina',
                                'Sport Boys (Bol)': 'Sport Boys Bolivia',
                                'U. Catolica (Ecu)': 'U. Catolica Ecuador',
                                'U. Catolica (Chi)': 'U. Catolica Chile',
                                }

        matches['home_team'] = matches['home_team'].map(lambda x: rename_international[x] if x in rename_international else x)
        matches['away_team'] = matches['away_team'].map(lambda x: rename_international[x] if x in rename_international else x)

        for row in duplicated_teams.itertuples():
            team, country = row.home_team, row.country
            for location_team in ['home_team', 'away_team']:
                is_duplicate_team = ((matches[location_team] == team) & (matches['country'] == country))

                matches[location_team] = np.where(is_duplicate_team,
                                                  (matches[location_team] + ' ' + matches['country']),
                                                  matches[location_team])

        # fix the case when two River Plates played in one international cup during one season
        matches['home_team'] = np.where(((matches['home_team'] == 'River Plate')
                                         & (matches['away_team'].isin({'Aguilas', 'Blooming'}))
                                         & (matches['season'] == 2013)),
                                        'River Plate Uruguay',
                                        matches['home_team'])

        matches['away_team'] = np.where(((matches['away_team'] == 'River Plate')
                                         & (matches['home_team'].isin({'Aguilas', 'Blooming'}))
                                         & (matches['season'] == 2013)),
                                        'River Plate Uruguay',
                                        matches['away_team'])

        rename = [
            (2010, 'Aris', 'Aris Greece'),
            (2010, 'Benfica', 'Benfica Portugal'),
            (2011, 'Benfica', 'Benfica Portugal'),
            (2012, 'Benfica', 'Benfica Portugal'),
            (2013, 'Benfica', 'Benfica Portugal'),
            (2014, 'Benfica', 'Benfica Portugal'),
            (2015, 'Benfica', 'Benfica Portugal'),
            (2016, 'Benfica', 'Benfica Portugal'),
            (2017, 'Benfica', 'Benfica Portugal'),
            (2018, 'Benfica', 'Benfica Portugal'),
            (2019, 'Benfica', 'Benfica Portugal'),
            (2020, 'Benfica', 'Benfica Portugal'),
            (2021, 'Benfica', 'Benfica Portugal'),
            (2010, 'Bohemians', 'Bohemians Ireland'),
            (2014, 'Everton', 'Everton England'),
            (2017, 'Everton', 'Everton England'),
            (2010, 'Guarani', 'Guarani Paraguay'),
            (2011, 'Guarani', 'Guarani Paraguay'),
            (2012, 'Guarani', 'Guarani Paraguay'),
            (2013, 'Guarani', 'Guarani Paraguay'),
            (2014, 'Guarani', 'Guarani Paraguay'),
            (2015, 'Guarani', 'Guarani Paraguay'),
            (2010, 'River Plate', 'River Plate Uruguay'),
            (2015, 'River Plate', 'River Plate Argentina'),
            (2013, 'River Plate', 'River Plate Argentina'),
            (2010, 'Nacional', 'Nacional Uruguay'),
            (2011, 'Nacional', 'Nacional Uruguay'),
            (2012, 'Nacional', 'Nacional Uruguay'),
            (2013, 'Nacional', 'Nacional Uruguay'),
            (2014, 'Nacional', 'Nacional Uruguay'),
            (2015, 'Nacional', 'Nacional Uruguay'),
            (2013, 'Portuguesa', 'Portuguesa Brazil'),
            (2010, 'Rangers', 'Rangers Scotland'),
            (2018, 'Rangers', 'Rangers Scotland'),
            (2019, 'Rangers', 'Rangers Scotland'),
            (2020, 'Rangers', 'Rangers Scotland'),
            (2021, 'Rangers', 'Rangers Scotland'),
            (2010, 'Rudar', 'Rudar Montenegro'),
            (2013, 'San Lorenzo', 'San Lorenzo Argentina'),
            (2014, 'San Lorenzo', 'San Lorenzo Argentina'),
            (2015, 'San Lorenzo', 'San Lorenzo Argentina'),
            (2010, 'U. Catolica', 'U. Catolica Chile'),
            (2011, 'U. Catolica', 'U. Catolica Chile'),
            (2012, 'U. Catolica', 'U. Catolica Chile'),
            (2013, 'U. Catolica', 'U. Catolica Chile'),
        ]

        for row in rename:
            season, team, new_team = row[0], row[1], row[2]
            for location_team in ['home_team', 'away_team']:
                is_duplicate_team = ((matches[location_team] == team) & (matches['season'] == season))

                matches[location_team] = np.where(is_duplicate_team,
                                                  new_team,
                                                  matches[location_team])

        return matches

    def _rename_teams(self, matches: pd.DataFrame) -> pd.DataFrame:
        """"""

        matches = self._duplicated_teams(matches)

        international = matches.loc[matches['country'].isin(self.international_cups)]

        # removes countries names in brackets
        bracket_teams = (set(international.loc[(matches['home_team'].map(lambda x:
                                                                         1 if '(' in x else 0) == 1), 'home_team'])
                         .union(set(international.loc[(matches['away_team'].map(lambda x:
                                                                                1 if '(' in x else 0) == 1), 'away_team'])))

        renaming_teams = {}
        for team in bracket_teams:
            renaming_teams[team] = team.split('(')[0].strip()

        matches['home_team'] = matches['home_team'].map(lambda x: renaming_teams[x] if x in renaming_teams else x)
        matches['away_team'] = matches['away_team'].map(lambda x: renaming_teams[x] if x in renaming_teams else x)

        matches['home_team'] = matches['home_team'].map(lambda x: x.replace("'", "").strip())
        matches['away_team'] = matches['away_team'].map(lambda x: x.replace("'", "").strip())

        return matches

    @staticmethod
    def _remove_matches_with_unknown_team(matches: pd.DataFrame) -> pd.DataFrame:
        """Remove matches featuring teams from unknown leagues to prevent overfitting.
        For instance, matches from the national cup involving third league teams will be removed."""
        for season in matches['season'].unique():
            no_cups = (matches.loc[matches['tournament_type'].isin({1, 2})
                                   & (matches['season'] == season), ['home_team', 'away_team']])

            known_teams = set(no_cups['home_team']).union(no_cups['away_team'])

            is_both_teams_known = ((matches['season'] == season)
                                   & matches['home_team'].isin(known_teams)
                                   & matches['away_team'].isin(known_teams))

            matches = matches.loc[(matches['season'] != season) | is_both_teams_known]

        return matches

    def draw_probability(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Utilizes a gradient boosting model that calculates draw probability and
        uses this probability as a parameter in the modified Glicko model."""

        # update match outcomes predictions
        if self.is_actual_draw_predictions:
            features = TrainCreator().for_predictions(matches)
            predictions = OutcomesLGBM().predict(features)
            draw_predictions = predictions.loc[:, ['match_id', 'draw']]
            draw_predictions = dict(zip(draw_predictions['match_id'], draw_predictions['draw']))
        else:
            draw_predictions = pd.read_feather(Config().outcomes_paths['lgbm_predictions'], columns=['match_id', 'draw'])
            draw_predictions = dict(zip(draw_predictions['match_id'], draw_predictions['draw']))

        matches['draw_probability'] = matches['match_id'].map(draw_predictions)

        mean_draw = matches.loc[matches['outcome'] == 'D'].shape[0] / matches.shape[0]
        matches['draw_probability'] = matches['draw_probability'].fillna(mean_draw)

        return matches

    @staticmethod
    def _get_finals(matches: pd.DataFrame, is_end_of_season=True) -> pd.DataFrame:
        """Get final matches for the detection of neutral field games."""

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

        # final matches that finished with penalty shootouts are considered as ties.
        matches['outcome'] = np.where((matches['tournament_type'] == 4) & (matches['notes'] == 'Pen'),
                                      'D',
                                      matches['outcome'])

        return matches

    def preprocessing(self, matches: pd.DataFrame = None) -> pd.DataFrame:
        """"""
        if matches is None:
            matches = pd.read_csv(Config().matches_path)

        # create column 'index'
        matches = matches.reset_index()

        # drop future matches: we need only ratings, not further predictions
        matches = matches.loc[matches['home_score'] != '-']

        # drop forfeiting wins.
        matches = matches.loc[~matches['notes'].isin({"Awrd"})]

        matches['date'] = pd.to_datetime(matches['date'].str.replace('29.02', '28.02'), format='%d.%m.%Y', dayfirst=True)

        # 2023-2024 -> 2023
        matches['season'] = matches['season'].map(lambda x: x.split('-')[0]).to_numpy('int')

        matches = (matches
                   .loc[matches['season'].between(self.min_season, self.max_season)]
                   .sort_values(['date'])
                   .reset_index(drop=True))

        for international_cup in self.international_cups:
            matches['league'] = np.where((matches['tournament_type'] == 'cups')
                                         & matches['league'].str.contains(international_cup),
                                         international_cup,
                                         matches['league'])

        matches["tournament"] = np.where(matches["league"].isin(self.international_cups),
                                         matches["league"],
                                         matches["country"] + '. ' + matches["tournament_type"].str.title())

        matches = matches.loc[~matches['tournament'].isin(self.small_tournaments)]

        matches['tournament_type'] = matches['tournament_type'].map({'first': 1,
                                                                     'second': 2,
                                                                     'cups': 3,
                                                                     'super_cups': 4})

        matches = self._rename_teams(matches)
        matches = self._remove_matches_with_unknown_team(matches)

        # outcome
        matches[['home_score', 'away_score']] = matches[['home_score', 'away_score']].to_numpy('int')

        conditions = [(matches['home_score'] > matches['away_score']),
                      (matches['home_score'] == matches['away_score']),
                      (matches['home_score'] < matches['away_score'])]

        outcomes = ['H', 'D', 'A']
        matches['outcome'] = np.select(conditions, outcomes)

        matches = self._get_finals(matches)

        # matches played during the pandemic
        matches['is_pandemic'] = (matches['date'] > '2020-03-03') & (matches['date'] < '2021-06-06')

        matches = (matches
                   .drop(columns='notes')
                   .sort_values('date')
                   .rename(columns={'index': 'match_id'}))

        if not self.is_boosting_train:
            matches = self.draw_probability(matches)

        return matches

    def test_data(self, matches: pd.DataFrame):
        """Use this function for testing data"""
        # short season tournaments
        count_matches = (matches
                         .loc[matches['tournament_type'].isin({1, 2}) & (matches['season'] != matches['season'].max())]
                         .groupby(['country', 'tournament_type', 'season'])
                         ['match_id']
                         .count()
                         .reset_index()
                         .rename(columns={'match_id': 'number_matches'}))

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

        # teams in eurocups without league
        international_teams = (set(matches.loc[matches['country'].isin(self.international_cups), 'home_team'])
                               .union(set(matches.loc[matches['country'].isin(self.international_cups), 'away_team'])))

        national_teams = set(matches.loc[matches['tournament_type'].isin({1, 2}), 'home_team'])

        international_teams_without_national = sorted([t for t in international_teams if t not in national_teams])

        # find teams that participated in different countries
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
