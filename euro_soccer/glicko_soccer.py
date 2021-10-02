from itertools import product
from typing import Tuple, Dict

import joblib
import numpy as np
import pandas as pd

from glicko2 import Glicko2, Rating
from utils.metrics import three_outcomes_log_loss


class GlickoSoccer(object):

    def __init__(self, is_draw_mode=True, init_mu=1500, init_rd=120, update_rd=30, lift_update_mu=0, lift_update_rd=30,
                 home_advantage=30, draw_inclination=-0.15, cup_penalty=0, new_team_update_mu=-20):
        self.is_draw_mode = is_draw_mode
        self.init_mu = init_mu
        self.init_rd = init_rd
        self.lift_update_mu = lift_update_mu
        self.update_rd = update_rd
        self.lift_update_rd = lift_update_rd
        self.home_advantage = home_advantage
        self.draw_inclination = draw_inclination
        self.cup_penalty = cup_penalty
        self.new_team_update_mu = new_team_update_mu
        self.euro_cups = {'Champions League', 'Europa League'}  # !!!!

    def preprocessing(self, results: pd.DataFrame) -> pd.DataFrame:
        """"""
        rename_teams = {

        }

        results['home_team'] = results['home_team'].map(lambda x: x.replace("'", "").strip())
        results['away_team'] = results['away_team'].map(lambda x: x.replace("'", "").strip())

        results['home_team'] = results['home_team'].map(lambda x: rename_teams[x] if x in rename_teams else x)
        results['away_team'] = results['away_team'].map(lambda x: rename_teams[x] if x in rename_teams else x)

        # results = results.loc[results['tournament'].isin(self.first_leagues)
        #                       | results['tournament'].isin(self.second_leagues)
        #                       | results['tournament'].isin(self.cups)
        #                       | results['tournament'].isin(self.euro_cups)]

        conditions = [(results['home_score'] > results['away_score']),
                      (results['home_score'] == results['away_score']),
                      (results['home_score'] < results['away_score'])]

        outcomes = ['H', 'D', 'A']
        results['outcome'] = np.select(conditions, outcomes)

        # min_season = results['season'].min()
        # results['season'] = np.where(results['tournament'].isin(self.playoff), results['season'] - 1, results['season'])
        # results = results.loc[results['season'] >= min_season]

        results = results.drop(columns=['home_score', 'away_score']).sort_values(['date'])
        results = self._remove_matches_with_unknown_team(results)

        return results

    def _team_leagues(self, results: pd.DataFrame, season: int) -> dict:
        """"""
        team_leagues = (results
                        .loc[(results['tournament_type'] != 3)
                             & (results['season'] == season), ['home_team', 'tournament']]
                        .drop_duplicates(['home_team']))

        team_leagues = dict(zip(team_leagues['home_team'], team_leagues['tournament']))

        away_team_leagues = (results
                             .loc[(results['tournament_type'] != 3)
                                  & (results['season'] == season), ['away_team', 'tournament']]
                             .drop_duplicates(['away_team']))

        team_leagues.update(dict(zip(away_team_leagues['away_team'], away_team_leagues['tournament'])))

        return team_leagues

    def _remove_matches_with_unknown_team(self, results: pd.DataFrame) -> pd.DataFrame:
        """Remove matches between teams from leagues we dont know anything about."""
        for season in results['season'].unique():
            team_leagues = self._team_leagues(results, season)

            known_teams = team_leagues.keys()

            is_both_teams_known = ((results['season'] == season)
                                   & results['home_team'].isin(known_teams)
                                   & results['away_team'].isin(known_teams))

            results = results.loc[(results['season'] != season) | is_both_teams_known]

        return results

    def _team_international_cups(self, results: pd.DataFrame, season: int) -> set:
        """
        """

        international_cups = results.loc[results['tournament'].isin(self.euro_cups) & (results['season'] == season)]

        international_teams = set(international_cups['home_team']).union(set(international_cups['away_team']))

        return international_teams

    def _league_params_initialization(self, results: pd.DataFrame) -> dict:
        """"""
        leagues = results.loc[results['tournament_type'] != 3, 'tournament'].unique()
        league_params = dict()
        for league in leagues:
            league_params[league] = {'init_mu': self.init_mu,
                                     'init_rd': self.init_rd,
                                     'update_rd': self.update_rd,
                                     'lift_update_mu': self.lift_update_mu,
                                     'lift_update_rd': self.lift_update_rd,
                                     'home_advantage': self.home_advantage,
                                     'draw_inclination': self.draw_inclination,  # ????
                                     'cup_penalty': self.cup_penalty,
                                     'new_team_update_mu': self.new_team_update_mu,
                                     }

        return league_params

    def _team_params(self, results: pd.DataFrame, season: int, league_params: dict) -> dict:
        """
            For each team get league params.
        """

        team_leagues = self._team_leagues(results, season)

        team_params = {team: league_params[league] for team, league in team_leagues.items()}

        return team_params

    def _rating_initialization(self, results: pd.DataFrame, league_params: dict) -> dict:
        """"""
        seasons = sorted(results['season'].unique())

        ratings = dict()
        for season in seasons:

            team_params = self._team_params(results, season, league_params)

            for team, params in team_params.items():
                if team not in ratings:
                    ratings[team] = Rating(mu=params['init_mu'], rd=params['init_rd'])

        return ratings

    def _update_ratings_indexes(self, results: pd.DataFrame) -> Tuple[dict, dict, dict]:
        """"""
        no_cups = results.loc[results['tournament_type'] != 3]

        no_cups = pd.concat([no_cups.loc[:, ['date', 'index', 'home_team', 'season', 'tournament']]
                            .rename(columns={'home_team': 'team'}),
                             no_cups.loc[:, ['date', 'index', 'away_team', 'season', 'tournament']]
                            .rename(columns={'away_team': 'team'})])

        no_cups = (no_cups
                   .sort_values(['team', 'date'])
                   .drop_duplicates(['team', 'season'], keep='first'))

        # teams missed previous season
        # remove first seasons too
        missed_previous_season = (no_cups
                                  .loc[(no_cups['season'] != no_cups['season'].shift() + 1)
                                       & (no_cups['team'] == no_cups['team'].shift())]
                                  .groupby(['team'])
                                  ['index']
                                  .apply(set)
                                  .to_dict())

        # teams changed league
        changed_league = (no_cups
                          .loc[(no_cups['season'] == no_cups['season'].shift() + 1)
                               & (no_cups['tournament'] != no_cups['tournament'].shift())
                               & (no_cups['team'] == no_cups['team'].shift())]
                          .groupby(['team'])
                          ['index']
                          .apply(set)
                          .to_dict())

        # the same league
        same_league = (no_cups
                       .loc[(no_cups['season'] == no_cups['season'].shift() + 1)
                            & (no_cups['tournament'] == no_cups['tournament'].shift())
                            & (no_cups['team'] == no_cups['team'].shift())]
                       .groupby(['team'])
                       ['index']
                       .apply(set)
                       .to_dict())

        return missed_previous_season, changed_league, same_league

    def _season_update_rating(self, ratings: dict, home_team: str, away_team: str, index: int, team_params: dict,
                              missed_previous_season: Dict[str, set], changed_league: Dict[str, set],
                              same_league: Dict[str, set]) -> dict:
        """"""
        for team in [home_team, away_team]:
            params = team_params[team]
            if team in missed_previous_season:
                if index in missed_previous_season[team]:
                    ratings[team] = Rating(mu=params['init_mu'] + params['new_team_update_mu'], rd=params['init_rd'])

            elif team in changed_league:
                if index in changed_league[team]:
                    ratings[team] = Rating(mu=ratings[team].mu + params['lift_update_mu'],
                                           rd=ratings[team].rd + params['lift_update_rd'])

            elif team in same_league:
                if index in same_league[team]:
                    ratings[team] = Rating(mu=ratings[team].mu, rd=ratings[team].rd + params['update_rd'])

        return ratings

    def fit_params2(self, results: pd.DataFrame, number_iterations: int, is_params_initialization: True):
        """"""

        first_league_seasons = self._first_league_seasons(results)
        team_leagues = {season: self._team_leagues(results, season) for season in results['season'].unique()}

        eurocups_teams = dict()
        for season in results['season'].unique():
            season_team_leagues = team_leagues[season]

            teams = self._team_international_cups(results, season)
            teams = [team for team in teams.keys() if team in season_team_leagues]
            eurocups_teams[season] = [team for team in teams if season_team_leagues[team] in self.top_leagues]

        if is_params_initialization:
            league_params = self._league_params_initialization()
        else:
            league_params = joblib.load('data/league_params.pkl')

        for i in range(number_iterations):

            for league, params in league_params.items():

                init_mu = params['init_mu']
                init_rd = params['init_rd']
                update_rd = params['update_rd']
                lift_update_mu = params['lift_update_mu']
                lift_update_rd = params['lift_update_rd']
                home_advantage = params['home_advantage']
                draw_inclination = params['draw_inclination']
                cup_penalty = params['cup_penalty']
                new_team_update_mu = params['new_team_update_mu']

                init_mu_list = [init_mu - 2, init_mu, init_mu + 2]
                init_rd_list = [init_rd - 2, init_rd, init_rd + 2]
                update_rd_list = [update_rd - 2, update_rd, update_rd + 2]
                lift_update_mu_list = [lift_update_mu - 2, lift_update_mu, lift_update_mu + 2]
                lift_update_rd_list = [lift_update_rd - 2, lift_update_rd, lift_update_rd + 2]
                home_advantage_list = [home_advantage - 1, home_advantage, home_advantage + 1]
                draw_inclination_list = [draw_inclination - 0.01, draw_inclination, draw_inclination + 0.01]
                cup_penalty_list = [cup_penalty - 2, cup_penalty, cup_penalty + 2]
                new_team_update_mu_list = [new_team_update_mu - 2, new_team_update_mu, new_team_update_mu + 2]

                init_rd_list = [x for x in init_rd_list if x >= 100]
                update_rd_list = [x for x in update_rd_list if x >= 20]
                lift_update_rd_list = [x for x in lift_update_rd_list if x >= 20]

                if league in self.top_leagues:
                    new_team_update_mu_list = [0]

                else:
                    cup_penalty_list = [0]

                    if league not in self.second_leagues:
                        lift_update_mu_list = [lift_update_mu]
                        lift_update_rd_list = [lift_update_rd]

                if league in self.euro_cups:
                    update_rd_list = [update_rd]
                    new_team_update_mu_list = [0]
                    lift_update_mu_list = [lift_update_mu]
                    lift_update_rd_list = [lift_update_rd]

                parameters_list = list(product(init_mu_list,
                                               init_rd_list,
                                               update_rd_list,
                                               lift_update_mu_list,
                                               lift_update_rd_list,
                                               home_advantage_list,
                                               draw_inclination_list,
                                               cup_penalty_list,
                                               new_team_update_mu_list))

                parameters_loss = {parameters: 0 for parameters in parameters_list}
                for parameters in parameters_list:
                    league_params[league] = {'init_mu': parameters[0],
                                             'init_rd': parameters[1],
                                             'update_rd': parameters[2],
                                             'lift_update_mu': parameters[3],
                                             'lift_update_rd': parameters[4],
                                             'home_advantage': parameters[5],
                                             'draw_inclination': parameters[6],
                                             'cup_penalty': parameters[7],
                                             'new_team_update_mu': parameters[8],
                                             }

                    parameters_loss[parameters] = self.calculate_loss(results,
                                                                      league_params,
                                                                      first_league_seasons,
                                                                      team_leagues,
                                                                      eurocups_teams)

                optimal_parameters = min(parameters_loss, key=parameters_loss.get)

                optimal_parameters_dict = {'init_mu': optimal_parameters[0],
                                           'init_rd': optimal_parameters[1],
                                           'update_rd': optimal_parameters[2],
                                           'lift_update_mu': optimal_parameters[3],
                                           'lift_update_rd': optimal_parameters[4],
                                           'home_advantage': optimal_parameters[5],
                                           'draw_inclination': optimal_parameters[6],
                                           'cup_penalty': optimal_parameters[7],
                                           'new_team_update_mu': optimal_parameters[8],
                                           }

                league_params[league] = optimal_parameters_dict

                print(league, parameters_loss[optimal_parameters])
                print(optimal_parameters_dict)

                joblib.dump(league_params, 'data/league_params.pkl')

        return league_params

    def rate_teams(self, results: pd.DataFrame, league_params: dict) -> dict:
        """"""
        ratings = self._rating_initialization(results, league_params)

        team_leagues = {season: self._team_leagues(results, season) for season in results['season'].unique()}
        team_params = {season: self._team_params(results, season, league_params) for season in
                       results['season'].unique()}

        missed_previous_season, changed_league, same_league = self._update_ratings_indexes(results)

        eurocups_teams = dict()
        for season in results['season'].unique():
            eurocups_teams[season] = self._team_international_cups(results, season)

        for row in results.itertuples():

            season_team_params = team_params[row.season]

            ratings = self._season_update_rating(ratings, row.home_team, row.away_team, row.index, season_team_params,
                                                 missed_previous_season, changed_league, same_league)

            home_params = season_team_params[row.home_team]
            away_params = season_team_params[row.away_team]

            # for teams from different championships calculate the average of parameters
            home_advantage = home_params['home_advantage']
            draw_inclination = (home_params['draw_inclination'] + away_params['draw_inclination']) / 2

            glicko = Glicko2(draw_inclination=draw_inclination)

            if ((row.tournament_type == 3)
                    and (row.tournament not in self.euro_cups)
                    and (team_leagues[row.season][row.home_team] != team_leagues[row.season][row.away_team])):

                if row.home_team in eurocups_teams[row.season]:
                    home_advantage -= home_params['cup_penalty']
                elif row.away_team in eurocups_teams[row.season]:
                    home_advantage += away_params['cup_penalty']

            # get current team ratings
            home_rating, away_rating = ratings[row.home_team], ratings[row.away_team]

            # update team ratings
            ratings[row.home_team], ratings[row.away_team] = glicko.rate(home_rating, away_rating, home_advantage,
                                                                         row.outcome,
                                                                         2.5)

        return ratings

    def calculate_loss(self, results: pd.DataFrame, league_params: dict) -> float:
        """"""
        ratings = self._rating_initialization(results, league_params)

        team_leagues = {season: self._team_leagues(results, season) for season in results['season'].unique()}
        team_params = {season: self._team_params(results, season, league_params) for season in
                       results['season'].unique()}

        missed_previous_season, changed_league, same_league = self._update_ratings_indexes(results)

        eurocups_teams = dict()
        for season in results['season'].unique():
            eurocups_teams[season] = self._team_international_cups(results, season)

        log_loss_value = 0
        for row in results.itertuples():

            season_team_params = team_params[row.season]

            ratings = self._season_update_rating(ratings, row.home_team, row.away_team, row.index, season_team_params,
                                                 missed_previous_season, changed_league, same_league)

            home_params = season_team_params[row.home_team]
            away_params = season_team_params[row.away_team]

            # for teams from different championships calculate the average of parameters
            home_advantage = home_params['home_advantage']
            draw_inclination = (home_params['draw_inclination'] + away_params['draw_inclination']) / 2

            glicko = Glicko2(draw_inclination=draw_inclination)

            if ((row.tournament_type == 3)
                    and (row.tournament not in self.euro_cups)
                    and (team_leagues[row.season][row.home_team] != team_leagues[row.season][row.away_team])):

                if row.home_team in eurocups_teams[row.season]:
                    home_advantage -= home_params['cup_penalty']
                elif row.away_team in eurocups_teams[row.season]:
                    home_advantage += away_params['cup_penalty']

            # get current team ratings
            home_rating, away_rating = ratings[row.home_team], ratings[row.away_team]

            # calculate outcome probabilities
            win_probability, tie_probability, loss_probability = glicko.probabilities(home_rating, away_rating,
                                                                                      home_advantage, 2.5)

            log_loss_value += three_outcomes_log_loss(row.outcome, win_probability, tie_probability,
                                                      loss_probability)

            # update team ratings
            ratings[row.home_team], ratings[row.away_team] = glicko.rate(home_rating, away_rating, home_advantage,
                                                                         row.outcome,
                                                                         2.5)

        return log_loss_value

    def fit_params(self, results: pd.DataFrame, number_iterations: int, is_params_initialization: True):
        """"""

        eurocups_teams = dict()
        for season in results['season'].unique():
            eurocups_teams[season] = self._team_international_cups(results, season)

        if is_params_initialization:
            league_params = self._league_params_initialization(results)
        else:
            league_params = joblib.load('data/league_params.pkl')

        first_leagues = set(results.loc[results['tournament_type'] == 1, 'tournament'])

        for i in range(number_iterations):
            print(i)

            for league, params in league_params.items():

                init_mu = params['init_mu']
                init_rd = params['init_rd']
                update_rd = params['update_rd']
                lift_update_mu = params['lift_update_mu']
                lift_update_rd = params['lift_update_rd']
                home_advantage = params['home_advantage']
                draw_inclination = params['draw_inclination']
                cup_penalty = params['cup_penalty']
                new_team_update_mu = params['new_team_update_mu']

                init_mu_list = [init_mu - 2, init_mu, init_mu + 2]
                init_rd_list = [init_rd - 2, init_rd, init_rd + 2]
                update_rd_list = [update_rd - 2, update_rd, update_rd + 2]
                lift_update_mu_list = [lift_update_mu - 2, lift_update_mu, lift_update_mu + 2]
                lift_update_rd_list = [lift_update_rd - 2, lift_update_rd, lift_update_rd + 2]
                home_advantage_list = [home_advantage - 1, home_advantage, home_advantage + 1]
                draw_inclination_list = [draw_inclination - 0.01, draw_inclination, draw_inclination + 0.01]
                cup_penalty_list = [cup_penalty - 2, cup_penalty, cup_penalty + 2]
                new_team_update_mu_list = [new_team_update_mu - 2, new_team_update_mu, new_team_update_mu + 2]

                init_rd_list = [x for x in init_rd_list if x >= 100]
                update_rd_list = [x for x in update_rd_list if x >= 20]
                lift_update_rd_list = [x for x in lift_update_rd_list if x >= 20]

                if league in first_leagues:
                    new_team_update_mu_list = [0]
                else:
                    cup_penalty_list = [0]

                parameters_list = list(product(init_mu_list,
                                               init_rd_list,
                                               update_rd_list,
                                               lift_update_mu_list,
                                               lift_update_rd_list,
                                               home_advantage_list,
                                               draw_inclination_list,
                                               cup_penalty_list,
                                               new_team_update_mu_list))

                parameters_loss = {parameters: 0 for parameters in parameters_list}
                for parameters in parameters_list:
                    league_params[league] = {'init_mu': parameters[0],
                                             'init_rd': parameters[1],
                                             'update_rd': parameters[2],
                                             'lift_update_mu': parameters[3],
                                             'lift_update_rd': parameters[4],
                                             'home_advantage': parameters[5],
                                             'draw_inclination': parameters[6],
                                             'cup_penalty': parameters[7],
                                             'new_team_update_mu': parameters[8],
                                             }

                    parameters_loss[parameters] = self.calculate_loss(results, league_params)

                optimal_parameters = min(parameters_loss, key=parameters_loss.get)

                optimal_parameters_dict = {'init_mu': optimal_parameters[0],
                                           'init_rd': optimal_parameters[1],
                                           'update_rd': optimal_parameters[2],
                                           'lift_update_mu': optimal_parameters[3],
                                           'lift_update_rd': optimal_parameters[4],
                                           'home_advantage': optimal_parameters[5],
                                           'draw_inclination': optimal_parameters[6],
                                           'cup_penalty': optimal_parameters[7],
                                           'new_team_update_mu': optimal_parameters[8],
                                           }

                league_params[league] = optimal_parameters_dict

                print(league, parameters_loss[optimal_parameters])
                print(optimal_parameters_dict)

                joblib.dump(league_params, 'data/league_params.pkl')

        return league_params

    def ratings_to_df(self, ratings: dict, results: pd.DataFrame) -> pd.DataFrame:
        """"""
        seasons = sorted(results['season'].unique())

        ratings = {team: rating.mu for team, rating in ratings.items()}

        max_seasons = (results
                       .sort_values(['tournament', 'season'], ascending=False)
                       .drop_duplicates(['tournament'], keep='first'))

        max_seasons = dict(zip(max_seasons['tournament'], max_seasons['season']))

        results['max_season'] = results['tournament'].map(max_seasons)

        results = results.loc[results['season'] == results['max_season']]

        team_leagues = self._team_leagues(results, min(seasons))
        for season in seasons:
            team_leagues.update(self._team_leagues(results, season))

        ratings_df = (pd.DataFrame
                      .from_dict(ratings, orient='index')
                      .reset_index()
                      .rename(columns={'index': 'team', 0: 'rating'})
                      .sort_values(['rating'], ascending=False)
                      .reset_index(drop=True)
                      .reset_index()
                      .rename(columns={'index': '#'}))

        ratings_df['#'] = (ratings_df['#'] + 1)

        ratings_df['league'] = ratings_df['team'].map(team_leagues)

        return ratings_df

    @staticmethod
    def league_ratings(ratings: pd.DataFrame, number_top_teams=50) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """"""
        leagues_ratings = (ratings
                           .groupby(['league'])
                           .apply(lambda x: x.nlargest(number_top_teams, 'rating'))
                           .reset_index(drop=True))

        leagues_ratings = (leagues_ratings
                           .groupby(['league'])
                           ['rating']
                           .mean()
                           .reset_index()
                           .sort_values(['rating'], ascending=False)
                           .reset_index(drop=True)
                           .reset_index()
                           .rename(columns={'index': '#'}))

        leagues_ratings['#'] = (leagues_ratings['#'] + 1)

        return leagues_ratings
