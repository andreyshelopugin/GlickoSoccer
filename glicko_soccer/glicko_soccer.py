from itertools import product
from typing import List, Tuple, Dict

import pandas as pd
from sklearn.metrics import log_loss
from tqdm import tqdm

from glicko2 import Glicko2, Rating


class GlickoSoccer(object):

    @staticmethod
    def rate_teams(matches: pd.DataFrame, ratings: Rating(), home_advantage: float, draw_inclination: float) -> Rating():
        """Calculate the ratings of teams using the history of matches"""

        glicko = Glicko2(draw_inclination=draw_inclination)

        for index, row in matches.iterrows():
            outcome = row['outcome']

            home_team, away_team = row['home_team'], row['away_team']
            home_rating, away_rating = ratings[home_team], ratings[away_team]

            if outcome == 'H':
                ratings[home_team], ratings[away_team] = glicko.rate(home_rating, away_rating, home_advantage, is_draw=False)

            elif outcome == 'D':
                ratings[home_team], ratings[away_team] = glicko.rate(home_rating, away_rating, home_advantage, is_draw=True)

            else:
                ratings[away_team], ratings[home_team] = glicko.rate(away_rating, home_rating, -home_advantage, is_draw=False)

        return ratings

    def get_ratings(self, results: pd.DataFrame, schedule: pd.DataFrame, current_season: int, tournament: str,
                    home_advantage: float, draw_inclination: float, new_teams_rating: float, is_prev_season_init: bool) -> Rating():
        """Calculate the ratings of teams, which depend on a way of initialization"""
        matches = schedule.loc[schedule['tournament'] == tournament]
        teams = matches['home_team'].unique()

        if is_prev_season_init:
            previous_matches = results.loc[(results['tournament'] == tournament) & (results['season'] == (current_season - 1))]
            previous_teams = previous_matches['home_team'].unique()

            init_ratings = {team: Rating() for team in previous_teams}

            # get ratings from previous season
            previous_ratings = self.rate_teams(previous_matches, init_ratings, home_advantage, draw_inclination)

            # update a rating deviation
            ratings = {team: Rating(mu=previous_ratings[team].mu) for team in previous_ratings}

            # teams from second league
            new_teams = [team for team in teams if team not in previous_teams]

            # initialize the ratings of new teams
            ratings.update({team: Rating(mu=new_teams_rating) for team in new_teams})

        else:
            ratings = {team: Rating() for team in teams}

        ratings = self.rate_teams(matches, ratings, home_advantage, draw_inclination)

        return ratings

    def calculate_loss(self, results: pd.DataFrame, tournament: str, season: int, home_advantage: float,
                       draw_inclination: float, new_teams_rating: float, is_prev_season_init: bool) -> float:
        """Calculate the value of the loss function"""

        # initialize Glicko2 with specific draw inclination
        glicko = Glicko2(draw_inclination=draw_inclination)

        matches = results.loc[(results['tournament'] == tournament) & (results['season'] == season)]

        ratings = self.get_ratings(results, matches, season, tournament, home_advantage, draw_inclination, new_teams_rating,
                                   is_prev_season_init)

        log_loss_value = 0
        number_matches = matches.shape[0]
        for index, row in matches.iterrows():
            outcome = row['outcome']

            # get current team ratings
            home_team, away_team = row['home_team'], row['away_team']
            home_rating, away_rating = ratings[home_team], ratings[away_team]

            # calculate outcome probabilities
            win_probability, tie_probability, loss_probability = glicko.probabilities(home_rating, away_rating, home_advantage)

            predict = [win_probability, tie_probability, loss_probability]

            if outcome == 'H':
                target = [1, 0, 0]
                ratings[home_team], ratings[away_team] = glicko.rate(home_rating, away_rating, home_advantage, is_draw=False)

            elif outcome == 'D':
                target = [0, 1, 0]
                ratings[home_team], ratings[away_team] = glicko.rate(home_rating, away_rating, home_advantage, is_draw=True)

            else:
                target = [0, 0, 1]
                ratings[away_team], ratings[home_team] = glicko.rate(away_rating, home_rating, -home_advantage, is_draw=False)

            log_loss_value += log_loss(target, predict)

        log_loss_value /= number_matches

        return log_loss_value

    def fit_parameters(self, results: pd.DataFrame, tournament: str, seasons: List[int], home_advantages: List[float],
                       draw_tendencies: List[float], new_teams_ratings: List[float],
                       is_prev_season_init: bool) -> Tuple[Tuple[float], float, Dict]:

        matches = results.loc[(results['tournament'] == tournament) & results['season'].isin(seasons)]

        # get all combinations of parameters
        parameters_list = list(product(home_advantages, draw_tendencies, new_teams_ratings))

        parameters_loss = {parameters: 0 for parameters in parameters_list}
        for parameters in tqdm(parameters_list):
            parameters_loss[parameters] = 0
            for season in seasons:
                loss = self.calculate_loss(matches, tournament, season, parameters[0], parameters[1], parameters[2],
                                           is_prev_season_init)
                parameters_loss[parameters] += loss

            parameters_loss[parameters] /= len(seasons)

        optimal_parameters = min(parameters_loss, key=parameters_loss.get)
        optimal_loss = parameters_loss[optimal_parameters]

        return optimal_parameters, optimal_loss, parameters_loss
