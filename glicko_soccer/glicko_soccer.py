from itertools import product
from typing import List, Tuple, Dict

import pandas as pd
from sklearn.metrics import log_loss
from tqdm import tqdm

from glicko2 import Glicko2, Rating


class GlickoSoccer(object):

    def __init__(self, is_draw_mode=True, is_prev_season_init=False):
        self.is_draw_mode = is_draw_mode
        self.is_prev_season_init = is_prev_season_init

    def rate_teams(self, matches: pd.DataFrame, ratings: Rating(), draw_inclination: float, home_advantage: float) -> Rating():
        """
            Calculate the ratings of teams using the history of matches
        """

        glicko = Glicko2(is_draw_mode=self.is_draw_mode, draw_inclination=draw_inclination)

        for index, row in matches.iterrows():
            outcome, home_team, away_team = row['outcome'], row['home_team'], row['away_team']

            ratings[home_team], ratings[away_team] = glicko.rate(ratings[home_team], ratings[away_team], home_advantage, outcome)

        return ratings

    def ratings_initialization(self, results: pd.DataFrame, schedule: pd.DataFrame, current_season: int, tournament: str,
                               init_rd: float, draw_inclination: float, home_advantage: float, new_teams_rating: float) -> Rating():
        """
            Two ways of rating initialization:
            1) all teams get the same rating
            2) teams get ratings from previous season with new rating deviation,
               teams from second league get specific ratings
        """
        teams = schedule.loc[(schedule['tournament'] == tournament), 'home_team'].unique()

        if self.is_prev_season_init:
            previous_matches = results.loc[(results['tournament'] == tournament) & (results['season'] == (current_season - 1))]
            previous_teams = previous_matches['home_team'].unique()

            init_ratings = {team: Rating(rd=init_rd) for team in previous_teams}

            # get ratings from previous season
            previous_ratings = self.rate_teams(previous_matches, init_ratings, draw_inclination, home_advantage)

            # update a rating deviation
            ratings = {team: Rating(mu=previous_ratings[team].mu, rd=init_rd) for team in previous_ratings}

            # teams from second league
            new_teams = [team for team in teams if team not in previous_teams]

            # initialize the ratings of new teams
            ratings.update({team: Rating(mu=new_teams_rating, rd=init_rd) for team in new_teams})

        else:
            ratings = {team: Rating(rd=init_rd) for team in teams}

        return ratings

    @staticmethod
    def _log_loss(outcome: str, win_probability: float, tie_probability: float, loss_probability: float) -> float:
        """
            Calculate log loss value of one match.
        """
        predict = [win_probability, tie_probability, loss_probability]

        if outcome == 'H':
            target = [1, 0, 0]

        elif outcome == 'D':
            target = [0, 1, 0]

        else:
            target = [0, 0, 1]

        return log_loss(target, predict)

    def calculate_loss(self, results: pd.DataFrame, tournament: str, season: int, init_rd: float, draw_inclination: float,
                       home_advantage: float, new_teams_rating: float) -> float:
        """
            Calculate the value of the loss function
        """

        # initialize Glicko2 with specific draw inclination
        glicko = Glicko2(is_draw_mode=self.is_draw_mode, draw_inclination=draw_inclination)

        matches = results.loc[(results['tournament'] == tournament) & (results['season'] == season)]

        ratings = self.ratings_initialization(results, matches, season, tournament, init_rd, draw_inclination,
                                              home_advantage, new_teams_rating)

        log_loss_value = 0
        number_matches = matches.shape[0]
        for index, row in matches.iterrows():

            outcome, home_team, away_team = row['outcome'], row['home_team'], row['away_team']

            # get current team ratings
            home_rating, away_rating = ratings[home_team], ratings[away_team]

            # calculate outcome probabilities
            win_probability, tie_probability, loss_probability = glicko.probabilities(home_rating, away_rating, home_advantage)

            ratings[home_team], ratings[away_team] = glicko.rate(home_rating, away_rating, home_advantage, outcome)

            log_loss_value += self._log_loss(outcome, win_probability, tie_probability, loss_probability)

        log_loss_value /= number_matches

        return log_loss_value

    def fit_parameters(self, results: pd.DataFrame, tournament: str, seasons: List[int],
                       init_rds: List[float], draw_inclinations: List[float],
                       home_advantages: List[float], new_teams_ratings: List[float]) -> Tuple[Tuple[float], float, Dict]:

        matches = results.loc[(results['tournament'] == tournament) & results['season'].isin(seasons)]

        # get all combinations of parameters
        parameters_list = list(product(init_rds, draw_inclinations, home_advantages, new_teams_ratings))

        parameters_loss = {parameters: 0 for parameters in parameters_list}
        for parameters in tqdm(parameters_list):

            init_rd, draw_inclination, home_advantage, new_teams_rating = parameters

            parameters_loss[parameters] = 0
            for season in seasons:
                loss = self.calculate_loss(matches, tournament, season, init_rd, draw_inclination, home_advantage, new_teams_rating)

                parameters_loss[parameters] += loss

            parameters_loss[parameters] /= len(seasons)

        optimal_parameters = min(parameters_loss, key=parameters_loss.get)
        optimal_loss = parameters_loss[optimal_parameters]

        return optimal_parameters, optimal_loss, parameters_loss
