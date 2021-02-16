from itertools import product
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from glicko2 import Glicko2, Rating
from lgbm_glicko.preprocessing import DataPreprocessor
from utils.metrics import three_outcomes_log_loss


class GlickoExpectedGoals(object):

    def __init__(self, is_draw_mode=True, is_xg_mode=True, between_train_test_round=19, current_season=2020):
        self.is_draw_mode = is_draw_mode
        self.is_xg_mode = is_xg_mode
        self.between_train_test_round = between_train_test_round
        self.current_season = current_season
        self.tournaments = ['eng', 'fr', 'ger', 'ita', 'spa']
        self.seasons = [2017, 2018, 2019]

    def rate_teams(self, matches: pd.DataFrame, ratings: Rating(), draw_inclination: float, home_advantage: float) -> dict:
        """
            Calculate the ratings of teams using the history of matches
        """

        glicko = Glicko2(is_draw_mode=self.is_draw_mode, draw_inclination=draw_inclination)

        if self.is_xg_mode:
            for row in matches.itertuples():
                home_team, away_team, outcome = row.home_team, row.away_team, row.xg_outcome

                ratings[home_team], ratings[away_team] = glicko.rate(ratings[home_team], ratings[away_team], home_advantage, outcome)
        else:
            for row in matches.itertuples():
                home_team, away_team, outcome = row.home_team, row.away_team, row.outcome

                ratings[home_team], ratings[away_team] = glicko.rate(ratings[home_team], ratings[away_team], home_advantage, outcome)

        return ratings

    def calculate_loss(self, results: pd.DataFrame, season: int, init_rd: float, draw_inclination: float,
                       home_advantage: float) -> float:
        """
            Calculate the value of the loss function
        """

        # initialize Glicko2 with specific draw inclination
        glicko = Glicko2(is_draw_mode=self.is_draw_mode, draw_inclination=draw_inclination)

        matches = results.loc[(results['season'] == season)]

        # rating initialization
        ratings = {team: Rating(rd=init_rd) for team in matches['home_team'].unique()}

        if self.between_train_test_round == 0:
            test_matches = matches
        else:
            # separate matches into two parts: calculate ratings from first part,
            # for second part predict matches, calculate loss value
            train_matches = matches.loc[matches['round'] <= self.between_train_test_round]
            test_matches = matches.loc[matches['round'] > self.between_train_test_round]

            ratings = self.rate_teams(train_matches, ratings, draw_inclination, home_advantage)

        log_loss_value = 0
        number_matches = test_matches.shape[0]
        for row in test_matches.itertuples():
            outcome, home_team, away_team = row.outcome, row.home_team, row.away_team

            # calculate outcome probabilities
            win_probability, tie_probability, loss_probability = glicko.probabilities(ratings[home_team], ratings[away_team],
                                                                                      home_advantage)

            log_loss_value += three_outcomes_log_loss(outcome, win_probability, tie_probability, loss_probability)

        log_loss_value /= number_matches

        return log_loss_value

    def fit_parameters(self, results: pd.DataFrame, tournament: str, seasons: List[int],
                       init_rds: np.ndarray, draw_inclinations: np.ndarray,
                       home_advantages: np.ndarray) -> Tuple[Tuple[float], float, Dict]:

        matches = results.loc[(results['tournament'] == tournament) & results['season'].isin(seasons)]

        # get all combinations of parameters
        parameters_list = list(product(init_rds, draw_inclinations, home_advantages))

        parameters_loss = {parameters: 0 for parameters in parameters_list}
        for parameters in tqdm(parameters_list):
            init_rd, draw_inclination, home_advantage = parameters

            parameters_loss[parameters] = 0
            for season in seasons:
                loss = self.calculate_loss(matches, season, init_rd, draw_inclination, home_advantage)
                parameters_loss[parameters] += loss

            parameters_loss[parameters] /= len(seasons)

        optimal_parameters = min(parameters_loss, key=parameters_loss.get)
        optimal_loss = parameters_loss[optimal_parameters]

        return optimal_parameters, optimal_loss, parameters_loss

    def fit_all_parameters(self, init_rds: np.ndarray, draw_inclinations: np.ndarray,
                           home_advantages: np.ndarray) -> Tuple[dict, dict]:
        """"""
        matches_list = []
        for tournament in self.tournaments:
            for season in self.seasons:
                matches = pd.read_excel('data/xg_' + tournament + '.xlsx', sheet_name=str(season))
                matches = DataPreprocessor().preprocessing_one_season(matches)
                matches['tournament'] = tournament
                matches['season'] = season
                matches_list.append(matches)

        matches = pd.concat(matches_list)

        goals_glicko = GlickoExpectedGoals(is_xg_mode=False)
        xg_glicko = GlickoExpectedGoals(is_xg_mode=True)

        goal_params = dict()
        xg_params = dict()
        for tournament in self.tournaments:
            optimal_parameters = goals_glicko.fit_parameters(matches, tournament, self.seasons, init_rds, draw_inclinations,
                                                             home_advantages)

            goal_params[tournament] = optimal_parameters[0], optimal_parameters[1]

            optimal_parameters = xg_glicko.fit_parameters(matches, tournament, self.seasons, init_rds, draw_inclinations,
                                                          home_advantages)

            xg_params[tournament] = optimal_parameters[0], optimal_parameters[1]

        return goal_params, xg_params

    def predict(self, results: pd.DataFrame, tournament: str, season: int, init_rd: float,
                draw_inclination: float, home_advantage: float) -> pd.DataFrame:
        """"""

        glicko = Glicko2(is_draw_mode=self.is_draw_mode, draw_inclination=draw_inclination)

        matches = results.loc[(results['tournament'] == tournament) & (results['season'] == season)]

        # rating initialization
        ratings = {team: Rating(rd=init_rd) for team in matches['home_team'].unique()}

        if season == self.current_season:
            played_matches = matches.loc[(matches['outcome'] != 'F')]
            future_matches = matches.loc[(matches['outcome'] == 'F')].reset_index()

        else:
            played_matches = matches.loc[matches['round'] <= self.between_train_test_round]
            future_matches = matches.loc[matches['round'] > self.between_train_test_round].reset_index()

        ratings = self.rate_teams(played_matches, ratings, draw_inclination, home_advantage)

        predictions = dict()
        for row in future_matches.itertuples():
            home_rating, away_rating = ratings[row.home_team], ratings[row.away_team]

            win_probability, tie_probability, loss_probability = glicko.probabilities(home_rating, away_rating, home_advantage)

            predictions[row.index] = (win_probability, tie_probability, loss_probability)

        predictions = pd.DataFrame.from_dict(predictions.items())

        predictions = predictions.rename(columns={0: 'index', 1: 'prediction'})

        predictions[['home_win', 'draw', 'away_win']] = pd.DataFrame(predictions['prediction'].tolist(), index=predictions.index)

        predictions = future_matches.merge(predictions, how='inner', on=['index'])

        predictions = predictions.loc[:, ['tournament', 'season', 'date', 'round', 'home_team', 'away_team',
                                          'home_win', 'draw', 'away_win']]

        return predictions

    def predict_all(self, matches: pd.DataFrame, goal_params: dict, xg_params: dict, seasons: List[int]):
        """"""
        if seasons is None:
            seasons = self.seasons

        # predictions based on goal params
        predictions_list = []
        for tournament, params in goal_params.items():
            init_rd, draw_inclination, home_advantage = params[0]
            for season in seasons:
                predictions = GlickoExpectedGoals(is_xg_mode=False).predict(matches, tournament, season,
                                                                            init_rd, draw_inclination, home_advantage)
                predictions_list.append(predictions)

        goal_predictions = pd.concat(predictions_list)

        goal_predictions = goal_predictions.rename(columns={'home_win': 'glicko_home_win', 'draw': 'glicko_draw',
                                                            'away_win': 'glicko_away_win'})

        # predictions based on xg params
        predictions_list = []
        for tournament, params in xg_params.items():
            init_rd, draw_inclination, home_advantage = params[0]
            for season in seasons:
                predictions = GlickoExpectedGoals(is_xg_mode=True).predict(matches, tournament, season,
                                                                           init_rd, draw_inclination, home_advantage)
                predictions_list.append(predictions)

        xg_predictions = pd.concat(predictions_list)

        xg_predictions = (xg_predictions
                          .rename(columns={'home_win': 'glicko_xg_home_win', 'draw': 'glicko_xg_draw',
                                           'away_win': 'glicko_xg_away_win'})
                          .drop(columns=['date']))

        predictions = goal_predictions.merge(xg_predictions, how='inner',
                                             on=['tournament', 'season', 'round', 'home_team', 'away_team'])

        return predictions
