from collections import Counter
from random import uniform
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from glicko_soccer.predict import Predict
from lgbm_glicko.glicko_xg import GlickoExpectedGoals
from lgbm_glicko.poisson_model import PoissonLightGBM
from lgbm_glicko.train_test_creator import TrainTestCreator
from utils.metrics import three_outcomes_log_loss


class PoissonGlicko(object):

    def __init__(self, between_train_test_round=19, current_season=2020, avg_win_score_diff=1):
        self.between_train_test_round = between_train_test_round
        self.current_season = current_season
        self.avg_win_score_diff = avg_win_score_diff
        self.tournaments = ['eng', 'fr', 'ger', 'ita', 'spa']
        self.seasons = [2017, 2018, 2019]

    def fit(self, matches: pd.DataFrame, goal_params: dict, xg_params: dict, seasons: List[int]) -> Tuple[Tuple, float]:
        """"""
        glicko_predictions = GlickoExpectedGoals().predict_all(matches, goal_params, xg_params, seasons)

        _, _ = TrainTestCreator().train_test(self.tournaments, seasons)
        poisson_predictions = PoissonLightGBM().predict_all()

        poisson_predictions = (poisson_predictions
                               .drop(columns=['home_prediction', 'away_prediction'])
                               .rename(columns={'home_win': 'poisson_home_win', 'draw': 'poisson_draw',
                                                'away_win': 'poisson_away_win'}))

        predictions = (glicko_predictions
                       .merge(poisson_predictions, how='inner', on=['tournament', 'season', 'round', 'home_team', 'away_team'])
                       .merge(matches.loc[:, ['tournament', 'season', 'round', 'home_team', 'away_team', 'outcome']],
                              how='inner', on=['tournament', 'season', 'round', 'home_team', 'away_team']))

        glicko_log_loss = (predictions
                           .loc[:, ['outcome', 'glicko_home_win', 'glicko_draw', 'glicko_away_win']]
                           .apply(lambda df: three_outcomes_log_loss(df[0], df[1], df[2], df[3]), axis=1)
                           .mean())

        glicko_xg_log_loss = (predictions
                              .loc[:, ['outcome', 'glicko_xg_home_win', 'glicko_xg_draw', 'glicko_xg_away_win']]
                              .apply(lambda df: three_outcomes_log_loss(df[0], df[1], df[2], df[3]), axis=1)
                              .mean())

        poisson_log_loss = (predictions
                            .loc[:, ['outcome', 'poisson_home_win', 'poisson_draw', 'poisson_away_win']]
                            .apply(lambda df: three_outcomes_log_loss(df[0], df[1], df[2], df[3]), axis=1)
                            .mean())

        print("Glicko Log Loss: ", round(glicko_log_loss, 5))
        print("Glicko + xG Log Loss: ", round(glicko_xg_log_loss, 5))
        print("Poisson Log Loss: ", round(poisson_log_loss, 5))

        losses = dict()
        for w1 in tqdm(np.linspace(0, 1, 100)):
            for w2 in np.linspace(0, 1, 100):
                w3 = (1 - w1 - w2)

                if w3 >= 0:
                    predictions['home_win'] = (w1 * predictions['glicko_home_win']
                                               + w2 * predictions['glicko_xg_home_win']
                                               + w3 * predictions['poisson_home_win'])

                    predictions['draw'] = (w1 * predictions['glicko_draw']
                                           + w2 * predictions['glicko_xg_draw']
                                           + w3 * predictions['poisson_draw'])

                    predictions['away_win'] = (w1 * predictions['glicko_away_win']
                                               + w2 * predictions['glicko_xg_away_win']
                                               + w3 * predictions['poisson_away_win'])

                    avg_log_loss = (predictions
                                    .loc[:, ['outcome', 'home_win', 'draw', 'away_win']]
                                    .apply(lambda df: three_outcomes_log_loss(df[0], df[1], df[2], df[3]), axis=1)
                                    .mean())

                    losses[(w1, w2, w3)] = avg_log_loss

        optimal_weights = min(losses, key=losses.get)

        return optimal_weights, losses[optimal_weights]

    def predict(self, matches: pd.DataFrame, goal_params: dict, xg_params: dict, season, weights: Tuple[float]):
        """"""
        glicko_predictions = GlickoExpectedGoals().predict_all(matches, goal_params, xg_params, [season])

        _, _ = TrainTestCreator().train_test(self.tournaments, [season])
        PoissonLightGBM().save_model(season)
        poisson_predictions = PoissonLightGBM().predict(season)
        poisson_predictions = (poisson_predictions
                               .drop(columns=['home_prediction', 'away_prediction'])
                               .rename(columns={'home_win': 'poisson_home_win', 'draw': 'poisson_draw',
                                                'away_win': 'poisson_away_win'}))

        predictions = (glicko_predictions
                       .merge(poisson_predictions, how='inner', on=['tournament', 'season', 'round', 'home_team', 'away_team']))

        w1, w2, w3 = weights
        predictions['home_win'] = (w1 * predictions['glicko_home_win']
                                   + w2 * predictions['glicko_xg_home_win']
                                   + w3 * predictions['poisson_home_win'])

        predictions['draw'] = (w1 * predictions['glicko_draw']
                               + w2 * predictions['glicko_xg_draw']
                               + w3 * predictions['poisson_draw'])

        predictions['away_win'] = (w1 * predictions['glicko_away_win']
                                   + w2 * predictions['glicko_xg_away_win']
                                   + w3 * predictions['poisson_away_win'])

        return predictions

    def monte_carlo(self, matches: pd.DataFrame, predictions: pd.DataFrame, tournament: str, season: int,
                    number_iterations: int) -> pd.DataFrame:
        """"""
        played_matches = matches.loc[(matches['tournament'] == tournament)
                                     & (matches['season'] == season)
                                     & (matches['outcome'] != 'F')]

        future_matches = predictions.loc[(predictions['tournament'] == tournament)
                                         & (predictions['season'] == season)]

        points, goal_diff, matches_played = Predict().current_standings(played_matches)

        teams = played_matches['home_team'].unique()

        predicted_places = {team: [] for team in teams}
        for i in tqdm(range(number_iterations)):
            predicted_points = {team: 0 for team in teams}
            predicted_goal_diff = {team: 0 for team in teams}

            for row in future_matches.itertuples():

                home_team, away_team = row.home_team, row.away_team
                win_probability, tie_probability, loss_probability = row.home_win, row.draw, row.away_win

                random_number = uniform(0, 1)
                if random_number < win_probability:
                    predicted_points[home_team] += 3

                    predicted_goal_diff[home_team] += self.avg_win_score_diff
                    predicted_goal_diff[away_team] -= self.avg_win_score_diff

                elif random_number < win_probability + tie_probability:
                    predicted_points[home_team] += 1
                    predicted_points[away_team] += 1

                else:
                    predicted_points[away_team] += 3

                    predicted_goal_diff[away_team] += self.avg_win_score_diff
                    predicted_goal_diff[home_team] -= self.avg_win_score_diff

            total_points = {team: (points[team] + predicted_points[team], goal_diff[team] + predicted_goal_diff[team])
                            for team in teams}

            sorted_points = {team: points for team, points in sorted(total_points.items(), key=lambda item: item[1], reverse=True)}

            # rank teams in order of number of points and goal difference in played matches
            team_places = {team: list(sorted_points.keys()).index(team) + 1 for team in sorted_points}
            for team, place in team_places.items():
                predicted_places[team].append(place)

        # calculate frequency of places of each team
        for team, places in predicted_places.items():
            predicted_places[team] = {place: 0 for place in range(1, 21)}
            counter = Counter(places)
            predicted_places[team].update({place: round(100 * counts / number_iterations, 2) for place, counts in counter.items()})

        predicted_places = pd.DataFrame.from_dict(predicted_places, orient='index')
        predicted_places = (predicted_places
                            .reset_index()
                            .rename(columns={'index': 'team'}))

        # current points and matches played
        predicted_places['points'] = predicted_places['team'].map(lambda team: points[team])
        predicted_places['goal_diff'] = predicted_places['team'].map(lambda team: goal_diff[team])

        predicted_places['matches'] = predicted_places['team'].map(matches_played)

        columns_order = (['team', 'matches', 'points', 'goal_diff'] + [n for n in range(1, 21)])

        predicted_places = (predicted_places
                            .loc[:, columns_order]
                            .sort_values(['points', 'goal_diff'], ascending=False)
                            .reset_index(drop=True)
                            .reset_index()
                            .rename(columns={'index': 'rk'}))

        predicted_places['rk'] += 1

        return predicted_places
