from collections import Counter
from random import uniform
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from glicko2 import Glicko2, Rating
from glicko_soccer.glicko_soccer import GlickoSoccer


class Predict(object):

    def __init__(self, is_draw_mode=True, is_prev_season_init=False, avg_win_score_diff=1.28):
        self.is_draw_mode = is_draw_mode
        self.is_prev_season_init = is_prev_season_init
        self.avg_win_score_diff = avg_win_score_diff

    @staticmethod
    def current_standings(matches: pd.DataFrame) -> Tuple[dict, dict, dict]:
        """
            Calculate points, goal difference, a number of played matches in previous games.
        """
        matches[['home_score', 'away_score']] = matches[['home_score', 'away_score']].to_numpy('int')

        teams = matches['home_team'].unique()

        points = {team: 0 for team in teams}
        goal_diff = {team: 0 for team in teams}
        matches_played = {team: 0 for team in teams}

        for row in matches.itertuples():

            outcome, home_team, away_team = row.outcome, row.home_team, row.away_team
            home_score, away_score = row.home_score, row.away_score

            matches_played[home_team] += 1
            matches_played[away_team] += 1

            goal_diff[home_team] += (home_score - away_score)
            goal_diff[away_team] += (away_score - home_score)

            if outcome == 'H':
                points[home_team] += 3

            elif outcome == 'D':
                points[home_team] += 1
                points[away_team] += 1

            else:
                points[away_team] += 3

        return points, goal_diff, matches_played

    def monte_carlo(self, ratings: Rating(), schedule: pd.DataFrame, tournament: str,
                    home_advantage: float, draw_inclination: float, number_iterations: int) -> pd.DataFrame:
        """"""
        glicko = Glicko2(is_draw_mode=self.is_draw_mode, draw_inclination=draw_inclination)

        played_matches = schedule.loc[(schedule['outcome'] != 'F') & (schedule['tournament'] == tournament)]

        # calculate ratings in played matches
        ratings = GlickoSoccer().rate_teams(played_matches, ratings, draw_inclination, home_advantage)

        # calculate points in played matches
        points, goal_diff, matches_played = self.current_standings(played_matches)

        teams = played_matches['home_team'].unique()

        # simulate future matches
        future_matches = schedule.loc[(schedule['outcome'] == 'F') & (schedule['tournament'] == tournament),
                                      ['home_team', 'away_team']]

        predicted_places = {team: [] for team in teams}
        for i in tqdm(range(number_iterations)):
            predicted_points = {team: 0 for team in teams}
            predicted_goal_diff = {team: 0 for team in teams}

            for row in future_matches.itertuples():

                home_team, away_team = row.home_team, row.away_team
                home_rating, away_rating = ratings[home_team], ratings[away_team]

                win_probability, tie_probability, loss_probability = glicko.probabilities(home_rating, away_rating, home_advantage)

                # monte carlo
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

    def transform(self, results: pd.DataFrame, schedule: pd.DataFrame, current_season: int, tournament: str,
                  init_rd: float, draw_inclination: float, home_advantage: float, new_teams_rating: float,
                  number_iterations: int):
        """"""

        init_ratings = GlickoSoccer().ratings_initialization(results, schedule, current_season, tournament,
                                                             init_rd, draw_inclination, home_advantage, new_teams_rating)

        team_places = self.monte_carlo(init_ratings, schedule, tournament, home_advantage, draw_inclination, number_iterations)
        return team_places
