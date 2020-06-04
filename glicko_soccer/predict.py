from random import uniform

import pandas as pd
from tqdm import tqdm

from glicko2 import Glicko2, Rating
from glicko_soccer.glicko_soccer import GlickoSoccer


class FutureMatches(object):

    @staticmethod
    def monte_carlo(ratings: Rating(), schedule: pd.DataFrame, tournament: str,
                    home_advantage: float, draw_inclination: float, number_iterations: int):
        """"""

        played_matches = schedule.loc[(schedule['outcome'] != 'F') & (schedule['tournament'] == tournament)]
        future_matches = schedule.loc[(schedule['outcome'] == 'F') & (schedule['tournament'] == tournament)]

        glicko = Glicko2(draw_inclination=draw_inclination)

        # calculate points in played matches
        points = {team: 0 for team in played_matches['home_team'].unique()}
        matches_played = {team: 0 for team in played_matches['home_team'].unique()}
        for index, row in played_matches.iterrows():

            home_team, away_team = row['home_team'], row['away_team']
            matches_played[home_team] += 1
            matches_played[away_team] += 1

            if row['outcome'] == 'H':
                points[home_team] += 3

            elif row['outcome'] == 'D':
                points[home_team] += 1
                points[away_team] += 1

            else:
                points[away_team] += 3

        # simulate future matches
        predicted_places = {team: [] for team in points}
        for i in tqdm(range(number_iterations)):
            predicted_points = {team: 0 for team in points}

            for index, row in future_matches.iterrows():
                home_team, away_team = row['home_team'], row['away_team']
                home_rating, away_rating = ratings[home_team], ratings[away_team]

                win_probability, tie_probability, loss_probability = glicko.probabilities(home_rating, away_rating, home_advantage)

                # monte carlo
                random_number = uniform(0, 1)
                if random_number < win_probability:
                    predicted_points[home_team] += 3

                elif random_number < win_probability + tie_probability:
                    predicted_points[home_team] += 1
                    predicted_points[away_team] += 1

                else:
                    predicted_points[away_team] += 3

            total_points = {team: (points[team] + predicted_points[team]) for team in points}
            sorted_points = {team: points for team, points in sorted(total_points.items(), key=lambda item: item[1], reverse=True)}

            # rank teams in order of number of points
            team_places = {team: list(sorted_points.keys()).index(team) + 1 for team in sorted_points}
            for team, place in team_places.items():
                predicted_places[team].append(place)

        # calculate frequency of places of each team
        for team, places in predicted_places.items():
            predicted_places[team] = {place: 0 for place in range(1, 21)}
            predicted_places[team].update({place: round(100 * places.count(place) / number_iterations, 1) for place in places})

        predicted_places = pd.DataFrame.from_dict(predicted_places, orient='index')
        predicted_places = (predicted_places
                            .reset_index()
                            .rename(columns={'index': 'team'}))

        # current points and matches played
        predicted_places['points'] = predicted_places['team'].map(points)
        predicted_places['matches'] = predicted_places['team'].map(matches_played)

        columns_order = (['team', 'matches', 'points'] + [n for n in range(1, 20)])

        predicted_places = (predicted_places
                            .loc[:, columns_order]
                            .sort_values(['points'], ascending=False)
                            .reset_index(drop=True)
                            .reset_index()
                            .rename(columns={'index': 'place'}))

        predicted_places['place'] += 1

        return predicted_places

    def transform(self, results: pd.DataFrame, schedule: pd.DataFrame, current_season: int, tournament: str,
                  home_advantage: float, draw_inclination: float, new_teams_rating: float, number_iterations: int,
                  is_prev_season_init):
        """"""

        init_ratings = GlickoSoccer().get_ratings(results, schedule, current_season, tournament, home_advantage,
                                                  draw_inclination, new_teams_rating, is_prev_season_init)

        team_places = self.monte_carlo(init_ratings, schedule, tournament, home_advantage, draw_inclination, number_iterations)
        return team_places
