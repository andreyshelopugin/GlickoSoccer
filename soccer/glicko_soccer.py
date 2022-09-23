from itertools import product
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from config import Config
from glicko2 import Glicko2, Rating
from utils.metrics import three_outcomes_log_loss


class GlickoSoccer(object):

    def __init__(self, init_mu=1500, init_rd=150, volatility=0.04, update_rd=40, lift_update_mu=30,
                 home_advantage=28, pandemic_home_advantage=19, draw_inclination=-0.5052, new_team_update_mu=-42):
        self.init_mu = init_mu
        self.init_rd = init_rd
        self.volatility = volatility
        self.lift_update_mu = lift_update_mu
        self.update_rd = update_rd
        self.home_advantage = home_advantage
        self.pandemic_home_advantage = pandemic_home_advantage
        self.draw_inclination = draw_inclination
        self.new_team_update_mu = new_team_update_mu
        self.first_leagues = None
        self.south_america_leagues = {'Brazil. First',
                                      'Argentina. First',
                                      'Paraguay. First',
                                      'Uruguay. First',
                                      'Ecuador. First',
                                      'Colombia. First',
                                      'Chile. First',
                                      'Bolivia. First',
                                      'Peru. First',
                                      'Venezuela. First',
                                      'Brazil. Second',
                                      'Argentina. Second'}

    @staticmethod
    def _team_leagues(results: pd.DataFrame, season: int) -> dict:
        """Returns dict {Team:League} for a specific season"""

        no_cups = (results.loc[results['tournament_type'].isin({1, 2})
                               & (results['season'] == season), ['home_team', 'away_team', 'tournament']])

        home_teams = no_cups.drop_duplicates(['home_team'], keep='first')
        away_teams = no_cups.drop_duplicates(['away_team'], keep='first')

        team_leagues = dict(zip(home_teams['home_team'], home_teams['tournament']))

        team_leagues.update(dict(zip(away_teams['away_team'], away_teams['tournament'])))

        return team_leagues

    def _league_params_initialization(self, results: pd.DataFrame) -> dict:
        """"""
        first_leagues = set(results.loc[results['tournament_type'] == 1, 'tournament'])
        second_leagues = set(results.loc[results['tournament_type'] == 2, 'tournament'])

        leagues = first_leagues.union(second_leagues)

        league_params = dict()
        for league in leagues:

            if league in first_leagues:
                league_params[league] = {'init_mu': self.init_mu,
                                         'init_rd': self.init_rd,
                                         'update_rd': self.update_rd,
                                         'lift_update_mu': self.lift_update_mu,
                                         'home_advantage': self.home_advantage,
                                         'pandemic_home_advantage': self.pandemic_home_advantage,
                                         'new_team_update_mu': 0}
            else:
                league_params[league] = {'init_mu': self.init_mu,
                                         'init_rd': self.init_rd,
                                         'update_rd': self.update_rd,
                                         'lift_update_mu': -self.lift_update_mu,
                                         'home_advantage': self.home_advantage,
                                         'pandemic_home_advantage': self.pandemic_home_advantage,
                                         'new_team_update_mu': self.new_team_update_mu}

        return league_params

    @staticmethod
    def _team_params(season: int, league_params: dict, team_leagues_all: dict) -> dict:
        """Gets league params for each team ."""
        return {team: league_params[league] for team, league in team_leagues_all[season].items()}

    def _rating_initialization(self, results: pd.DataFrame, team_params: dict) -> dict:
        """Initializes teams ratings. Initialization depends on league parameters"""
        seasons = sorted(results['season'].unique())
        ratings = dict()
        for season in seasons:
            for team, params in team_params[season].items():
                if team not in ratings:
                    ratings[team] = Rating(mu=params['init_mu'], rd=params['init_rd'], volatility=self.volatility)

        # normalization as a way for fighting inflation
        mean_rating = np.mean([rating.mu for _, rating in ratings.items()])
        ratings = {team: Rating(mu=self.init_mu * rating.mu / mean_rating, rd=rating.rd, volatility=self.volatility)
                   for team, rating in ratings.items()}

        return ratings

    @staticmethod
    def _update_ratings_match_ids(results: pd.DataFrame) -> Tuple[dict, dict, dict, set]:
        """Finds matches that require params update"""
        no_cups = results.loc[results['tournament_type'].isin({1, 2})]

        no_cups = pd.concat([no_cups.loc[:, ['date', 'match_id', 'home_team', 'season', 'tournament']]
                            .rename(columns={'home_team': 'team'}),
                             no_cups.loc[:, ['date', 'match_id', 'away_team', 'season', 'tournament']]
                            .rename(columns={'away_team': 'team'})])

        no_cups = (no_cups
                   .sort_values(['team', 'date'])
                   .drop_duplicates(['team', 'season'], keep='first'))

        # teams with no data about previous seasons:
        # (removes first seasons too, fix it later)
        missed_previous_season = (no_cups
                                  .loc[(no_cups['season'] != no_cups['season'].shift() + 1)
                                       & (no_cups['team'] == no_cups['team'].shift())]
                                  .groupby(['team'])
                                  ['match_id']
                                  .apply(set)
                                  .to_dict())

        # don't take into account first season
        first_team_season = no_cups.drop_duplicates(['team'])

        first_team_season = first_team_season.loc[first_team_season['season'] != first_team_season['season'].min()]

        first_team_season = dict(zip(first_team_season['team'], first_team_season['match_id']))

        for team, first_season_index in first_team_season.items():
            if team in missed_previous_season:
                new_value = missed_previous_season[team]
                new_value.add(first_season_index)
                missed_previous_season[team] = new_value
            else:
                missed_previous_season[team] = {first_season_index}

        # teams that changed league
        changed_league = (no_cups
                          .loc[(no_cups['season'] == no_cups['season'].shift() + 1)
                               & (no_cups['tournament'] != no_cups['tournament'].shift())
                               & (no_cups['team'] == no_cups['team'].shift())]
                          .groupby(['team'])
                          ['match_id']
                          .apply(set)
                          .to_dict())

        # teams that did not change league
        same_league = (no_cups
                       .loc[(no_cups['season'] == no_cups['season'].shift() + 1)
                            & (no_cups['tournament'] == no_cups['tournament'].shift())
                            & (no_cups['team'] == no_cups['team'].shift())]
                       .groupby(['team'])
                       ['match_id']
                       .apply(set)
                       .to_dict())

        matches_ids_for_update_ratings = (set.union(*missed_previous_season.values())
                                          .union(set.union(*changed_league.values()))
                                          .union(set.union(*same_league.values())))

        return missed_previous_season, changed_league, same_league, matches_ids_for_update_ratings

    def _season_update_rating(self, ratings: dict, home_team: str, away_team: str, match_id: int,
                              home_params: dict, away_params: dict,
                              missed_previous_season: Dict[str, set], changed_league: Dict[str, set],
                              same_league: Dict[str, set]) -> dict:
        """Update rating's mu and RD after each season."""
        for team in [home_team, away_team]:
            if team == home_team:
                params = home_params
            else:
                params = away_params

            if team in same_league:
                if match_id in same_league[team]:
                    ratings[team] = Rating(mu=ratings[team].mu,
                                           rd=ratings[team].rd + params['update_rd'],
                                           volatility=self.volatility)

            elif team in missed_previous_season:
                if match_id in missed_previous_season[team]:
                    ratings[team] = Rating(mu=params['init_mu'] + params['new_team_update_mu'],
                                           rd=params['init_rd'],
                                           volatility=self.volatility)

            elif team in changed_league:
                if match_id in changed_league[team]:
                    ratings[team] = Rating(mu=ratings[team].mu + params['lift_update_mu'],
                                           rd=ratings[team].rd + params['update_rd'],
                                           volatility=self.volatility)

        return ratings

    def rate_teams(self, results: pd.DataFrame, league_params: dict) -> dict:
        """Calculate team ratings."""

        glicko = Glicko2(draw_inclination=self.draw_inclination)

        seasons = set(results['season'])

        team_leagues_all = {season: self._team_leagues(results, season) for season in seasons}

        missed_prev, changed, same, match_ids_for_update = self._update_ratings_match_ids(results)
        results = results.drop(columns=['date', 'country'])

        team_params = {season: self._team_params(season, league_params, team_leagues_all) for season in seasons}

        ratings = self._rating_initialization(results, team_params)

        for row in results.itertuples(index=False):

            match_id, home_team, away_team, season = row.match_id, row.home_team, row.away_team, row.season
            outcome, skellam_draw_probability, is_pandemic = row.outcome, row.draw_probability, row.is_pandemic
            tournament_type = row.tournament_type

            home_params = team_params[season][home_team]
            away_params = team_params[season][away_team]

            if match_id in match_ids_for_update:
                ratings = self._season_update_rating(ratings, home_team, away_team, match_id, home_params, away_params,
                                                     missed_prev, changed, same)

            if tournament_type == 4:
                home_advantage = 0
            else:
                if is_pandemic == 1:
                    home_advantage = home_params['pandemic_home_advantage']
                else:
                    home_advantage = home_params['home_advantage']

            # update team ratings
            ratings[home_team], ratings[away_team] = glicko.rate(ratings[home_team], ratings[away_team], home_advantage,
                                                                 outcome,
                                                                 skellam_draw_probability)

        return ratings

    def predictions(self, results: pd.DataFrame, league_params: dict, start_season: int) -> pd.DataFrame:
        """"""
        glicko = Glicko2(draw_inclination=self.draw_inclination)

        seasons = set(results['season'])

        team_leagues_all = {season: self._team_leagues(results, season) for season in seasons}

        missed_prev, changed, same, match_ids_for_update = self._update_ratings_match_ids(results)
        results = results.drop(columns=['date', 'country'])

        team_params = {season: self._team_params(season, league_params, team_leagues_all) for season in seasons}

        ratings = self._rating_initialization(results, team_params)

        predictions = dict()
        for row in results.itertuples(index=False):

            match_id, home_team, away_team, season = row.match_id, row.home_team, row.away_team, row.season
            outcome, skellam_draw_probability, is_pandemic = row.outcome, row.draw_probability, row.is_pandemic
            tournament_type = row.tournament_type

            home_params = team_params[season][home_team]
            away_params = team_params[season][away_team]

            if match_id in match_ids_for_update:
                ratings = self._season_update_rating(ratings, home_team, away_team, match_id, home_params, away_params,
                                                     missed_prev, changed, same)

            if tournament_type == 4:
                home_advantage = 0
            else:
                if is_pandemic == 1:
                    home_advantage = home_params['pandemic_home_advantage']
                else:
                    home_advantage = home_params['home_advantage']

            if season >= start_season:
                win_probability, tie_probability, loss_probability = glicko.probabilities(ratings[home_team],
                                                                                          ratings[away_team],
                                                                                          home_advantage,
                                                                                          skellam_draw_probability)

                predictions[row.match_id] = (win_probability, tie_probability, loss_probability)

            # update team ratings
            ratings[home_team], ratings[away_team] = glicko.rate(ratings[home_team], ratings[away_team], home_advantage,
                                                                 outcome,
                                                                 skellam_draw_probability)

        predictions = pd.DataFrame.from_dict(predictions.items())

        predictions = predictions.rename(columns={0: 'match_id', 1: 'prediction'})

        predictions[['home_win', 'draw', 'away_win']] = pd.DataFrame(predictions['prediction'].tolist(), index=predictions.index)

        predictions = predictions.drop(columns=['prediction'])

        return predictions

    def calculate_loss(self, results: pd.DataFrame, league_params: dict, team_leagues_all: dict,
                       missed_previous_season: dict, changed_league: dict, same_league: dict,
                       match_ids_for_update: set) -> float:
        """"""

        glicko = Glicko2(draw_inclination=self.draw_inclination)

        team_params = {season: self._team_params(season, league_params, team_leagues_all) for season
                       in results['season'].unique()}

        first_season = results['season'].min()

        ratings = self._rating_initialization(results, team_params)

        log_loss_value = 0
        for row in results.itertuples(index=False):

            match_id, home_team, away_team, season = row.match_id, row.home_team, row.away_team, row.season
            outcome, skellam_draw_probability, is_pandemic = row.outcome, row.draw_probability, row.is_pandemic
            tournament_type = row.tournament_type

            home_params = team_params[season][home_team]
            away_params = team_params[season][away_team]

            if match_id in match_ids_for_update:
                ratings = self._season_update_rating(ratings, home_team, away_team, match_id, home_params, away_params,
                                                     missed_previous_season, changed_league, same_league)

            # neutral field
            if tournament_type == 4:
                home_advantage = 0
            else:
                if is_pandemic == 1:
                    home_advantage = home_params['pandemic_home_advantage']
                else:
                    home_advantage = home_params['home_advantage']

            # get current team ratings
            home_rating, away_rating = ratings[home_team], ratings[away_team]

            # don't optimize loss for first two seasons in order to prevent overfitting
            if season >= first_season + 2:
                # calculate outcome probabilities
                win_probability, tie_probability, loss_probability = glicko.probabilities(home_rating,
                                                                                          away_rating,
                                                                                          home_advantage,
                                                                                          skellam_draw_probability)

                # assign the last seasons greater weights to make the model more actual
                log_loss_value += (season - first_season - 1) * three_outcomes_log_loss(win_probability,
                                                                                        tie_probability,
                                                                                        loss_probability,
                                                                                        outcome)

            # update team ratings
            ratings[home_team], ratings[away_team] = glicko.rate(home_rating,
                                                                 away_rating,
                                                                 home_advantage,
                                                                 outcome,
                                                                 skellam_draw_probability)

        return log_loss_value

    def _params_regularization(self, league: str, init_rds: list, update_rds: list,
                               homes: list, pandemic_homes: list,
                               lift_update_mus: list, new_team_update_mus: list) -> tuple:
        """ Home advantage during the pandemia should be less than home advantage at other times.
        Other params are restricted for preventing overfitting."""
        init_rds = [x for x in init_rds if x >= 100]
        update_rds = [x for x in update_rds if x >= 25]
        homes = [x for x in homes if x >= 5]
        pandemic_homes = [x for x in pandemic_homes if (x >= 0) and (x < max(homes))]

        if league in self.first_leagues:
            lift_update_mus = [x for x in lift_update_mus if 0 <= x <= 120]
            new_team_update_mus = [0]
        else:
            lift_update_mus = [x for x in lift_update_mus if -120 <= x <= 0]
            new_team_update_mus = [x for x in new_team_update_mus if -10 >= x >= -150]

        if not lift_update_mus:
            lift_update_mus = [0]

        if not new_team_update_mus:
            new_team_update_mus = [0]

        if not pandemic_homes:
            pandemic_homes = [max(homes)]

        if league == 'Netherlands. Second':
            new_team_update_mus = [0]

        return init_rds, update_rds, homes, pandemic_homes, lift_update_mus, new_team_update_mus

    def fit_params(self, results: pd.DataFrame, number_iterations: int, is_params_initialization: True):
        """"""

        if is_params_initialization:
            league_params = self._league_params_initialization(results)
        else:
            league_params = joblib.load(Config().project_path + Config().ratings_paths['league_params'])

        self.first_leagues = set(results.loc[results['tournament_type'] == 1, 'tournament'])

        seasons = set(results['season'])

        team_leagues_all = {season: self._team_leagues(results, season) for season in seasons}

        missed_prev, changed, same, indexes_for_update = self._update_ratings_match_ids(results)
        results = results.drop(columns=['date', 'country'])

        current_loss = self.calculate_loss(results, league_params, team_leagues_all, missed_prev, changed, same, indexes_for_update)

        for i in range(number_iterations):

            draw_inclination_list = np.linspace(self.draw_inclination - 0.0001, self.draw_inclination + 0.0001, 6)

            best_draw = self.draw_inclination
            for draw in draw_inclination_list:
                self.draw_inclination = draw
                loss = self.calculate_loss(results, league_params, team_leagues_all, missed_prev, changed, same, indexes_for_update)
                if loss < current_loss:
                    current_loss = loss
                    best_draw = draw

            self.draw_inclination = best_draw

            for league, params in league_params.items():

                init_mu = params['init_mu']
                init_rd = params['init_rd']
                update_rd = params['update_rd']
                lift_update_mu = params['lift_update_mu']
                home_advantage = params['home_advantage']
                pandemic_home_advantage = params['pandemic_home_advantage']
                new_team_update_mu = params['new_team_update_mu']

                init_mus = [init_mu - 10, init_mu, init_mu + 10]
                init_rds = [init_rd]
                update_rds = [update_rd]
                lift_update_mus = [lift_update_mu]
                homes = [home_advantage - 2, home_advantage, home_advantage + 2]
                pandemic_homes = [pandemic_home_advantage]
                new_team_update_mus = [new_team_update_mu]

                init_rds, update_rds, homes, pandemic_homes, lift_update_mus, new_team_update_mus = self._params_regularization(
                    league,
                    init_rds,
                    update_rds,
                    homes,
                    pandemic_homes,
                    lift_update_mus,
                    new_team_update_mus)

                params_list = list(product(init_mus,
                                           init_rds,
                                           update_rds,
                                           lift_update_mus,
                                           homes,
                                           pandemic_homes,
                                           new_team_update_mus))

                params_loss = {params: 0 for params in params_list}
                league_params_copy = league_params.copy()
                for country_params in params_list:
                    league_params_copy[league].update(zip(league_params_copy[league], country_params))

                    params_loss[country_params] = self.calculate_loss(results, league_params_copy, team_leagues_all, missed_prev,
                                                                      changed,
                                                                      same,
                                                                      indexes_for_update)

                optimal_params = min(params_loss, key=params_loss.get)

                if params_loss[optimal_params] < current_loss:
                    optimal_params_dict = dict(zip(league_params_copy[league], optimal_params))
                    league_params[league] = optimal_params_dict

                    print(league, "Loss down by:", round(current_loss - params_loss[optimal_params], 3))
                    print(optimal_params_dict)

                    current_loss = params_loss[optimal_params]

                    joblib.dump(league_params, Config().project_path + Config().ratings_paths['league_params'])

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

        ratings_df['is_europe'] = np.where(ratings_df['league'].isin(self.south_america_leagues), False, True)

        return ratings_df

    def league_ratings(self, ratings: pd.DataFrame, number_top_teams=50) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                           .sort_values(['rating'], ascending=False))

        europe_leagues_ratings = leagues_ratings.loc[~leagues_ratings['league'].isin(self.south_america_leagues)]
        south_america_leagues_ratings = leagues_ratings.loc[leagues_ratings['league'].isin(self.south_america_leagues)]

        europe_leagues_ratings = (europe_leagues_ratings
                                  .reset_index(drop=True)
                                  .reset_index()
                                  .rename(columns={'index': '#'}))

        europe_leagues_ratings['#'] = (europe_leagues_ratings['#'] + 1)

        south_america_leagues_ratings = (south_america_leagues_ratings
                                         .reset_index(drop=True)
                                         .reset_index()
                                         .rename(columns={'index': '#'}))

        south_america_leagues_ratings['#'] = (south_america_leagues_ratings['#'] + 1)

        return europe_leagues_ratings, south_america_leagues_ratings
