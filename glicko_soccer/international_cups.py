from itertools import product
from typing import List

import numpy as np
import pandas as pd

from glicko2 import Glicko2, Rating
from utils.metrics import match_log_loss


class InternationalCups(object):
    """
        Add a new parameter "init_mu" which means the level of a league.
    """

    def __init__(self, is_draw_mode=True, is_prev_season_init=False, init_mu=1500, init_mu_top=1600):
        self.is_draw_mode = is_draw_mode
        self.is_prev_season_init = is_prev_season_init
        self.init_mu = init_mu
        self.init_mu_top = init_mu_top

        self.international_cups = ['Europe UEFA Europa League',
                                   'Europe UEFA Champions League',
                                   'South America Copa Libertadores',
                                   'South America Copa Sudamericana']
        self.first_leagues = ['Spain Primera División',
                              'England Premier League',
                              'Germany Bundesliga',
                              'Italy Serie A',
                              'France Ligue 1',
                              'Russia Premier League',
                              'Portugal Primeira Liga',
                              'Belgium First Division A',
                              'Netherlands Eredivisie',
                              'Ukraine FAVBET Liga',
                              'Turkey Süper Lig',
                              'Austria Bundesliga',
                              'Denmark Superliga',
                              'Scotland Premiership',
                              'Czech Republic Fortuna Liga',
                              'Cyprus 1. Division',
                              'Greece Super League',
                              'Switzerland Super League',
                              'Serbia Super Liga',
                              'Croatia 1. HNL',
                              'Sweden Allsvenskan',
                              'Norway Eliteserien',
                              "Israel Ligat ha'Al",
                              'Kazakhstan Premier League',
                              'Belarus Premier League',
                              'Azerbaijan Premyer Liqa',
                              'Bulgaria First League',
                              'Romania Liga I',
                              'Poland Ekstraklasa',
                              'Slovakia Super Liga',
                              'Brazil Serie A',
                              'Argentina Superliga',
                              'Uruguay Primera División',
                              'Colombia Primera A',
                              'Mexico Liga MX',
                              'Chile Primera División',
                              ]
        self.uefa_leagues = ['Europe UEFA Champions League',
                             'Europe UEFA Europa League',
                             'Spain Primera División',
                             'Spain Segunda División',
                             'England Premier League',
                             'England Championship',
                             'Germany Bundesliga',
                             'Germany 2. Bundesliga',
                             'Italy Serie A',
                             'Italy Serie B',
                             'France Ligue 1',
                             'France Ligue 2',
                             'Russia Premier League',
                             'Russia FNL',
                             'Portugal Primeira Liga',
                             'Portugal Segunda Liga',
                             'Belgium First Division A',
                             'Netherlands Eredivisie',
                             'Ukraine FAVBET Liga',
                             'Turkey Süper Lig',
                             'Austria Bundesliga',
                             'Denmark Superliga',
                             'Scotland Premiership',
                             'Czech Republic Fortuna Liga',
                             'Cyprus 1. Division',
                             'Greece Super League',
                             'Switzerland Super League',
                             'Serbia Super Liga',
                             'Croatia 1. HNL',
                             'Sweden Allsvenskan',
                             'Norway Eliteserien',
                             "Israel Ligat ha'Al",
                             'Kazakhstan Premier League',
                             'Belarus Premier League',
                             'Azerbaijan Premyer Liqa',
                             'Bulgaria First League',
                             'Romania Liga I',
                             'Poland Ekstraklasa',
                             'Slovakia Super Liga']
        self.uefa_and_first_leagues = ['Europe UEFA Champions League',
                                       'Europe UEFA Europa League',
                                       'Spain Primera División',
                                       'England Premier League',
                                       'Germany Bundesliga',
                                       'Italy Serie A',
                                       'France Ligue 1',
                                       'Russia Premier League',
                                       'Portugal Primeira Liga',
                                       'Belgium First Division A',
                                       'Netherlands Eredivisie',
                                       'Ukraine FAVBET Liga',
                                       'Turkey Süper Lig',
                                       'Austria Bundesliga',
                                       'Denmark Superliga',
                                       'Scotland Premiership',
                                       'Czech Republic Fortuna Liga',
                                       'Cyprus 1. Division',
                                       'Greece Super League',
                                       'Switzerland Super League',
                                       'Serbia Super Liga',
                                       'Croatia 1. HNL',
                                       'Sweden Allsvenskan',
                                       'Norway Eliteserien',
                                       "Israel Ligat ha'Al",
                                       'Kazakhstan Premier League',
                                       'Belarus Premier League',
                                       'Azerbaijan Premyer Liqa',
                                       'Bulgaria First League',
                                       'Romania Liga I',
                                       'Poland Ekstraklasa',
                                       'Slovakia Super Liga'
                                       ]
        self.top_leagues = ['Spain Primera División',
                            'England Premier League',
                            'Germany Bundesliga',
                            'Italy Serie A',
                            'France Ligue 1']

    @staticmethod
    def preprocessing(results: pd.DataFrame) -> pd.DataFrame:
        """"""
        rename_teams = {
            "Mariupol": 'FK Mariupol',
            "Newells Old Boys": "Newels Old Boys",
            "Universidad de Chile": "Universidad Chile",
            "SKA-Khabarovsk": 'SKA Khabarovsk',
            "Ethnikos Achna": "Ethnikos Achnas",
            "AS Eupen": "KAS Eupen",
            "MOL Vidi FC": "MOL Vidi",
            "Progrès Niederkorn": "Progrès Niedercorn",
            "Dinamo Brest": "Dynamo Brest",
            "Independiente FBC": "Independiente",
            "Energetyk-BDU": "Energetyk-BGU",
            "IFK Norrköping": "Norrköping",
            "Slaven Belupo Koprivnica": "Slaven Koprivnica",
            "Zenit St. Petersburg II": "Zenit II",
            "CS Universitatea Craiova": "Universitatea Craiova",
            "Rukh Brest": "Ruh Brest",
            "CSMS Iaşi": "CSM Iaşi",
            "Cova Piedade": "Cova da Piedade",
            "Angers SCO": "Angers",
            "Səbail": "Sebail",
            "Sabail": "Sebail",
            "FC Akhmat": "Akhmat Grozny",
            "CS U Craiova": "Universitatea Craiova",
            "Kuban Krasnodar": "Kuban",
            "Čukarički": "Cukaricki",
            "Akhisarspor": "Akhisar Belediyespor",
            "FCSB": "FCS Bucureşti",
        }

        results['home_team'] = results['home_team'].map(lambda x: x.replace("'", "").strip())
        results['away_team'] = results['away_team'].map(lambda x: x.replace("'", "").strip())

        results['home_team'] = results['home_team'].map(lambda x: rename_teams[x] if x in rename_teams else x)
        results['away_team'] = results['away_team'].map(lambda x: rename_teams[x] if x in rename_teams else x)

        results = results.drop(columns=['home_score', 'away_score']).sort_values(['season', 'date'])

        return results

    def _get_team_leagues(self, results: pd.DataFrame, season: int) -> dict:
        """"""

        team_leagues = (results
                        .loc[(~results['tournament'].isin(self.international_cups))
                             & (results['season'] == season), ['home_team', 'tournament']]
                        .drop_duplicates(['home_team', 'tournament']))

        team_leagues = dict(zip(team_leagues['home_team'], team_leagues['tournament']))

        return team_leagues

    def _get_team_international_cups(self, results: pd.DataFrame, season: int) -> dict:
        """
            For teams get an international tournament in which these teams have participated.
            Important: take a first chronologically tournament,
            because team can be eliminated in some tournament and be transferred to another one.
        """

        international_cups = results.loc[results['tournament'].isin(self.international_cups)
                                         & (results['season'] == season), ['home_team', 'away_team', 'tournament']]

        home_international = international_cups.drop_duplicates(['home_team', 'tournament'], keep='first')
        away_international = international_cups.drop_duplicates(['away_team', 'tournament'], keep='first')

        team_international_cups = dict(zip(home_international['home_team'], home_international['tournament']))
        team_international_cups.update(dict(zip(away_international['away_team'], away_international['tournament'])))

        return team_international_cups

    def _team_params(self, results: pd.DataFrame, season: int, league_params: dict) -> dict:
        """
            league_params are parameters such as home advantage for each league.
            For each team get league params.
        """

        team_leagues = self._get_team_leagues(results, season)
        team_international_cups = self._get_team_international_cups(results, season)

        team_params = {team: league_params[league] for team, league in team_leagues.items()}

        unknown_teams = [team for team in team_international_cups if team not in team_params]

        team_params.update({team: league_params[league] for team, league in team_international_cups.items()
                            if team in unknown_teams})

        return team_params

    @staticmethod
    def _ratings_initialization(team_params: dict) -> Rating():
        """
            Init mu is a level of league.
        """

        ratings = {team: Rating(mu=params['init_mu'], rd=params['init_rd']) for team, params in team_params.items()}

        return ratings

    def calculate_loss(self, results: pd.DataFrame, season: int, league_params: dict) -> float:
        """
            Calculate the value of the loss function
        """

        matches = results.loc[(results['season'] == season)]

        team_params = self._team_params(matches, season, league_params)

        ratings = self._ratings_initialization(team_params)

        log_loss_value = 0
        for row in matches.loc[:, ['home_team', 'away_team', 'outcome']].itertuples():
            outcome, home_team, away_team = row.outcome, row.home_team, row.away_team

            home_params, away_params = team_params[home_team], team_params[away_team]

            # for teams from different championships calculate the average of parameters
            home_advantage = (home_params['home_advantage'] + away_params['home_advantage']) / 2
            draw_inclination = (home_params['draw_inclination'] + away_params['draw_inclination']) / 2

            glicko = Glicko2(is_draw_mode=self.is_draw_mode, draw_inclination=draw_inclination, draw_penalty=0)

            # get current team ratings
            home_rating, away_rating = ratings[home_team], ratings[away_team]

            # update team ratings
            ratings[home_team], ratings[away_team] = glicko.rate(home_rating, away_rating, home_advantage, outcome)

            # calculate outcome probabilities
            win_probability, tie_probability, loss_probability = glicko.probabilities(home_rating, away_rating,
                                                                                      home_advantage)

            log_loss_value += match_log_loss(outcome, win_probability, tie_probability, loss_probability)

        return log_loss_value

    def _remove_matches_between_unknown_teams(self, results: pd.DataFrame, season: int) -> pd.DataFrame:
        """
            Remove matches between teams from leagues we dont know anything about.
        """
        team_leagues = self._get_team_leagues(results, season)
        team_international_cups = self._get_team_international_cups(results, season)

        unknown_teams = [team for team in team_international_cups if team not in team_leagues]

        results['is_home_unknown'] = np.where((results['season'] == season) & results['home_team'].isin(unknown_teams),
                                              1, 0)

        results['is_away_unknown'] = np.where((results['season'] == season) & results['away_team'].isin(unknown_teams),
                                              1, 0)

        results = (results
                   .loc[(results['is_home_unknown'] == 0) | (results['is_away_unknown'] == 0)]
                   .drop(columns=['is_home_unknown', 'is_away_unknown']))

        return results

    def _league_params_initialization(self, league_params: dict, tournaments: List[str],
                                      international_cups: List[str]) -> dict:
        """
            Special parameters initialization for some leagues for acceleration of parameters optimization process.
        """
        league_params = {league: params for league, params in league_params.items() if league in tournaments}

        params = league_params.copy()
        for league, params in params.items():
            if league in self.top_leagues:
                league_params[league] = {'init_mu': self.init_mu_top,
                                         'init_rd': params['init_rd'],
                                         'draw_inclination': params['draw_inclination'],
                                         'home_advantage': params['home_advantage']}
            else:
                league_params[league] = {'init_mu': self.init_mu,
                                         'init_rd': params['init_rd'],
                                         'draw_inclination': params['draw_inclination'],
                                         'home_advantage': params['home_advantage']}

        for league in international_cups:
            league_params[league] = {'init_mu': 1400, 'init_rd': 200, 'draw_inclination': -0.15, 'home_advantage': 40}

        return league_params

    def fit_parameters(self, results: pd.DataFrame, tournaments: List[str], international_cups: List[str],
                       seasons: List[int], league_params: dict, number_iterations: int,
                       is_params_initialization: True) -> dict:
        """"""

        results = self.preprocessing(results)
        results = results.loc[results['tournament'].isin(tournaments) & results['season'].isin(seasons)]

        for season in seasons:
            results = self._remove_matches_between_unknown_teams(results, season)

        if is_params_initialization:
            league_params = self._league_params_initialization(league_params, tournaments, international_cups)

        results = results.sort_values(['season', 'date'])

        for i in range(number_iterations):

            for league, params in league_params.items():

                current_mu = league_params[league]['init_mu']
                current_rd = league_params[league]['init_rd']
                current_ha = league_params[league]['home_advantage']

                mu_domain = np.linspace(current_mu - 2, current_mu + 2, 3)
                rd_domain = np.linspace(current_rd - 2, current_rd + 2, 3)
                ha_domain = np.linspace(current_ha - 1, current_ha + 1, 3)

                parameters_list = list(product(mu_domain, rd_domain, ha_domain))

                parameters_loss = {parameters: 0 for parameters in parameters_list}
                for parameters in parameters_list:
                    init_mu, init_rd, home_advantage = parameters

                    league_params[league] = {'init_mu': init_mu,
                                             'init_rd': init_rd,
                                             'draw_inclination': params['draw_inclination'],
                                             'home_advantage': home_advantage}

                    parameters_loss[parameters] = sum([self.calculate_loss(results, season, league_params)
                                                       for season in seasons])

                optimal_parameters = min(parameters_loss, key=parameters_loss.get)

                optimal_mu, optimal_rd, optimal_ha = optimal_parameters

                league_params[league]['init_mu'] = optimal_mu
                league_params[league]['init_rd'] = optimal_rd
                league_params[league]['home_advantage'] = optimal_ha

        return league_params
