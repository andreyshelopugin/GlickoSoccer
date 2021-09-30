from itertools import product
from typing import Tuple

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

        self.first_leagues = {'Albania. Superliga',
                              'Andorra. Primera Divisió',
                              'Armenia. Premier League',
                              'Austria. Admiral Bundesliga',
                              'Azerbaijan. Premier League',
                              'Belarus. Vysshaya Liga',
                              'Belgium. Jupiler Pro League',
                              'Bosnia And Herzegovina. Premier League',
                              'Bulgaria. Parva liga',
                              'Croatia. 1. HNL',
                              'Cyprus. First Division',
                              'Czech Republic. 1. Liga',
                              'Denmark. Superliga',
                              'England. Premier League',
                              'Estonia. Meistriliiga',
                              'Faroe Islands. Premier League',
                              'Finland. Veikkausliiga',
                              'France. Ligue 1',
                              'Georgia. Crystalbet Erovnuli Liga',
                              'Germany. Bundesliga',
                              'Gibraltar. National League',
                              'Greece. Super League',
                              'Hungary. OTP Bank Liga',
                              'Iceland. Pepsideild',
                              'Ireland. Premier Division',
                              "Israel. Ligat ha'Al",
                              'Italy. Serie A',
                              'Kazakhstan. Premier League',
                              'Kosovo. Superliga',
                              'Latvia. Optibet Virsliga',
                              'Lithuania. A Lyga',
                              'Luxembourg. National Division',
                              'Malta. Premier League',
                              'Moldova. Divizia Nationala',
                              'Montenegro. Prva Crnogorska Liga',
                              'Netherlands. Eredivisie',
                              'North Macedonia. 1. MFL',
                              'Northern Ireland. NIFL Premiership',
                              'Norway. Eliteserien',
                              'Poland. Ekstraklasa',
                              'Portugal. Primeira Liga',
                              'Romania. Liga 1',
                              'Russia. Premier League',
                              'San Marino. Campionato Sammarinese',
                              'Scotland. Premiership',
                              'Serbia. Super Liga',
                              'Slovakia. Fortuna liga',
                              'Slovenia. Prva liga',
                              'Spain. LaLiga',
                              'Sweden. Allsvenskan',
                              'Switzerland. Super League',
                              'Turkey. Super Lig',
                              'Ukraine. Premier League',
                              'Uzbekistan. Super League',
                              'Wales. Cymru Premier'}
        self.second_leagues = {}
        self.championships = self.first_leagues.union(self.second_leagues)
        self.cups = {'Wales. FA Cup',
                     'Northern Ireland. Irish Cup',
                     'Andorra. Andorra Cup',
                     'Israel. State Cup',
                     'Azerbaijan. Azerbaijan Cup',
                     'Serbia. Serbian Cup',
                     'Montenegro. Montenegrin Cup',
                     'Norway. NM Cup',
                     'England. EFL Cup',
                     'Cyprus. Cyprus Cup',
                     'Greece. Greek Cup',
                     'Gibraltar. Gibraltar Cup',
                     'Turkey. Turkish Cup',
                     'Belgium. Belgian Cup',
                     'Switzerland. Swiss Cup',
                     'Scotland. Scottish Cup',
                     'Albania. Albanian Cup',
                     'Croatia. Croatian Cup',
                     'Finland. Suomen Cup',
                     'Georgia. Georgian Cup',
                     'Denmark. Landspokal Cup',
                     'Luxembourg. Luxembourg Cup',
                     'Sweden. Svenska Cupen',
                     'Russia. Russian Cup',
                     'Hungary. Hungarian Cup',
                     'Poland. Polish Cup',
                     'Bosnia And Herzegovina. Bosnia and Herzegovina Cup',
                     'Bulgaria. Bulgarian Cup',
                     'Armenia. Armenian Cup',
                     'North Macedonia. Macedonian Cup',
                     'Belarus. Belarusian Cup',
                     'Uzbekistan. Uzbekistan Cup',
                     'Moldova. Moldovan Cup',
                     'Romania. Romanian Cup',
                     'Ukraine. Ukrainian Cup',
                     'Czech Republic. MOL Cup',
                     'Slovenia. Slovenian Cup',
                     'Ireland. FAI Cup',
                     'Faroe Islands. Faroe Islands Cup',
                     'Latvia. Latvian Cup',
                     'Slovakia. Slovak Cup',
                     'Austria. OFB Cup',
                     'Lithuania. Lithuanian Cup',
                     'Kazakhstan. Kazakhstan Cup',
                     'Estonia. Estonian Cup',
                     'Iceland. Icelandic Cup',
                     'Kosovo. Kosovar Cup'}
        self.euro_cups = {'Champions League', 'Europa League'}  # !!!!
        self.top_leagues = {}

    def _preprocessing(self, results: pd.DataFrame) -> pd.DataFrame:
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
            "Gaziantep": "Gazişehir",
            "Hapoel Hadera": "Hapoel Eran Hadera",
            "Odd": "Odds",
            "Pafos": "Paphos",
            "Spartak Moskva II": "Spartak II",
            "Gaziantep BB": "Gazişehir",
            "Olimpiyets": "Nizhny Novgorod",
            "Terek Grozny": "Akhmat Grozny"
        }

        results['home_team'] = results['home_team'].map(lambda x: x.replace("'", "").strip())
        results['away_team'] = results['away_team'].map(lambda x: x.replace("'", "").strip())

        results['home_team'] = results['home_team'].map(lambda x: rename_teams[x] if x in rename_teams else x)
        results['away_team'] = results['away_team'].map(lambda x: rename_teams[x] if x in rename_teams else x)

        results = results.loc[results['tournament'].isin(self.first_leagues)
                              | results['tournament'].isin(self.second_leagues)
                              | results['tournament'].isin(self.cups)
                              | results['tournament'].isin(self.euro_cups)]

        conditions = [(results['home_score'] > results['away_score']),
                      (results['home_score'] == results['away_score']),
                      (results['home_score'] < results['away_score'])]

        outcomes = ['H', 'D', 'A']
        results['outcome'] = np.select(conditions, outcomes)

        min_season = results['season'].min()
        results['season'] = np.where(results['tournament'].isin(self.playoff), results['season'] - 1, results['season'])
        results = results.loc[results['season'] >= min_season]

        results = results.drop(columns=['home_score', 'away_score']).sort_values(['date'])
        return results

    def _team_leagues(self, results: pd.DataFrame, season: int) -> dict:
        """"""
        team_leagues = (results
                        .loc[results['tournament'].isin(self.championships)
                             & (results['season'] == season), ['home_team', 'tournament']]
                        .drop_duplicates(['home_team']))

        team_leagues = dict(zip(team_leagues['home_team'], team_leagues['tournament']))

        away_team_leagues = (results
                             .loc[results['tournament'].isin(self.championships)
                                  & (results['season'] == season), ['away_team', 'tournament']]
                             .drop_duplicates(['away_team']))

        team_leagues.update(dict(zip(away_team_leagues['away_team'], away_team_leagues['tournament'])))

        return team_leagues

    def _team_international_cups(self, results: pd.DataFrame, season: int) -> dict:
        """
            For teams get an international tournament in which these teams have participated.
            Important: take a first chronologically tournament,
            because team can be eliminated in some tournament and be transferred to another one.
        """

        international_cups = (results
                              .loc[results['tournament'].isin(self.euro_cups) & (results['season'] == season)]
                              .sort_values(['date']))

        home_international = international_cups.drop_duplicates(['home_team'], keep='first')
        away_international = international_cups.drop_duplicates(['away_team'], keep='first')

        team_international_cups = dict(zip(home_international['home_team'], home_international['tournament']))
        team_international_cups.update(dict(zip(away_international['away_team'], away_international['tournament'])))

        return team_international_cups

    def _remove_matches_with_unknown_team(self, results: pd.DataFrame) -> pd.DataFrame:
        """Remove matches between teams from leagues we dont know anything about."""

        for season in results['season'].unique():
            team_leagues = self._team_leagues(results, season)

            known_teams = team_leagues.keys()

            is_championship = ((results['season'] == season)
                               & results['tournament'].isin(self.championships))

            is_one_team_known_in_eurocups = ((results['season'] == season)
                                             & results['tournament'].isin(self.euro_cups)
                                             & (results['home_team'].isin(known_teams)
                                                | results['away_team'].isin(known_teams)))

            is_both_teams_known_in_cups = ((results['season'] == season)
                                           & (results['tournament'].isin(self.cups)
                                              | results['tournament'].isin(self.playoff))
                                           & results['home_team'].isin(known_teams)
                                           & results['away_team'].isin(known_teams))

            results = results.loc[(results['season'] != season)
                                  | is_championship
                                  | is_one_team_known_in_eurocups
                                  | is_both_teams_known_in_cups]

        return results

    def _league_params_initialization(self) -> dict:
        """"""
        leagues = (set(self.championships) | set(self.euro_cups))

        league_params = dict()
        for league in leagues:
            league_params[league] = {'init_mu': self.init_mu,
                                     'init_rd': self.init_rd,
                                     'update_rd': self.update_rd,
                                     'lift_update_mu': self.lift_update_mu,
                                     'lift_update_rd': self.lift_update_rd,
                                     'home_advantage': self.home_advantage,
                                     'draw_inclination': self.draw_inclination,
                                     'cup_penalty': self.cup_penalty,
                                     'new_team_update_mu': self.new_team_update_mu,
                                     }

        return league_params

    def _team_params(self, results: pd.DataFrame, season: int, league_params: dict) -> dict:
        """
            For each team get league params.
        """

        team_leagues = self._team_leagues(results, season)
        team_international_cups = self._team_international_cups(results, season)

        team_params = {team: league_params[league] for team, league in team_leagues.items()}

        unknown_teams = [team for team in team_international_cups if team not in team_params]

        team_params.update({team: league_params[league] for team, league in team_international_cups.items()
                            if team in unknown_teams})
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

    def _first_league_seasons(self, results: pd.DataFrame) -> dict:
        """"""
        first_seasons = (results
                         .loc[results['tournament'].isin(self.championships), ['tournament', 'season']]
                         .sort_values(['tournament', 'season'])
                         .drop_duplicates(['tournament'], keep='first'))

        first_seasons = dict(zip(first_seasons['tournament'], first_seasons['season']))
        return first_seasons

    def _update_ratings_indexes(self, results: pd.DataFrame) -> Tuple[dict, dict, dict]:
        """"""
        no_cups = results.loc[~results['tournament'].isin(self.cups)]

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

    def _season_update_rating(self, ratings: dict, results: pd.DataFrame, season: int, league_params: dict,
                              first_league_seasons: dict) -> dict:
        """"""

        team_leagues = self._team_leagues(results, season)
        prev_team_leagues = self._team_leagues(results, season - 1)
        team_params = self._team_params(results, season, league_params)

        for team, league in team_leagues.items():
            params = team_params[team]

            if team in prev_team_leagues:
                prev_league = prev_team_leagues[team]

                # if a team didnt change a league update rating deviation
                if league == prev_league:
                    ratings.update({team: Rating(mu=ratings[team].mu,
                                                 rd=ratings[team].rd + params['update_rd'])})

                # if a team changed a league update rating mu and rating deviation
                else:
                    ratings.update({team: Rating(mu=ratings[team].mu + params['lift_update_mu'],
                                                 rd=ratings[team].rd + params['lift_update_rd'])})

            # if a team is back to a league initialize rating again
            else:
                if season > first_league_seasons[league]:
                    ratings.update({team: Rating(mu=params['init_mu'] + params['new_team_update_mu'],
                                                 rd=params['init_rd'])})

        # teams from unknown leagues need to be initialized every season
        team_international_cups = self._team_international_cups(results, season)

        for team, eurocup in team_international_cups.items():
            if team not in team_leagues:
                ratings[team] = Rating(mu=team_params[team]['init_mu'], rd=team_params[team]['init_rd'])

        return ratings

    def calculate_loss(self, results: pd.DataFrame, league_params: dict, first_league_seasons: dict,
                       team_leagues: dict, eurocups_teams: dict) -> float:
        """"""

        seasons = sorted(results['season'].unique())
        min_season = min(seasons)

        ratings = self._rating_initialization(results, league_params)

        log_loss_value = 0
        for season in seasons:

            team_params = self._team_params(results, season, league_params)
            season_team_leagues = team_leagues[season]
            season_eurocups_teams = eurocups_teams[season]

            if season != min_season:
                ratings = self._season_update_rating(ratings, results, season, league_params, first_league_seasons)

            matches = results.loc[results['season'] == season]

            for row in matches.itertuples():

                outcome, home_team, away_team, tournament = row.outcome, row.home_team, row.away_team, row.tournament

                home_params, away_params = team_params[home_team], team_params[away_team]

                # for teams from different championships calculate the average of parameters
                home_advantage = home_params['home_advantage']
                draw_inclination = (home_params['draw_inclination'] + away_params['draw_inclination']) / 2

                glicko = Glicko2(is_draw_mode=self.is_draw_mode, draw_inclination=draw_inclination)

                if (tournament in self.cups) and (season_team_leagues[home_team] != season_team_leagues[away_team]):
                    if home_team in season_eurocups_teams:
                        home_advantage -= home_params['cup_penalty']
                    elif away_team in season_eurocups_teams:
                        home_advantage += away_params['cup_penalty']

                # get current team ratings
                home_rating, away_rating = ratings[home_team], ratings[away_team]

                # update team ratings
                ratings[home_team], ratings[away_team] = glicko.rate(home_rating, away_rating, home_advantage, outcome)

                if season != min_season:

                    reg = (season - min_season)

                    # calculate outcome probabilities
                    win_probability, tie_probability, loss_probability = glicko.probabilities(home_rating, away_rating,
                                                                                              home_advantage)
                    if tournament not in self.cups:
                        log_loss_value += reg * three_outcomes_log_loss(outcome, win_probability, tie_probability,
                                                                        loss_probability)

                    else:
                        log_loss_value += reg * three_outcomes_log_loss(outcome, win_probability, tie_probability,
                                                                        loss_probability) / 2
        return log_loss_value

    def fit_params(self, results: pd.DataFrame, number_iterations: int, is_params_initialization: True):
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
        # !!!!!!!!!!!!
        avg_scoring = 2.5

        seasons = sorted(results['season'].unique())
        min_season = min(seasons)

        ratings = self._rating_initialization(results, league_params)

        first_league_seasons = self._first_league_seasons(results)
        team_leagues = {season: self._team_leagues(results, season) for season in results['season'].unique()}

        eurocups_teams = dict()
        for season in results['season'].unique():
            season_team_leagues = team_leagues[season]

            teams = self._team_international_cups(results, season)
            teams = [team for team in teams.keys() if team in season_team_leagues]
            eurocups_teams[season] = [team for team in teams if season_team_leagues[team] in self.top_leagues]

        for season in seasons:

            team_params = self._team_params(results, season, league_params)
            season_team_leagues = team_leagues[season]
            season_eurocups_teams = eurocups_teams[season]

            if season != min_season:
                ratings = self._season_update_rating(ratings, results, season, league_params, first_league_seasons)

            matches = results.loc[results['season'] == season]

            for row in matches.itertuples():

                outcome, home_team, away_team, tournament = row.outcome, row.home_team, row.away_team, row.tournament

                home_params, away_params = team_params[home_team], team_params[away_team]

                # for teams from different championships calculate the average of parameters
                home_advantage = home_params['home_advantage']
                draw_inclination = (home_params['draw_inclination'] + away_params['draw_inclination']) / 2

                glicko = Glicko2(draw_inclination=draw_inclination)

                if (tournament in self.cups) and (season_team_leagues[home_team] != season_team_leagues[away_team]):
                    if home_team in season_eurocups_teams:
                        home_advantage -= home_params['cup_penalty']
                    elif away_team in season_eurocups_teams:
                        home_advantage += away_params['cup_penalty']

                # get current team ratings
                home_rating, away_rating = ratings[home_team], ratings[away_team]

                # update team ratings
                ratings[home_team], ratings[away_team] = glicko.rate(home_rating, away_rating, home_advantage, outcome,
                                                                     avg_scoring)

        return ratings

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
