import pandas as pd
import numpy as np
import re
from typing import Tuple


class DataPreprocessor(object):

    @staticmethod
    def _rename_teams():
        """Rename some teams so that their names become identical in both tables"""

        rename_teams = {'1. FC Köln': 'Köln',
                        '1. FC Union Berlin': 'Union Berlin',
                        '1. FSV Mainz 05': 'Mainz 05',
                        'AC Milan': 'Milan',
                        'ACF Fiorentina': 'Fiorentina',
                        'AFC Bournemouth': 'Bournemouth',
                        'AS Monaco': 'Monaco',
                        'AS Roma': 'Roma',
                        'AS Saint-Étienne': 'Saint-Étienne',
                        'Amiens SC': 'Amiens',
                        'Angers SCO': 'Angers',
                        'Arsenal FC': 'Arsenal',
                        'Aston Villa FC': 'Aston Villa',
                        'Atalanta BC': 'Atalanta',
                        'Athletic Club Bilbao': 'Athletic Club',
                        'Bayer 04 Leverkusen': 'Leverkusen',
                        'Bayern München': 'Bayern Munich',
                        'Bologna FC': 'Bologna',
                        'Bor. Mönchengladbach': "M'gladbach",
                        'Borussia Dortmund': 'Dortmund',
                        'Brighton & Hove Albion FC': 'Brighton',
                        'Burnley FC': 'Burnley',
                        'CA Osasuna': 'Osasuna',
                        'CD Leganés': 'Leganés',
                        'Cagliari Calcio': 'Cagliari',
                        'Chelsea FC': 'Chelsea',
                        'Crystal Palace FC': 'Crystal Palace',
                        'Deportivo Alavés': 'Alavés',
                        'Dijon FCO': 'Dijon',
                        'Eintracht Frankfurt': 'Eintracht Frankfurt',
                        'Everton FC': 'Everton',
                        'FC Augsburg': 'Augsburg',
                        'FC Barcelona': 'Barcelona',
                        'FC Internazionale Milano': 'Inter',
                        'FC Metz': 'Metz',
                        'FC Nantes': 'Nantes',
                        'FC Schalke 04': 'Schalke 04',
                        'Fortuna Düsseldorf': 'Düsseldorf',
                        'Genoa CFC': 'Genoa',
                        'Getafe CF': 'Getafe',
                        'Girondins de Bordeaux': 'Bordeaux',
                        'Granada CF': 'Granada',
                        'Hellas Verona FC': 'Hellas Verona',
                        'Leicester City FC': 'Leicester City',
                        'Levante UD': 'Levante',
                        'Lille OSC': 'Lille',
                        'Liverpool FC': 'Liverpool',
                        'Manchester City FC': 'Manchester City',
                        'Manchester United FC': 'Manchester Utd',
                        'Montpellier HSC': 'Montpellier',
                        'Newcastle United FC': 'Newcastle Utd',
                        'Norwich City FC': 'Norwich City',
                        'Nîmes Olympique': 'Nîmes',
                        'OGC Nice': 'Nice',
                        'Olympique Lyonnais': 'Lyon',
                        'Olympique de Marseille': 'Marseille',
                        'Paris Saint-Germain': 'Paris S-G',
                        'RC Celta Vigo': 'Celta Vigo',
                        'RC Strasbourg': 'Strasbourg',
                        'RCD Espanyol': 'Espanyol',
                        'RCD Mallorca': 'Mallorca',
                        'Real Betis': 'Betis',
                        'Real Valladolid CF': 'Valladolid',
                        'SC Freiburg': 'Freiburg',
                        'SC Paderborn 07': 'Paderborn 07',
                        'SD Eibar': 'Eibar',
                        'SS Lazio': 'Lazio',
                        'SSC Napoli': 'Napoli',
                        'Sevilla FC': 'Sevilla',
                        'Sheffield United FC': 'Sheffield Utd',
                        'Southampton FC': 'Southampton',
                        'Stade Brestois 29': 'Brest',
                        'Stade Rennais FC': 'Rennes',
                        'Stade de Reims': 'Reims',
                        'TSG 1899 Hoffenheim': 'Hoffenheim',
                        'Torino FC': 'Torino',
                        'Tottenham Hotspur FC': 'Tottenham',
                        'Toulouse FC': 'Toulouse',
                        'UC Sampdoria': 'Sampdoria',
                        'US Lecce': 'Lecce',
                        'US Sassuolo Calcio': 'Sassuolo',
                        'Udinese Calcio': 'Udinese',
                        'Valencia CF': 'Valencia',
                        'VfL Wolfsburg': 'Wolfsburg',
                        'Villarreal CF': 'Villarreal',
                        'Watford FC': 'Watford',
                        'West Ham United FC': 'West Ham',
                        'Wolverhampton Wanderers FC': 'Wolves'}

        return rename_teams

    def _refactoring(self, results: pd.DataFrame, schedule: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """"""
        results = (results
                   .loc[:, ['Date', 'Country', 'Year', 'Round', 'Team 1', 'Team 2', 'FT Team 1', 'FT Team 2']]
                   .rename(columns={'Date': 'date', 'Round': 'round', 'Country': 'tournament', 'Team 1': 'home_team',
                                    'Team 2': 'away_team', 'Year': 'season',
                                    'FT Team 1': 'home_score', 'FT Team 2': 'away_score'}))

        results['date'] = results['date'].map(lambda x: re.sub(r'\([^)]*\)', '', x))
        results['date'] = pd.to_datetime(results['date'])

        conditions = [(results['home_score'] > results['away_score']),
                      (results['home_score'] == results['away_score']),
                      (results['home_score'] < results['away_score'])]

        outcomes = ['H', 'D', 'A']
        results['outcome'] = np.select(conditions, outcomes)

        renaming_teams = self._rename_teams()
        results['home_team'] = results['home_team'].map(lambda x: x.strip())
        results['away_team'] = results['away_team'].map(lambda x: x.strip())

        results['home_team'] = results['home_team'].map(lambda x: renaming_teams[x] if x in renaming_teams else x)
        results['away_team'] = results['away_team'].map(lambda x: renaming_teams[x] if x in renaming_teams else x)

        schedule = (schedule
                    .loc[schedule['Home'].notna(), ['Date', 'tournament', 'Wk', 'Home', 'Away', 'Score']]
                    .rename(columns={'Date': 'date', 'Wk': 'round', 'Home': 'home_team', 'Away': 'away_team'}))

        schedule['home_score'], schedule['away_score'] = schedule['Score'].fillna('–').str.split('–', 1).str

        schedule = schedule.drop(columns=['Score'])

        conditions = [(schedule['home_score'] > schedule['away_score']),
                      (schedule['home_score'] == schedule['away_score']) & (schedule['home_score'] != ''),
                      (schedule['home_score'] < schedule['away_score'])]

        schedule['outcome'] = np.select(conditions, outcomes, default='F')

        return results, schedule

    @staticmethod
    def _filter_results(results: pd.DataFrame) -> pd.DataFrame:
        """"""
        results = (results
                   .loc[(results['season'] >= 2009) & (results['season'] < 2019)]
                   .sort_values(['tournament', 'date']))

        return results

    def transform(self, results: pd.DataFrame, schedule: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        results, schedule = self._refactoring(results, schedule)
        results = self._filter_results(results)
        return results, schedule
