import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from euro_soccer.preprocessing import DataPreprocessor
from euro_soccer.glicko_soccer import GlickoSoccer


matches = pd.read_csv('data/matches.csv')
matches = DataPreprocessor().preprocessing(matches)
matches = GlickoSoccer().preprocessing(matches)


# init_league_params = GlickoSoccer()._league_params_initialization(matches)
league_params = joblib.load('data/league_params.pkl')

print(league_params)


# for league, params in league_params.items():
#     if league != 'Armenia. First League':
#         init_league_params[league]['home_advantage'] = params['home_advantage']
#         init_league_params[league]['home_advantage'] = params['home_advantage']

# league_params.pop('Uzbekistan. Super League', None)
#
# joblib.dump(league_params, 'data/league_params.pkl')


# league_params = GlickoSoccer().fit_params(matches, 10, is_params_initialization=True)





