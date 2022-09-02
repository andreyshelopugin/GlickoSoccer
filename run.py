import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from euro_soccer.preprocessing import DataPreprocessor
from euro_soccer.glicko_soccer import GlickoSoccer


matches = pd.read_csv('data/matches.csv')
matches = DataPreprocessor().preprocessing(matches)
matches = GlickoSoccer().preprocessing(matches)

# first_leagues = set(matches.loc[matches['tournament_type'] == 1, 'tournament'])

# init_league_params = GlickoSoccer()._league_params_initialization(matches)
# league_params = joblib.load('data/league_params.pkl')
#
# print(np.mean([v['home_advantage'] for k, v in league_params.items()]))
# print(np.mean([v['pandemic_home_advantage'] for k, v in league_params.items()]))
# print(np.mean([v['init_rd'] for k, v in league_params.items()]))
# print(np.mean([v['update_rd'] for k, v in league_params.items()]))
# print(np.mean([v['lift_update_mu'] for k, v in league_params.items() if k in first_leagues]))
# print(np.mean([v['cup_penalty'] for k, v in league_params.items() if k in first_leagues]))
# print(np.mean([v['lift_update_mu'] for k, v in league_params.items() if k not in first_leagues]))
# print(np.mean([v['new_team_update_mu'] for k, v in league_params.items() if k not in first_leagues]))


# for league, params in league_params.items():
#     if league != 'Armenia. First League':
#         init_league_params[league]['home_advantage'] = params['home_advantage']
#         init_league_params[league]['home_advantage'] = params['home_advantage']


# joblib.dump(league_params, 'data/league_params.pkl')


league_params = GlickoSoccer().fit_params(matches, 10, is_params_initialization=False)





