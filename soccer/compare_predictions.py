import joblib

from config import Config
from soccer.glicko_soccer import GlickoSoccer
from soccer.outcomes_catboost import OutcomesCatBoost
from soccer.outcomes_features import TrainCreator
from soccer.preprocessing import DataPreprocessor
from utils.metrics import three_outcomes_log_loss


def compare_predictions(start_season: int):
    """Compares models loss function values."""
    matches = DataPreprocessor(is_actual_draw_predictions=False).preprocessing()

    # catboost predictions
    _, _, test = TrainCreator().train_validation_test(matches)

    catboost_predictions = OutcomesCatBoost().predict(test)

    catboost_predictions = (catboost_predictions
                            .rename(columns={'home_win': 'home_win_cb', 'draw': 'draw_cb', 'away_win': 'away_win_cb'})
                            .drop(columns=['home_goals', 'away_goals']))

    # glicko predictions
    league_params = joblib.load(Config().project_path + Config().ratings_paths['league_params'])

    glicko_predictions = GlickoSoccer().predictions(matches, league_params, start_season)

    glicko_predictions = glicko_predictions.rename(columns={'home_win': 'home_win_glicko',
                                                            'draw': 'draw_glicko',
                                                            'away_win': 'away_win_glicko'})
    # compare losses
    predictions = (matches
                   .loc[matches['season'] >= start_season, ['match_id', 'season', 'date', 'home_team', 'away_team', 'outcome']]
                   .merge(catboost_predictions, how='inner', on=['match_id'])
                   .merge(glicko_predictions, how='inner', on=['match_id']))

    catboost_loss = 0
    glicko_loss = 0
    for row in predictions.itertuples():
        catboost_loss += three_outcomes_log_loss(row.home_win_cb, row.draw_cb, row.away_win_cb, row.outcome)
        glicko_loss += three_outcomes_log_loss(row.home_win_glicko, row.draw_glicko, row.away_win_glicko, row.outcome)

    print("Catboost Loss:", catboost_loss / predictions.shape[0])
    print("Glicko Loss:", glicko_loss / predictions.shape[0])

    return predictions
