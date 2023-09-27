import joblib
import pandas as pd

from config import Config
from soccer.glicko_soccer import GlickoSoccer
from soccer.outcomes_catboost import OutcomesCatBoost
from soccer.outcomes_features import TrainCreator
from soccer.outcomes_lgbm import OutcomesLGBM
from soccer.preprocessing import DataPreprocessor
from utils.metrics import three_outcomes_log_loss


def compare_models(start_season: int = 2021) -> pd.DataFrame:
    """Compares the loss function values of the models."""
    matches = DataPreprocessor(is_actual_draw_predictions=False, is_train=True).preprocessing()

    _, _, test = TrainCreator().train_validation_test(matches)

    # catboost predictions
    catboost_predictions = OutcomesCatBoost().predict(test)

    catboost_predictions = (catboost_predictions
                            .rename(columns={'home_win': 'home_win_cb', 'draw': 'draw_cb', 'away_win': 'away_win_cb'})
                            .drop(columns=['home_goals', 'away_goals']))

    # lgbm predictions
    lgbm_predictions = OutcomesLGBM().predict(test)

    lgbm_predictions = (lgbm_predictions
                        .rename(columns={'home_win': 'home_win_lgbm', 'draw': 'draw_lgbm', 'away_win': 'away_win_lgbm'})
                        .drop(columns=['home_goals', 'away_goals']))

    # glicko predictions
    matches = DataPreprocessor(is_actual_draw_predictions=False, is_train=False).preprocessing()

    league_params = joblib.load(Config().ratings_paths['league_params'])

    glicko_predictions = GlickoSoccer().predictions(matches, league_params, start_season)

    glicko_predictions = glicko_predictions.rename(columns={'home_win': 'home_win_glicko',
                                                            'draw': 'draw_glicko',
                                                            'away_win': 'away_win_glicko'})

    # original glicko predictions
    # "disable" all params
    league_params = {league: {'init_mu': 1500,
                              'init_rd': 150,
                              'update_rd': 0,
                              'lift_update_mu': 0,
                              'home_advantage': 0,
                              'pandemic_home_advantage': 0,
                              'new_team_update_mu': 0} for league in league_params}

    train_matches = matches.loc[matches['season'] < start_season]
    mean_draw = train_matches.loc[train_matches['outcome'] == 'D'].shape[0] / train_matches.shape[0]

    matches['draw_probability'] = mean_draw

    original_glicko_predictions = GlickoSoccer().predictions(matches, league_params, start_season)

    original_glicko_predictions = original_glicko_predictions.rename(columns={'home_win': 'home_win_original_glicko',
                                                                              'draw': 'draw_original_glicko',
                                                                              'away_win': 'away_win_original_glicko'})

    # compare losses
    predictions = (matches
                   .loc[matches['season'] >= start_season, ['match_id', 'season', 'date', 'home_team', 'away_team', 'outcome']]
                   .merge(catboost_predictions, how='inner', on='match_id')
                   .merge(lgbm_predictions, how='inner', on='match_id')
                   .merge(glicko_predictions, how='inner', on='match_id')
                   .merge(original_glicko_predictions, how='inner', on='match_id'))

    catboost_loss = 0
    lgbm_loss = 0
    glicko_loss = 0
    original_glicko_loss = 0
    for row in predictions.itertuples():
        catboost_loss += three_outcomes_log_loss(row.home_win_cb, row.draw_cb, row.away_win_cb, row.outcome)
        lgbm_loss += three_outcomes_log_loss(row.home_win_lgbm, row.draw_lgbm, row.away_win_lgbm, row.outcome)
        glicko_loss += three_outcomes_log_loss(row.home_win_glicko, row.draw_glicko, row.away_win_glicko, row.outcome)
        original_glicko_loss += three_outcomes_log_loss(row.home_win_original_glicko, row.draw_original_glicko,
                                                        row.away_win_original_glicko, row.outcome)

    prediction_set_size = predictions.shape[0]

    results = {"model": ["Catboost", "LightGBM", "Modified Glicko-2", "Original Glicko-2"],
               "loss": [catboost_loss / prediction_set_size, lgbm_loss / prediction_set_size,
                        glicko_loss / prediction_set_size, original_glicko_loss / prediction_set_size]}

    return pd.DataFrame(results)
