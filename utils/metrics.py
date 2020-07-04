from sklearn.metrics import log_loss


def match_log_loss(outcome: str, win_probability: float, tie_probability: float, loss_probability: float) -> float:
    """
        Calculate log loss value of one match.
    """
    predict = [win_probability, tie_probability, loss_probability]

    if outcome == 'H':
        target = [1, 0, 0]

    elif outcome == 'D':
        target = [0, 1, 0]

    else:
        target = [0, 0, 1]

    return log_loss(target, predict)

