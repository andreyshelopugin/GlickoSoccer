from sklearn.metrics import log_loss
from math import log


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


def three_outcomes_log_loss(win_probability: float, tie_probability: float,
                            loss_probability: float, outcome: str) -> float:
    """
        Calculate log loss value of one match.
    """

    if outcome == 'H':
        return - (log(win_probability) + log(1 - tie_probability) + log(1 - loss_probability)) / 3

    elif outcome == 'D':
        return - (log(1 - win_probability) + log(tie_probability) + log(1 - loss_probability)) / 3

    else:
        return - (log(1 - win_probability) + log(1 - tie_probability) + log(loss_probability)) / 3


def two_outcomes_log_loss(win_probability: float, loss_probability: float, outcome: int = 1) -> float:
    """
        Calculate log loss value of one match.
    """

    if outcome == 1:
        return - (log(win_probability) + log(1 - loss_probability)) / 2

    else:
        return - (log(1 - win_probability) + log(loss_probability)) / 2
