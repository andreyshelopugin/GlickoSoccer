from math import exp, log, sqrt, pi
from typing import Tuple

from dataclasses import dataclass


@dataclass
class Rating:
    mu: float = 1500.
    rd: float = 200.
    volatility: float = 0.06


class Glicko2(object):

    def __init__(self, mu=1500., rd=200., tau=1, epsilon=0.000001, draw_inclination=-0.2):
        self.mu = mu
        self.rd = rd
        self.tau = tau
        self.q = log(10) / 400
        self.q1 = 400 / log(10)  # 173.71
        self.g_constant = 1.5 / pi ** 2
        self.epsilon = epsilon
        self.draw_inclination = draw_inclination

    def _convert_into_glicko2(self, rating: Rating) -> Rating:
        """
            Step 2.
            Convert the ratings and RD’s onto the Glicko-2 scale.
        """
        mu = ((rating.mu - self.mu) * self.q)
        phi = (rating.rd * self.q)
        return Rating(mu, phi, rating.volatility)

    def _convert_from_glicko2(self, rating: Rating) -> Rating:
        """
            Step 8. Convert ratings and RD’s back to original scale.
        """
        mu = (rating.mu * self.q1 + self.mu)
        phi = (rating.rd * self.q1)
        return Rating(mu, phi, rating.volatility)

    def _g(self, phi: float, phi_opp: float) -> float:
        """
            Function g, which is used in step 3 and others.
        """
        # g = 1 / sqrt(1 + 1.5 * (phi ** 2 + phi_opp ** 2) / pi ** 2)
        g = 1 / sqrt(1 + self.g_constant * (phi ** 2 + phi_opp ** 2))
        return g

    def _expected_score(self, mu: float, mu_opp: float, g: float, skellam_draw_probability: float) -> float:
        """
            Calculate expected score of game.
            It is used in step 3 and others.
        """

        delta_mu = g * (mu - mu_opp)

        draw_coefficient = 1 + exp(self.draw_inclination + skellam_draw_probability)
        exp_delta_mu = exp(delta_mu)

        win_probability = 1 / (1 + draw_coefficient / exp_delta_mu)
        loss_probability = 1 / (1 + draw_coefficient * exp_delta_mu)
        tie_probability = (1 - win_probability - loss_probability)

        return win_probability + 0.5 * tie_probability

    @staticmethod
    def _v(g: float, expected_score: float) -> float:
        """
            Step 3.
            Compute the quantity rating variance v. This is the estimated variance of the team’s/player’s
            rating based only on game outcomes.
        """
        v = 1 / (g ** 2 * expected_score * (1 - expected_score))
        return v

    @staticmethod
    def _rating_improvement(v: float, g: float, outcome: float, expected_score: float) -> float:
        """
            Step 4.
            Compute the quantity ∆, the estimated improvement in rating by comparing the
            pre-period rating to the performance rating based only on game outcomes.
        """
        delta = (v * g * (outcome - expected_score))
        return delta

    @staticmethod
    def f(rating_improvement: float, v: float, rating: Rating, x: float) -> float:
        """
            Used in step 5.
        """
        exp_x = exp(x)

        temp_var = (rating.rd ** 2 + v + exp_x)

        first_term = exp_x * (rating_improvement ** 2 - temp_var) / (2 * temp_var ** 2)

        # second_term = (x - log(vol ** 2)) / (self.tau ** 2)
        second_term = (x - log(rating.volatility ** 2))

        return first_term - second_term

    def _update_volatility(self, rating: Rating, rating_opp: Rating, outcome: float,
                           skellam_draw_probability: float) -> float:
        """
            Step 5.
            Determine the new value σ, of the volatility. This computation requires iterations.
        """
        mu = rating.mu
        phi = rating.rd

        mu_opp = rating_opp.mu
        phi_opp = rating_opp.rd

        g = self._g(phi, phi_opp)
        expected_score = self._expected_score(mu, mu_opp, g, skellam_draw_probability)
        v = self._v(g, expected_score)
        rating_improvement = self._rating_improvement(v, g, outcome, expected_score)

        a = log(rating.volatility ** 2)

        temp = (rating_improvement ** 2 - phi ** 2 - v)
        if temp > 0:
            b = log(temp)
        else:
            k = 1
            while self.f(rating_improvement, v, rating, a - k * self.tau) < 0:
                k += 1

            b = (a - k * self.tau)

        f_a = self.f(rating_improvement, v, rating, a)
        f_b = self.f(rating_improvement, v, rating, b)

        while abs(b - a) > self.epsilon:
            c = a + (a - b) * f_a / (f_b - f_a)
            f_c = self.f(rating_improvement, v, rating, c)

            if f_c * f_b < 0:
                a = b
                f_a = f_b
            else:
                f_a /= 2

            b = c
            f_b = f_c

        new_volatility = exp(a / 2)

        return new_volatility

    def _update_rating(self, rating: Rating, rating_opp: Rating, outcome: float,
                       skellam_draw_probability: float) -> Rating:
        """
            Run all steps.
            Step 6. Update the rating deviation to the new pre-rating period value, φ*.
            Step 7. Update the rating and RD to the new values, µ' and φ'.
            Step 8. Convert ratings and RD’s back to original scale
        """

        rating = self._convert_into_glicko2(rating)
        rating_opp = self._convert_into_glicko2(rating_opp)

        mu = rating.mu
        phi = rating.rd

        mu_opp = rating_opp.mu
        phi_opp = rating_opp.rd

        new_volatility = self._update_volatility(rating, rating_opp, outcome, skellam_draw_probability)

        g = self._g(phi, phi_opp)
        expected_score = self._expected_score(mu, mu_opp, g, skellam_draw_probability)
        v = self._v(g, expected_score)

        # step 6
        squared_phi_star = (phi ** 2 + new_volatility ** 2)

        # step 7
        new_phi = 1 / sqrt((1 / squared_phi_star) + (1 / v))

        new_mu = mu + (new_phi ** 2) * g * (outcome - expected_score)

        new_rating = Rating(new_mu, new_phi, new_volatility)

        # step 8
        new_rating = self._convert_from_glicko2(new_rating)

        return new_rating

    @staticmethod
    def _increase_mu(rating: Rating, addition: float) -> Rating:
        """"""
        return Rating(rating.mu + addition, rating.rd, rating.volatility)

    def rate(self, home_rating: Rating, away_rating: Rating, advantage: float, outcome: str,
             skellam_draw_probability: float) -> Tuple[Rating, Rating]:
        """
            home_rating - rating of home team.
            away_rating - rating of away team.
            advantage - home court, injures, etc. advantage of winning team, can be negative.
        """
        # take into account the advantage
        increased_home_rating = self._increase_mu(home_rating, advantage)
        decreased_away_rating = self._increase_mu(away_rating, -advantage)

        # update ratings
        if outcome == 'H':
            new_increased_home_rating = self._update_rating(increased_home_rating, decreased_away_rating, 1,
                                                            skellam_draw_probability)
            new_decreased_away_rating = self._update_rating(decreased_away_rating, increased_home_rating, 0,
                                                            skellam_draw_probability)

        elif outcome == 'D':
            new_increased_home_rating = self._update_rating(increased_home_rating, decreased_away_rating, 0.5,
                                                            skellam_draw_probability)
            new_decreased_away_rating = self._update_rating(decreased_away_rating, increased_home_rating, 0.5,
                                                            skellam_draw_probability)

        else:
            new_increased_home_rating = self._update_rating(increased_home_rating, decreased_away_rating, 0,
                                                            skellam_draw_probability)
            new_decreased_away_rating = self._update_rating(decreased_away_rating, increased_home_rating, 1,
                                                            skellam_draw_probability)

        # subtract advantage
        updated_home_rating = self._increase_mu(new_increased_home_rating, -advantage)
        updated_away_rating = self._increase_mu(new_decreased_away_rating, advantage)

        return updated_home_rating, updated_away_rating

    def probabilities(self, team: Rating, opp_team: Rating, advantage: float,
                      skellam_draw_probability: float) -> Tuple[float, float, float]:
        """
            Input ratings are in original scale.
            Return the probabilities of outcomes.
        """

        mu = ((team.mu + advantage - self.mu) * self.q)
        mu_opp = ((opp_team.mu - advantage - self.mu) * self.q)

        phi = (team.rd * self.q)
        phi_opp = (opp_team.rd * self.q)

        g = 1 / sqrt(1 + self.g_constant * (phi ** 2 + phi_opp ** 2))

        delta_mu = g * (mu - mu_opp)

        draw_coefficient = 1 + exp(self.draw_inclination + skellam_draw_probability)
        exp_delta_mu = exp(delta_mu)

        win_probability = 1 / (1 + draw_coefficient / exp_delta_mu)
        loss_probability = 1 / (1 + draw_coefficient * exp_delta_mu)
        tie_probability = (1 - win_probability - loss_probability)

        return win_probability, tie_probability, loss_probability


class TwoOutcomesGlicko2(Glicko2):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _expected_score(mu: float, mu_opp: float, g: float) -> float:
        """
            Calculate expected score of game.
            It is used in step 3 and others.
        """

        expected_score = 1 / (1 + exp(-g * (mu - mu_opp)))
        return expected_score

    def _update_volatility(self, rating: Rating, rating_opp: Rating, outcome: int) -> float:
        """
            Step 5.
            Determine the new value σ, of the volatility. This computation requires iterations.
        """
        mu = rating.mu
        phi = rating.rd

        mu_opp = rating_opp.mu
        phi_opp = rating_opp.rd

        g = self._g(phi, phi_opp)
        expected_score = self._expected_score(mu, mu_opp, g)
        v = self._v(g, expected_score)
        rating_improvement = self._rating_improvement(v, g, outcome, expected_score)

        a = log(rating.volatility ** 2)

        temp = (rating_improvement ** 2 - phi ** 2 - v)
        if temp > 0:
            b = log(temp)
        else:
            k = 1
            while self.f(rating_improvement, v, rating, a - k * self.tau) < 0:
                k += 1

            b = (a - k * self.tau)

        f_a = self.f(rating_improvement, v, rating, a)
        f_b = self.f(rating_improvement, v, rating, b)

        while abs(b - a) > self.epsilon:
            c = a + (a - b) * f_a / (f_b - f_a)
            f_c = self.f(rating_improvement, v, rating, c)

            if f_c * f_b < 0:
                a = b
                f_a = f_b
            else:
                f_a /= 2

            b = c
            f_b = f_c

        new_volatility = exp(a / 2)

        return new_volatility

    def _update_rating(self, rating: Rating, rating_opp: Rating, outcome: int) -> Rating:
        """
            Run all steps.
            Step 6. Update the rating deviation to the new pre-rating period value, φ*.
            Step 7. Update the rating and RD to the new values, µ' and φ'.
            Step 8. Convert ratings and RD’s back to original scale
        """

        rating = self._convert_into_glicko2(rating)
        rating_opp = self._convert_into_glicko2(rating_opp)

        mu = rating.mu
        phi = rating.rd

        mu_opp = rating_opp.mu
        phi_opp = rating_opp.rd

        new_volatility = self._update_volatility(rating, rating_opp, outcome)

        g = self._g(phi, phi_opp)
        expected_score = self._expected_score(mu, mu_opp, g)
        v = self._v(g, expected_score)

        # step 6
        squared_phi_star = (phi ** 2 + new_volatility ** 2)

        # step 7
        new_phi = 1 / sqrt((1 / squared_phi_star) + (1 / v))

        new_mu = mu + (new_phi ** 2) * g * (outcome - expected_score)

        new_rating = Rating(new_mu, new_phi, new_volatility)

        # step 8
        new_rating = self._convert_from_glicko2(new_rating)

        return new_rating

    def rate(self, home_rating: Rating, away_rating: Rating, advantage: float, outcome: int = 1) -> Tuple[Rating, Rating]:
        """
            home_rating - rating of home team.
            away_rating - rating of away team.
            advantage - home court, injures, etc. advantage of winning team, can be negative.
        """
        # take into account the advantage
        increased_home_rating = self._increase_mu(home_rating, advantage)
        decreased_away_rating = self._increase_mu(away_rating, -advantage)

        # update ratings
        if outcome == 1:
            new_increased_home_rating = self._update_rating(increased_home_rating, decreased_away_rating, 1)
            new_decreased_away_rating = self._update_rating(decreased_away_rating, increased_home_rating, 0)
        else:
            new_increased_home_rating = self._update_rating(increased_home_rating, decreased_away_rating, 0)
            new_decreased_away_rating = self._update_rating(decreased_away_rating, increased_home_rating, 1)

        # subtract advantage
        updated_home_rating = self._increase_mu(new_increased_home_rating, -advantage)
        updated_away_rating = self._increase_mu(new_decreased_away_rating, advantage)

        return updated_home_rating, updated_away_rating

    def probabilities(self, team: Rating, opp_team: Rating, advantage: float) -> Tuple[float, float]:
        """
            Input ratings are in original scale.
            Return the probabilities of outcomes.
        """

        mu = ((team.mu + advantage - self.mu) * self.q)
        mu_opp = ((opp_team.mu - advantage - self.mu) * self.q)

        phi = (team.rd * self.q)
        phi_opp = (opp_team.rd * self.q)

        g = 1 / sqrt(1 + 1.5 * (phi ** 2 + phi_opp ** 2) / pi ** 2)

        delta_mu = (mu - mu_opp)

        win_probability = 1 / (1 + exp(-g * delta_mu))
        loss_probability = (1 - win_probability)
        return win_probability, loss_probability

