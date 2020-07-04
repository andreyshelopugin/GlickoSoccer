from math import exp, log, sqrt, pi
from typing import Tuple


class Rating(object):

    def __init__(self, mu=1500., rd=200., volatility=0.0001):
        self.mu = mu
        self.rd = rd
        self.volatility = volatility

    def __repr__(self):
        args = (self.mu, self.rd, self.volatility)
        return '(mu=%.3f, rd=%.3f, volatility=%.3f)' % args


class Glicko2(object):

    def __init__(self, mu=1500., rd=200., tau=1, epsilon=0.000001, is_draw_mode=True, draw_inclination=-0.2,
                 draw_penalty=0.01):
        self.mu = mu
        self.rd = rd
        self.tau = tau
        self.q = log(10) / 400
        self.q1 = 400 / log(10)  # 173.71
        self.epsilon = epsilon
        self.is_draw_mode = is_draw_mode
        self.draw_inclination = draw_inclination
        self.draw_penalty = draw_penalty

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

    @staticmethod
    def _g(phi: float, phi_opp: float) -> float:
        """
            Function g, which is used in step 3 and others.
        """
        g = 1 / sqrt(1 + 1.5 * (phi ** 2 + phi_opp ** 2) / pi ** 2)
        return g

    @staticmethod
    def _expected_score(mu: float, mu_opp: float, g: float) -> float:
        """
            Calculate expected score of game.
            It is used in step 3 and others.
        """

        expected_score = 1 / (1 + exp(-g * (mu - mu_opp)))
        return expected_score

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

    def f(self, rating_improvement: float, v: float, rating: Rating, x: float) -> float:
        """
            Used in step 5.
        """
        phi = rating.rd
        vol = rating.volatility

        temp_var = (phi ** 2 + v + exp(x))

        first_term = exp(x) * (rating_improvement ** 2 - temp_var) / (2 * temp_var ** 2)
        second_term = (x - log(vol ** 2)) / (self.tau ** 2)

        return first_term - second_term

    def _update_volatility(self, rating: Rating, rating_opp: Rating, outcome: float) -> float:
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

    def _update_rating(self, rating: Rating, rating_opp: Rating, outcome: float) -> Rating:
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

    @staticmethod
    def _increase_mu(rating: Rating, addition: float) -> Rating:
        """"""
        return Rating(rating.mu + addition, rating.rd, rating.volatility)

    def rate(self, home_rating: Rating, away_rating: Rating, advantage: float, outcome: str) -> Tuple[Rating, Rating]:
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
            new_increased_home_rating = self._update_rating(increased_home_rating, decreased_away_rating, 1)
            new_decreased_away_rating = self._update_rating(decreased_away_rating, increased_home_rating, 0)

        elif outcome == 'D':
            new_increased_home_rating = self._update_rating(increased_home_rating, decreased_away_rating, 0.5)
            new_decreased_away_rating = self._update_rating(decreased_away_rating, increased_home_rating, 0.5)

        else:
            new_increased_home_rating = self._update_rating(increased_home_rating, decreased_away_rating, 0)
            new_decreased_away_rating = self._update_rating(decreased_away_rating, increased_home_rating, 1)

        # subtract advantage
        updated_home_rating = self._increase_mu(new_increased_home_rating, -advantage)
        updated_away_rating = self._increase_mu(new_decreased_away_rating, advantage)

        return updated_home_rating, updated_away_rating

    def probabilities(self, team: Rating, opp_team: Rating, advantage: float) -> Tuple[float, float, float]:
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

        if self.is_draw_mode:
            draw_penalty = self.draw_penalty * max(mu, mu_opp)

            win_probability = 1 / (1 + exp(-g * delta_mu) + exp(self.draw_inclination - delta_mu - draw_penalty))
            loss_probability = 1 / (1 + exp(g * delta_mu) + exp(self.draw_inclination + delta_mu - draw_penalty))
            tie_probability = (1 - win_probability - loss_probability)

        else:
            win_probability = 1 / (1 + exp(-g * delta_mu))
            loss_probability = (1 - win_probability)
            tie_probability = 0

        return win_probability, tie_probability, loss_probability


# t1 = Rating(mu=1516)
# t2 = Rating(mu=1500)
#
# print(Glicko2().probabilities(t1, t2, 0))