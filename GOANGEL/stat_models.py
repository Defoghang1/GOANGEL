import math
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, PoissonRegressor

GOAL_GRID = 12


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _hda_from_pmf(ph: np.ndarray, pa: np.ndarray) -> np.ndarray:
    n = ph.shape[0]
    joint = ph[:, :, None] * pa[:, None, :]
    idx = np.arange(joint.shape[1])
    home_mask = idx[:, None] > idx[None, :]
    draw_mask = idx[:, None] == idx[None, :]
    home = (joint * home_mask).sum(axis=(1, 2))
    draw = (joint * draw_mask).sum(axis=(1, 2))
    away = 1.0 - home - draw
    out = np.column_stack([home, draw, np.clip(away, 0.0, 1.0)])
    total = out.sum(axis=1, keepdims=True)
    return out / np.where(total > 0, total, 1.0)


class EloRatingModel:
    classes_ = np.array([0, 1, 2])

    def __init__(self, k_factor: float = 22.0, home_advantage: float = 60.0, initial: float = 1500.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial = initial
        self.ratings: Dict[str, float] = {}
        self.scale = 400.0
        self.draw_width = 0.5

    def _update(self, home: str, away: str, gh: int, ga: int) -> float:
        rh = self.ratings.get(home, self.initial)
        ra = self.ratings.get(away, self.initial)
        diff = rh + self.home_advantage - ra
        expected = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        score = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        multiplier = min(1.0 + math.log(max(abs(gh - ga), 1)), 2.5)
        delta = self.k_factor * multiplier * (score - expected)
        self.ratings[home], self.ratings[away] = rh + delta, ra - delta
        return diff

    def _probabilities(self, diff: np.ndarray) -> np.ndarray:
        z = diff / self.scale
        home = _sigmoid(z - self.draw_width)
        away = _sigmoid(-z - self.draw_width)
        out = np.column_stack([home, np.clip(1.0 - home - away, 1e-3, 1.0), away])
        return out / out.sum(axis=1, keepdims=True)

    def fit(self, gh: Sequence[int], ga: Sequence[int], th: Sequence[str], ta: Sequence[str], dates: Any = None) -> "EloRatingModel":
        self.ratings = {}
        diffs = np.array([self._update(h, a, int(g1), int(g2)) for h, a, g1, g2 in zip(th, ta, gh, ga)])
        outcome = np.where(np.asarray(gh) > np.asarray(ga), 0, np.where(np.asarray(gh) == np.asarray(ga), 1, 2))

        def loss(params: np.ndarray) -> float:
            self.scale = float(np.exp(params[0]))
            self.draw_width = float(np.log1p(np.exp(params[1])))
            p = self._probabilities(diffs)
            return float(-np.mean(np.log(np.clip(p[np.arange(len(outcome)), outcome], 1e-9, 1.0))))

        res = minimize(loss, np.array([math.log(400.0), 0.0]), method="Nelder-Mead", options={"maxiter": 200})
        self.scale = float(np.exp(res.x[0]))
        self.draw_width = float(np.log1p(np.exp(res.x[1])))
        return self

    def predict_proba_1x2(self, home: str, away: str) -> np.ndarray:
        diff = self.ratings.get(home, self.initial) + self.home_advantage - self.ratings.get(away, self.initial)
        return self._probabilities(np.array([diff]))[0]


class BayesianDynamicStrengthModel:
    classes_ = np.array([0, 1, 2])

    def __init__(self, prior_var: float = 0.30, process_var: float = 0.003, home_advantage: float = 0.25):
        self.prior_var = prior_var
        self.process_var = process_var
        self.home_advantage = home_advantage
        self.state: Dict[str, List[float]] = {}
        self.base_home = 0.3
        self.base_away = 0.0

    def _team(self, team: str) -> List[float]:
        return self.state.setdefault(team, [0.0, self.prior_var, 0.0, self.prior_var])

    def fit(self, gh: Sequence[int], ga: Sequence[int], th: Sequence[str], ta: Sequence[str], dates: Any = None) -> "BayesianDynamicStrengthModel":
        gh_arr, ga_arr = np.asarray(gh, dtype=float), np.asarray(ga, dtype=float)
        self.base_away = math.log(max(float(ga_arr.mean()), 0.1))
        self.base_home = math.log(max(float(gh_arr.mean()), 0.1))
        self.state = {}
        for home, away, g_h, g_a in zip(th, ta, gh_arr, ga_arr):
            h, a = self._team(home), self._team(away)
            for s in (h, a):
                s[1] += self.process_var
                s[3] += self.process_var
            lam_h = math.exp(self.base_home + h[0] + a[2])
            lam_a = math.exp(self.base_away + a[0] + h[2])
            for state, att_idx, lam, goals in ((h, 0, lam_h, g_h), (a, 2, lam_h, g_h)):
                v_idx = att_idx + 1
                new_var = 1.0 / (1.0 / state[v_idx] + lam)
                state[att_idx] += new_var * (goals - lam)
                state[v_idx] = new_var
            for state, att_idx, lam, goals in ((a, 0, lam_a, g_a), (h, 2, lam_a, g_a)):
                v_idx = att_idx + 1
                new_var = 1.0 / (1.0 / state[v_idx] + lam)
                state[att_idx] += new_var * (goals - lam)
                state[v_idx] = new_var
        return self

    def _lambdas(self, home: str, away: str):
        h = self.state.get(home, [0.0, self.prior_var, 0.0, self.prior_var])
        a = self.state.get(away, [0.0, self.prior_var, 0.0, self.prior_var])
        lam_h = math.exp(self.base_home + h[0] + a[2] + 0.5 * (h[1] + a[3]))
        lam_a = math.exp(self.base_away + a[0] + h[2] + 0.5 * (a[1] + h[3]))
        return float(np.clip(lam_h, 0.1, 6.0)), float(np.clip(lam_a, 0.1, 6.0))

    def predict_proba_1x2(self, home: str, away: str) -> np.ndarray:
        lam_h, lam_a = self._lambdas(home, away)
        k = np.arange(GOAL_GRID)
        return _hda_from_pmf(poisson.pmf(k, lam_h)[None, :], poisson.pmf(k, lam_a)[None, :])[0]


class OrdinalRegressionModel(ClassifierMixin, BaseEstimator):
    def __init__(self, C: float = 0.1, max_iter: int = 2000, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "OrdinalRegressionModel":
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1, 2])
        self.n_features_in_ = int(np.asarray(X).shape[1])
        self.le0_ = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.random_state).fit(X, (y <= 0).astype(int), sample_weight=sample_weight)
        self.le1_ = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.random_state).fit(X, (y <= 1).astype(int), sample_weight=sample_weight)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        q0 = self.le0_.predict_proba(X)[:, 1]
        q1 = np.maximum(self.le1_.predict_proba(X)[:, 1], q0 + 1e-4)
        q1 = np.minimum(q1, 1.0 - 1e-4)
        out = np.column_stack([q0, q1 - q0, 1.0 - q1])
        out = np.clip(out, 1e-4, 1.0)
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class NegativeBinomialRegressionModel(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha: float = 1.0, max_iter: int = 400):
        self.alpha = alpha
        self.max_iter = max_iter

    @staticmethod
    def _dispersion(goals: np.ndarray, mu: np.ndarray) -> float:
        value = float(np.sum((goals - mu) ** 2 - mu) / max(float(np.sum(mu ** 2)), 1e-9))
        return float(np.clip(value, 1e-3, 5.0))

    def fit(self, X: Any, y: Any, gh: Any, ga: Any) -> "NegativeBinomialRegressionModel":
        X = np.asarray(X, dtype=float)
        gh, ga = np.asarray(gh, dtype=float), np.asarray(ga, dtype=float)
        self.classes_ = np.array([0, 1, 2])
        self.n_features_in_ = X.shape[1]
        self.reg_home_ = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter).fit(X, gh)
        self.reg_away_ = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter).fit(X, ga)
        self.disp_home_ = self._dispersion(gh, self.reg_home_.predict(X))
        self.disp_away_ = self._dispersion(ga, self.reg_away_.predict(X))
        return self

    @staticmethod
    def _pmf(mu: np.ndarray, disp: float) -> np.ndarray:
        r = 1.0 / disp
        k = np.arange(GOAL_GRID)[None, :]
        return nbinom.pmf(k, r, r / (r + np.clip(mu, 0.05, 8.0)[:, None]))

    def predict_proba(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        ph = self._pmf(self.reg_home_.predict(X), self.disp_home_)
        pa = self._pmf(self.reg_away_.predict(X), self.disp_away_)
        out = np.clip(_hda_from_pmf(ph, pa), 1e-4, 1.0)
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
