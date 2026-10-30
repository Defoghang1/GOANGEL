import sys
sys.path[:0] = ["."]
import numpy as np
from sklearn.metrics import log_loss
from stat_models import EloRatingModel, BayesianDynamicStrengthModel, OrdinalRegressionModel, NegativeBinomialRegressionModel

def world(n=6000, T=20, seed=0):
    r = np.random.default_rng(seed)
    att, de = r.normal(0, .3, T), r.normal(0, .25, T)
    th, ta, gh, ga = [], [], [], []
    for _ in range(n):
        h, a = r.choice(T, 2, replace=False)
        th.append(f"t{h}"); ta.append(f"t{a}")
        gh.append(int(r.poisson(np.exp(.28 + att[h] - de[a])))); ga.append(int(r.poisson(np.exp(att[a] - de[h]))))
    return th, ta, np.array(gh), np.array(ga)

def outcome(gh, ga):
    return np.where(gh > ga, 0, np.where(gh == ga, 1, 2))

def check_team_model(model):
    th, ta, gh, ga = world()
    c = int(len(th) * .75)
    model.fit(list(gh[:c]), list(ga[:c]), th[:c], ta[:c])
    P = np.array([model.predict_proba_1x2(h, a) for h, a in zip(th[c:], ta[c:])])
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6) and (P >= 0).all()
    y = outcome(gh[c:], ga[c:])
    prior = np.bincount(outcome(gh[:c], ga[:c]), minlength=3) / c
    ll, base = log_loss(y, P, labels=[0, 1, 2]), log_loss(y, np.tile(prior, (len(y), 1)), labels=[0, 1, 2])
    print(type(model).__name__, round(ll, 4), "vs constant", round(base, 4))
    assert ll < base - 0.01

def test_elo_model_beats_constant():
    check_team_model(EloRatingModel())

def test_bayesian_dynamic_strength_beats_constant():
    check_team_model(BayesianDynamicStrengthModel())

def test_unknown_teams_get_valid_probabilities():
    th, ta, gh, ga = world(500)
    for m in (EloRatingModel(), BayesianDynamicStrengthModel()):
        m.fit(list(gh), list(ga), th, ta)
        p = m.predict_proba_1x2("inconnu1", "inconnu2")
        assert abs(p.sum() - 1) < 1e-6 and (p > 0).all()

def feature_world(n=5000, seed=3):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 12))
    mh, ma = np.exp(.25 + .35 * X[:, 0] - .2 * X[:, 1]), np.exp(-.05 - .3 * X[:, 0] + .25 * X[:, 2])
    gh, ga = r.poisson(mh), r.poisson(ma)
    return X, outcome(gh, ga), gh, ga

def test_ordinal_regression_beats_constant():
    X, y, _, _ = feature_world(); c = 3800
    m = OrdinalRegressionModel().fit(X[:c], y[:c])
    P = m.predict_proba(X[c:])
    assert np.allclose(P.sum(axis=1), 1) and list(m.classes_) == [0, 1, 2]
    prior = np.bincount(y[:c], minlength=3) / c
    assert log_loss(y[c:], P, labels=[0, 1, 2]) < log_loss(y[c:], np.tile(prior, (len(y) - c, 1)), labels=[0, 1, 2]) - 0.01

def test_negative_binomial_regression_beats_constant():
    X, y, gh, ga = feature_world(); c = 3800
    m = NegativeBinomialRegressionModel().fit(X[:c], y[:c], gh[:c], ga[:c])
    P = m.predict_proba(X[c:])
    assert np.allclose(P.sum(axis=1), 1) and P.shape == (len(y) - c, 3)
    prior = np.bincount(y[:c], minlength=3) / c
    ll = log_loss(y[c:], P, labels=[0, 1, 2]); base = log_loss(y[c:], np.tile(prior, (len(y) - c, 1)), labels=[0, 1, 2])
    print("NegBinReg", round(ll, 4), "vs", round(base, 4))
    assert ll < base - 0.01
