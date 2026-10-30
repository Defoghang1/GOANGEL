import numpy as np
from goangel_core import GoPreprocessor, fit_guarded_calibrator, safe_log_loss, fit_lambdas_to_hda, devig, blend_with_market

def synth(n=9000, p=251, n_sig=30, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, p))
    w = np.zeros(p); w[:n_sig] = r.normal(0, 0.45, n_sig)
    z = X @ w
    logits = np.column_stack([z + .2, 0 * z, -z])
    P = np.exp(logits); P /= P.sum(1, keepdims=True)
    y = np.array([r.choice(3, p=pp) for pp in P])
    X[:, 200:] = X[:, :51] + 0.01 * r.normal(size=(n, 51))
    X[r.random(X.shape) < 0.15] = np.nan
    X[:, 200:] = np.where(np.isnan(X[:, :51]), np.nan, X[:, 200:])
    X[:, 150:170] = np.where(r.random((n, 20)) < 0.9, np.nan, X[:, 150:170])
    return X, y, w

def test_preprocessor_dimension_contract():
    X, y, w = synth(2000)
    prep = GoPreprocessor(5.0, selected_idx=[3, 10, 20]).fit(X[:1500])
    assert prep.n_features_in_ == 251
    assert prep.transform(X[1500:]).shape[1] == 3
    assert prep.transform_raw(X[1500:]).shape[1] == 3
    one = prep.transform(X[:1]); assert one.shape == (1, 3)

def test_calibrator_detects_real_miscalibration_and_rejects_noise():
    r = np.random.default_rng(1)
    n = 3000
    true = r.dirichlet([2, 1.5, 2], n)
    y = np.array([r.choice(3, p=p) for p in true])
    over = true ** 2.2; over /= over.sum(1, keepdims=True)
    cal, status, info = fit_guarded_calibrator(over, y)
    print("overconfident ->", status, info)
    assert cal is not None
    assert safe_log_loss(y, cal.transform(over)) < safe_log_loss(y, over)
    cal2, status2, info2 = fit_guarded_calibrator(true, y)
    print("already calibrated ->", status2)
    assert cal2 is None

def test_lambda_fit_reproduces_hda():
    for tgt in ([0.5, 0.27, 0.23], [0.2, 0.28, 0.52], [0.33, 0.3, 0.37]):
        lh, la = fit_lambdas_to_hda(tgt, 2.6)
        from scipy.stats import poisson
        k = np.arange(12); m = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la)); m /= m.sum()
        got = np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
        print(tgt, np.round(got, 3), round(lh, 2), round(la, 2))
        assert np.abs(got - np.array(tgt)).max() < 0.03

def test_market_blend_pulls_toward_market():
    pm = np.array([0.16, 0.2, 0.64]); mk = devig([9.3, 5.5, 1.25])
    b = blend_with_market([0.16, 0.2, 0.64], mk, 0.35)
    assert abs(b[0] - mk[0]) < abs(pm[0] - mk[0])
