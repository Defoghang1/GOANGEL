"""
goangel_core.py — briques partagées (preprocessing, calibration, adaptateur PB, blend).

Tout ce qui doit être IDENTIQUE entre entraînement, finalisation et inférence
vit ici, pour qu'un seul code produise les 251 colonnes / l'échelle / la calibration.
Aucune dépendance à config.py (testable seul).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("goangel.core")

_EPS = 1e-9
_P_FLOOR = 1e-3          # plancher de proba après calibration (évite LogLoss -> inf)
LABELS = [0, 1, 2]


# ═══════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING UNIQUE
# ═══════════════════════════════════════════════════════════════════════════
class GoPreprocessor(BaseEstimator, TransformerMixin):
    """Imputer(median, keep_empty_features=True) -> clip ±z·σ -> StandardScaler -> sélection de colonnes.

    * Fit UNIQUEMENT sur le train chronologique (anti-leakage).
    * L'entrée reste TOUJOURS à n_features_in_ colonnes (251) : prédiction et entraînement partagent
      le même contrat. Si `selected_idx` est fourni, la SORTIE est restreinte à ces colonnes
      (les 100 features retenues par FeatureSelector) — pour transform() ET transform_raw().
    * transform_raw : pour XGB/LGBM/CatBoost (NaN natifs), seulement inf -> NaN (+ sélection).
    """

    def __init__(self, z_limit: float = 5.0, selected_idx: Optional[Sequence[int]] = None):
        self.z_limit = z_limit
        self.selected_idx = selected_idx

    @staticmethod
    def _clean(X: Any) -> np.ndarray:
        arr = np.array(X, dtype=np.float64, copy=True)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr[~np.isfinite(arr)] = np.nan
        return arr

    def _sel(self, A: np.ndarray) -> np.ndarray:
        if self.selected_idx is None:
            return A
        return A[:, np.asarray(self.selected_idx, dtype=int)]

    def fit(self, X: Any, y: Any = None) -> "GoPreprocessor":
        Xc = self._clean(X)
        self.n_features_in_ = int(Xc.shape[1])
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xc)
        Xi = self.imputer_.transform(Xc)
        mean = Xi.mean(axis=0)
        std = Xi.std(axis=0)
        std = np.where((std == 0) | ~np.isfinite(std), 1.0, std)
        z = float(self.z_limit) if self.z_limit and self.z_limit > 0 else np.inf
        self.lo_ = mean - z * std
        self.hi_ = mean + z * std
        self.scaler_ = StandardScaler().fit(np.clip(Xi, self.lo_, self.hi_))
        return self

    def _check(self, Xc: np.ndarray) -> None:
        if Xc.shape[1] != self.n_features_in_:
            raise ValueError(
                f"GoPreprocessor attend {self.n_features_in_} colonnes, reçu {Xc.shape[1]}"
            )

    def transform(self, X: Any) -> np.ndarray:
        Xc = self._clean(X)
        self._check(Xc)
        Xi = self.imputer_.transform(Xc)
        return self._sel(self.scaler_.transform(np.clip(Xi, self.lo_, self.hi_)))

    def transform_raw(self, X: Any) -> np.ndarray:
        Xc = self._clean(X)
        self._check(Xc)
        return self._sel(Xc)

    @property
    def n_features_out_(self) -> int:
        return int(len(self.selected_idx)) if self.selected_idx is not None else int(self.n_features_in_)


# ═══════════════════════════════════════════════════════════════════════════
# 1b. SÉLECTION DE FEATURES (100 features à vrai pouvoir prédictif, sans fuite)
# ═══════════════════════════════════════════════════════════════════════════
class FeatureSelector:
    """Sélectionne jusqu'à `k` features, en n'utilisant QUE le train chronologique.

    Pipeline :
      1. filtre couverture (>= min_coverage de valeurs non-NaN) et variance non nulle ;
      2. split chronologique INTERNE du train (75 % / 25 %) ;
      3. un HistGradientBoosting (NaN natif, déterministe) est ajusté sur les 75 % anciens ;
      4. importance par PERMUTATION = hausse de LogLoss sur les 25 % récents quand la colonne
         est mélangée (n_repeats fois) — mesure hors-échantillon, pas de score « gain » biaisé ;
      5. information mutuelle (MI) sur les 75 % anciens (signal non-linéaire univarié) ;
      6. score = 0.7·rang(permutation) + 0.3·rang(MI) ;
      7. n'est « qualifiée » qu'une feature avec permutation > 0 ET MI > 0 : AUCUN bruit à score nul
         n'est ajouté pour atteindre le quota ;
      8. dé-duplication des colonnes quasi identiques (|Spearman| > corr_threshold) ; en cas de
         doublon, la feature différentielle `diff_*` est préférée à ses composantes.
    Si moins de `min_k` features sont qualifiées, on complète avec les meilleures restantes (par score)
    jusqu'à min_k, et on le signale dans le rapport.
    """

    def __init__(self, k: int = 100, min_k: int = 60, min_coverage: float = 0.30,
                 corr_threshold: float = 0.97, n_repeats: int = 3, inner_ratio: float = 0.75,
                 seed: int = 42):
        self.k = k
        self.min_k = min_k
        self.min_coverage = min_coverage
        self.corr_threshold = corr_threshold
        self.n_repeats = n_repeats
        self.inner_ratio = inner_ratio
        self.seed = seed

    @staticmethod
    def _rank01(v: np.ndarray) -> np.ndarray:
        from scipy.stats import rankdata
        if len(v) <= 1:
            return np.zeros(len(v))
        return (rankdata(v) - 1.0) / (len(v) - 1.0)

    def fit(self, X: Any, y: Any, names: Optional[Sequence[str]] = None) -> "FeatureSelector":
        from scipy.stats import rankdata
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.feature_selection import mutual_info_classif

        X = np.asarray(X, dtype=np.float64).copy()
        X[~np.isfinite(X)] = np.nan
        y = np.asarray(y).astype(int)
        n, p = X.shape
        names = list(names) if names is not None else [f"f{i}" for i in range(p)]
        rng = np.random.default_rng(self.seed)

        cover = 1.0 - np.isnan(X).mean(axis=0)
        with np.errstate(all="ignore"):
            std = np.nanstd(X, axis=0)
        usable = np.where((cover >= self.min_coverage) & np.isfinite(std) & (std > 1e-12))[0]

        cut = int(n * self.inner_ratio)
        Xa, ya, Xb, yb = X[:cut], y[:cut], X[cut:], y[cut:]
        perm = np.zeros(p)
        perm_ok = False
        try:
            hgb = HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=60,
                l2_regularization=5.0, early_stopping=False, random_state=self.seed)
            hgb.fit(Xa[:, usable], ya)
            def _ll(M):
                return float(log_loss(yb, np.clip(hgb.predict_proba(M), 1e-6, 1), labels=LABELS))
            Xb_u = Xb[:, usable].copy()
            base = _ll(Xb_u)
            for jj in range(len(usable)):
                col = Xb_u[:, jj].copy()
                d = 0.0
                for _ in range(self.n_repeats):
                    Xb_u[:, jj] = rng.permutation(col)
                    d += _ll(Xb_u) - base
                Xb_u[:, jj] = col
                perm[usable[jj]] = d / self.n_repeats
            perm_ok = True
            self.base_val_ll_ = base
        except Exception as exc:  # noqa: BLE001
            logger.warning("permutation importance indisponible (%s) -> MI seule", exc)

        med = np.nanmedian(np.where(np.isnan(Xa), np.nan, Xa), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xa_imp = np.where(np.isnan(Xa), med, Xa)
        mi = np.zeros(p)
        try:
            mi[usable] = mutual_info_classif(Xa_imp[:, usable], ya, n_neighbors=5, random_state=self.seed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MI indisponible : %s", exc)

        score = np.full(p, -1.0)
        if perm_ok:
            score[usable] = 0.7 * self._rank01(perm[usable]) + 0.3 * self._rank01(mi[usable])
            qualified = [j for j in usable if perm[j] > 0 and mi[j] > 0]
        else:
            score[usable] = self._rank01(mi[usable])
            qualified = [j for j in usable if mi[j] > 0]
        order_q = sorted(qualified, key=lambda j: -score[j])
        order_rest = sorted([j for j in usable if j not in set(qualified)], key=lambda j: -score[j])

        # Spearman en « paires complètes » : rangs calculés sur les valeurs présentes, NaN -> 0 + masque.
        R = np.zeros_like(Xa)
        M = np.isfinite(Xa).astype(np.float64)
        for j in usable:
            ok = M[:, j] > 0
            if ok.sum() > 10:
                r = rankdata(Xa[ok, j])
                R[ok, j] = (r - r.mean()) / (r.std() + 1e-12)
        num = R.T @ R
        den = np.sqrt(np.maximum((R ** 2).T @ M, 1e-12) * np.maximum(M.T @ (R ** 2), 1e-12))
        C = np.abs(num / den)
        np.fill_diagonal(C, 0.0)

        def _corr(i: int, j: int) -> float:
            return float(C[i, j])

        chosen: List[int] = []
        dropped_dup = 0
        def _try_add(j: int) -> None:
            nonlocal dropped_dup
            for pos, c in enumerate(chosen):
                if _corr(j, c) > self.corr_threshold:
                    if names[j].startswith("diff_") and not names[c].startswith("diff_"):
                        chosen[pos] = j  # préférence aux différentielles
                    dropped_dup += 1
                    return
            chosen.append(j)

        for j in order_q:
            if len(chosen) >= self.k:
                break
            _try_add(j)
        n_qualified_kept = len(chosen)
        for j in order_rest:
            if len(chosen) >= self.min_k:
                break
            _try_add(j)

        self.selected_idx_ = sorted(int(j) for j in chosen)
        self.selected_names_ = [names[j] for j in self.selected_idx_]
        self.report_ = {
            "n_input": int(p), "n_usable": int(len(usable)), "n_qualified": int(len(qualified)),
            "n_selected": int(len(self.selected_idx_)), "n_qualified_kept": int(n_qualified_kept),
            "n_topped_up_below_min_k": int(len(chosen) - n_qualified_kept),
            "n_duplicates_removed": int(dropped_dup), "permutation_ok": bool(perm_ok),
            "n_diff_selected": int(sum(1 for nm in self.selected_names_ if nm.startswith("diff_"))),
            "top20": [(names[j], round(float(perm[j]), 5), round(float(mi[j]), 5))
                      for j in sorted(chosen, key=lambda j: -score[j])[:20]],
        }
        import hashlib
        self.selection_id_ = hashlib.sha1("|".join(self.selected_names_).encode()).hexdigest()[:12]
        return self


# ═══════════════════════════════════════════════════════════════════════════
# 2. MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════
def normalize_probas(P: Any, floor: float = _P_FLOOR) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    P = np.where(np.isfinite(P), P, 1.0 / 3.0)
    P = np.clip(P, floor, 1.0)
    return P / P.sum(axis=1, keepdims=True)


def safe_log_loss(y: Any, P: Any) -> float:
    try:
        return float(log_loss(np.asarray(y), normalize_probas(P, 1e-6), labels=LABELS))
    except Exception:
        return float("inf")


def prior_log_loss(y_train: Any, y_eval: Any) -> float:
    """LogLoss du prédicteur constant = fréquences de classes du TRAIN. Référence 'zéro skill'."""
    y_train = np.asarray(y_train)
    freq = np.array([(y_train == k).mean() for k in LABELS], dtype=np.float64)
    freq = freq / freq.sum()
    P = np.tile(freq, (len(y_eval), 1))
    return safe_log_loss(y_eval, P)


# ═══════════════════════════════════════════════════════════════════════════
# 3. CALIBRATION SUR PROBABILITÉS (ML et PB, même code)
# ═══════════════════════════════════════════════════════════════════════════
class ProbaCalibrator:
    """Calibre un tableau (n,3) de probabilités.

    method="isotonic" : isotonic one-vs-rest + renormalisation.
    method="sigmoid"  : Platt multinomial (régression logistique sur log-probas).
    Température volontairement ABSENTE (évite la double calibration).
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method

    def fit(self, P: Any, y: Any) -> "ProbaCalibrator":
        P = normalize_probas(P, 1e-6)
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 3:
            raise ValueError("calibration : les 3 classes doivent être présentes")
        if self.method == "isotonic":
            self.iso_ = [
                IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
                    P[:, k], (y == k).astype(np.float64)
                )
                for k in LABELS
            ]
        elif self.method == "sigmoid":
            self.lr_ = LogisticRegression(C=1.0, max_iter=2000).fit(np.log(P), y)
        else:
            raise ValueError(f"méthode inconnue : {self.method}")
        return self

    def transform(self, P: Any) -> np.ndarray:
        P = normalize_probas(P, 1e-6)
        if self.method == "isotonic":
            out = np.column_stack([self.iso_[k].predict(P[:, k]) for k in LABELS])
        else:
            out = self.lr_.predict_proba(np.log(P))
        return normalize_probas(out, _P_FLOOR)


def _per_sample_ll(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    P = normalize_probas(P, 1e-6)
    return -np.log(P[np.arange(len(y)), y])


def fit_guarded_calibrator(
    P_calib: Any,
    y_calib: Any,
    min_isotonic_samples: int = 1000,
    min_gain: float = 5e-4,
    n_blocks: int = 4,
    min_tstat: float = 1.0,
) -> Tuple[Optional[ProbaCalibrator], str, Dict[str, float]]:
    """Calibre SEULEMENT si le gain est réel, mesuré hors-échantillon ET statistiquement lisible.

    Ancienne version : un seul split 60/40 (~900 points) pour détecter un gain de 5e-4 alors que
    l'erreur-type du LogLoss sur 900 points vaut ~0.015 -> décision quasi aléatoire (« identity »
    presque toujours). Nouvelle version :
      * validation « forward-chaining » : le calibrateur est ajusté sur les blocs 0..k-1 et évalué sur
        le bloc k (jamais vu), pour k = 1..n_blocks-1 -> ~75 % du calibration set sert d'évaluation ;
      * test APPARIÉ : d_i = LL_brut_i - LL_calibré_i ; on exige mean(d) >= min_gain ET
        mean(d)/erreur-type(d) >= min_tstat ;
      * parmi isotonic / sigmoid, on garde celle au meilleur gain moyen qui passe le test ;
      * sinon probas brutes conservées (None) ; si OK : refit sur 100 % du calibration set.
    """
    P = normalize_probas(P_calib, 1e-6)
    y = np.asarray(y_calib).astype(int)
    n = len(y)
    info: Dict[str, float] = {"n": float(n)}
    if n < 200:
        return None, "skipped_small_calib_set", info
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    if edges[1] < 80 or len(np.unique(y[: edges[1]])) < 3:
        return None, "skipped_missing_classes", info

    ll_raw_all = _per_sample_ll(y, P)
    info["oos_ll_raw"] = float(ll_raw_all[edges[1]:].mean())
    methods = ["isotonic", "sigmoid"] if n >= min_isotonic_samples else ["sigmoid", "isotonic"]
    best_method, best_gain = None, -np.inf
    for m in methods:
        diffs: List[np.ndarray] = []
        try:
            for k in range(1, n_blocks):
                tr, te = slice(0, edges[k]), slice(edges[k], edges[k + 1])
                if len(np.unique(y[tr])) < 3:
                    raise ValueError("classe manquante")
                cal = ProbaCalibrator(m).fit(P[tr], y[tr])
                diffs.append(ll_raw_all[te] - _per_sample_ll(y[te], cal.transform(P[te])))
            d = np.concatenate(diffs)
        except Exception as exc:  # noqa: BLE001
            logger.debug("calibration %s échouée : %s", m, exc)
            continue
        gain = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.inf
        t = gain / se if se > 0 and np.isfinite(se) else 0.0
        info[f"oos_gain_{m}"], info[f"oos_t_{m}"] = gain, float(t)
        if gain >= min_gain and t >= min_tstat and gain > best_gain:
            best_method, best_gain = m, gain
    if best_method is None:
        return None, "raw_kept_no_gain", info
    final = ProbaCalibrator(best_method).fit(P, y)
    return final, f"calibrated_{best_method}", info


class PrefitCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """base.predict_proba -> ProbaCalibrator. Picklable, sans dépendance FrozenEstimator."""

    def __init__(self, base: Any = None, calibrator: Optional[ProbaCalibrator] = None):
        self.base = base
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1, 2])
        self._estimator_type = "classifier"

    def fit(self, X: Any, y: Any = None) -> "PrefitCalibratedClassifier":
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def predict_proba(self, X: Any) -> np.ndarray:
        P = np.asarray(self.base.predict_proba(X), dtype=np.float64)
        if self.calibrator is None:
            return normalize_probas(P, 1e-6)
        return self.calibrator.transform(P)

    def predict(self, X: Any) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# 4. ADAPTATEUR UNIQUE POUR LES MODÈLES "STATISTIQUES" (PB et assimilés)
# ═══════════════════════════════════════════════════════════════════════════
def _hda_from_grid(grid: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(grid.home_draw_away, dtype=np.float64).reshape(-1)
        # BUG CORRIGÉ : `arr.shape == (3)` comparait un tuple à un entier (toujours False)
        if arr.shape == (3,) and np.all(np.isfinite(arr)) and arr.sum() > 0:
            return arr / arr.sum()
    except Exception:
        pass
    return None


def pb_hda(model: Any, home: str, away: str) -> Optional[np.ndarray]:
    """Proba 1X2 d'un modèle stat, quelle que soit son interface :
    .predict(h,a)->grid | .predict_proba_1x2(h,a) | .predict_proba(DataFrame)."""
    if hasattr(model, "predict"):
        try:
            p = _hda_from_grid(model.predict(home, away))
            if p is not None:
                return p
        except Exception:
            pass
    if hasattr(model, "predict_proba_1x2"):
        try:
            p = np.asarray(model.predict_proba_1x2(home, away), dtype=np.float64).reshape(-1)
            if p.shape == (3,) and np.all(np.isfinite(p)) and p.sum() > 0:
                return p / p.sum()
        except Exception:
            pass
    if hasattr(model, "predict_proba"):
        try:
            import pandas as pd

            df = pd.DataFrame({"home_key": [home], "away_key": [away]})
            p = np.asarray(model.predict_proba(df), dtype=np.float64).reshape(-1)
            if p.shape == (3,) and np.all(np.isfinite(p)) and p.sum() > 0:
                return p / p.sum()
        except Exception:
            pass
    return None


def pb_hda_batch(model: Any, homes: Sequence[str], aways: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Retourne (P[n,3] avec NaN si échec, masque_valide[n])."""
    P = np.full((len(homes), 3), np.nan)
    ok = np.zeros(len(homes), dtype=bool)
    for i, (h, a) in enumerate(zip(homes, aways)):
        p = pb_hda(model, h, a)
        if p is not None:
            P[i] = p
            ok[i] = True
    return P, ok


class _GridProxy:
    """Grille PB dont home_draw_away est remplacé par la version calibrée."""

    def __init__(self, grid: Any, hda: np.ndarray):
        self._grid = grid
        self.home_draw_away = [float(x) for x in hda]

    def __getattr__(self, name: str) -> Any:
        grid = self.__dict__.get("_grid")
        if grid is None:
            raise AttributeError(name)
        return getattr(grid, name)


class CalibratedPBModel:
    """Enveloppe un modèle stat + calibrateur 1X2. Expose .predict(h,a) comme une grille PB
    (over/under, BTTS, xG restent ceux du modèle d'origine)."""

    def __init__(self, base: Any, calibrator: Optional[ProbaCalibrator]):
        self.base = base
        self.calibrator = calibrator

    def predict(self, home: str, away: str) -> Any:
        grid = None
        if hasattr(self.base, "predict"):
            try:
                grid = self.base.predict(home, away)
            except Exception:
                grid = None
        p = _hda_from_grid(grid) if grid is not None else None
        if p is None:
            p = pb_hda(self.base, home, away)
        if p is None:
            raise ValueError("aucune proba 1X2 disponible")
        if self.calibrator is not None:
            p = self.calibrator.transform(p.reshape(1, -1))[0]
        return _GridProxy(grid, p)

    def predict_proba_1x2(self, home: str, away: str) -> np.ndarray:
        return np.asarray(self.predict(home, away).home_draw_away, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# 5. POIDS D'ENSEMBLE : porte "zéro skill" + optimisation convexe sur VAL
# ═══════════════════════════════════════════════════════════════════════════
def skill_gate(
    val_ll: Dict[str, float],
    prior_ll: float,
    min_skill: float = 0.005,
    max_gap_to_best: float = 0.04,
) -> List[str]:
    """Garde un modèle seulement s'il bat le prédicteur constant d'au moins `min_skill`
    ET reste à moins de `max_gap_to_best` du meilleur. (Remplace l'ancien filtre x1.25,
    qui laissait passer HMM/Skellam/MonteCarlo/Copula = zéro information équipe.)"""
    finite = {k: v for k, v in val_ll.items() if np.isfinite(v)}
    if not finite:
        return []
    best = min(finite.values())
    return [
        k for k, v in finite.items()
        if v <= prior_ll - min_skill and v <= best + max_gap_to_best
    ]


def optimize_blend_weights(
    probas: Dict[str, np.ndarray],
    y: np.ndarray,
    w_max: float = 0.60,
    ridge: float = 2e-3,
) -> Dict[str, float]:
    """Poids >= 0, somme 1, plafond w_max, minimisant LogLoss(val) + ridge·||w-uniforme||²."""
    keys = list(probas.keys())
    m = len(keys)
    if m == 0:
        return {}
    if m == 1:
        return {keys[0]: 1.0}
    y = np.asarray(y).astype(int)
    stack = np.stack([normalize_probas(probas[k], 1e-6) for k in keys], axis=0)  # (m,n,3)
    onehot = np.eye(3)[y]
    cap = max(float(w_max), 1.0 / m + 1e-9)
    u = np.ones(m) / m

    def obj(w: np.ndarray) -> float:
        P = np.tensordot(w, stack, axes=(0, 0))
        P = np.clip(P, 1e-9, 1.0)
        return float(-np.mean(np.sum(onehot * np.log(P), axis=1)) + ridge * np.sum((w - u) ** 2))

    res = minimize(
        obj, u, method="SLSQP", bounds=[(0.0, cap)] * m,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = res.x if (res.success and np.all(np.isfinite(res.x))) else u
    w = np.clip(w, 0.0, cap)
    w = w / w.sum()
    w = np.where(w < 1e-4, 0.0, w)
    w = w / w.sum()
    return {k: float(v) for k, v in zip(keys, w)}


def blend(probas: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    keys = [k for k in weights if weights[k] > 0 and k in probas]
    tot = sum(weights[k] for k in keys)
    out = sum(weights[k] / tot * normalize_probas(probas[k], 1e-6) for k in keys)
    return normalize_probas(out, 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 6. COHÉRENCE 1X2 <-> MATRICE DE SCORES, ET INFORMATION DU MARCHÉ
# ═══════════════════════════════════════════════════════════════════════════
def fit_lambdas_to_hda(target_hda: Any, total_prior: float, rho: float = -0.05,
                       grid: int = 12, prior_strength: float = 0.02) -> Tuple[float, float]:
    """Trouve (λ_dom, λ_ext) dont la matrice Poisson(+Dixon-Coles) reproduit le 1X2 de l'ENSEMBLE
    validé. Ainsi scores exacts / over-under / BTTS sont cohérents avec les probabilités calibrées
    au lieu d'être mélangés à 50 % avec une heuristique non calibrée."""
    from scipy.stats import poisson
    t = normalize_probas(np.asarray(target_hda, dtype=np.float64), 1e-4)[0]
    ks = np.arange(grid)

    def hda(lh: float, la: float) -> np.ndarray:
        m = np.outer(poisson.pmf(ks, lh), poisson.pmf(ks, la))
        if rho != 0.0:
            m[0, 0] *= max(0.01, 1 - lh * la * rho)
            m[0, 1] *= max(0.01, 1 + lh * rho)
            m[1, 0] *= max(0.01, 1 + la * rho)
            m[1, 1] *= max(0.01, 1 - rho)
        m = m / m.sum()
        return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])

    def obj(v: np.ndarray) -> float:
        lh, la = float(np.exp(v[0])), float(np.exp(v[1]))
        e = hda(lh, la) - t
        return float(np.sum(e * e) + prior_strength * (np.log(lh + la) - np.log(max(total_prior, 0.5))) ** 2)

    x0 = np.log([max(total_prior, 1.0) * 0.55, max(total_prior, 1.0) * 0.45])
    res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 400})
    lh, la = np.exp(res.x)
    return float(np.clip(lh, 0.05, 6.0)), float(np.clip(la, 0.05, 6.0))


def devig(odds: Sequence[float]) -> np.ndarray:
    """Probabilités implicites sans marge (normalisation proportionnelle)."""
    inv = 1.0 / np.asarray(odds, dtype=np.float64)
    return inv / inv.sum()


def blend_with_market(p_model: Any, p_market: Any, w_model: float = 0.5) -> np.ndarray:
    """Moyenne géométrique pondérée (espace logit) modèle/marché, renormalisée.
    Le marché (cotes de clôture) est le meilleur prédicteur public : on ne laisse plus un modèle
    à LogLoss ~1.0 « contredire » le marché de 15-20 pts sans garde-fou."""
    pm = normalize_probas(p_model, 1e-4)[0]
    pk = normalize_probas(p_market, 1e-4)[0]
    z = w_model * np.log(pm) + (1.0 - w_model) * np.log(pk)
    z = np.exp(z - z.max())
    return z / z.sum()
