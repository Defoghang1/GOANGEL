from config import (
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE,
    XGB_MIN_CHILD_WEIGHT,
    XGB_REG_LAMBDA,
    XGB_REG_ALPHA,
    XGB_EVAL_METRIC,
    LGBM_N_ESTIMATORS,
    LGBM_MAX_DEPTH,
    LGBM_LEARNING_RATE,
    LGBM_NUM_LEAVES,
    LGBM_MIN_CHILD_SAMPLES,
    LGBM_SUBSAMPLE,
    LGBM_SUBSAMPLE_FREQ,
    LGBM_COLSAMPLE,
    LGBM_REG_LAMBDA,
    LGBM_REG_ALPHA,
    LGBM_EVAL_METRIC,
    CATBOOST_ITERATIONS,
    CATBOOST_DEPTH,
    CATBOOST_LEARNING_RATE,
    CATBOOST_L2_LEAF_REG,
    CATBOOST_RANDOM_STRENGTH,
    CATBOOST_EVAL_METRIC,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_MIN_SAMPLES_SPLIT,
    RF_MAX_FEATURES_RESOLVED,
    ET_N_ESTIMATORS,
    ET_MAX_DEPTH,
    ET_MAX_FEATURES,
    ET_MIN_SAMPLES_LEAF,
    ET_MIN_SAMPLES_SPLIT,
    SVM_C,
    SVM_GAMMA,
    SVM_KERNEL,
    SVM_PROBABILITY,
    SVM_TOL,
    LR_C,
    LR_PENALTY,
    LR_SOLVER,
    LR_MAX_ITER,
)
import os
import re
import sys
import time
import json
import signal
import hashlib
import logging
import argparse
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    COMPETITIONS_DETAILS,
    COMPETITIONS_BY_CODE,
    COMPETITIONS,
    CACHE_DIR,
    CHECKPOINT_DIR,
    CACHE_NEVER_EXPIRES,
    MIN_MATCHES_FOR_TRAINING,
    MANIFEST_FILE,
    CONFIG_FINGERPRINT,
    ARTIFACT_SCHEMA_VERSION,
    EXPECTED_FEATURE_COUNT_REFERENCE,
    EXPECTED_CLASS_COUNT,
    HISTORICAL_SEASONS_LOOKBACK,
    BZZOIRO_ENRICH_XG,
    BZZOIRO_WORKERS_PER_KEY,
    BZZOIRO_MAX_WORKERS,
    TOKEN_COUNT,
    TRAIN_RATIO,
    VAL_RATIO,
    CALIB_RATIO,
    TEST_RATIO,
    MODEL_WALK_FORWARD_FOLDS,
    TEMPERATURE_SCALING_ENABLED,
    TEMPERATURE_SCALING_DEFAULT,
    TEMPERATURE_SCALING_MIN,
    TEMPERATURE_SCALING_MAX,
    ENSEMBLE_MIN_WEIGHT,
    ENSEMBLE_MAX_WEIGHT,
    MODEL_RANDOM_SEED,
    DL_ENABLED,
    DL_LEARNING_RATE,
    DL_WEIGHT_DECAY,
    DL_BATCH_SIZE,
    DL_EPOCHS,
    DL_EARLY_STOPPING_PATIENCE,
    DL_GRAD_CLIP,
    FEATURE_OUTLIER_ZSCORE,
    now_local,
    stage_print)

from features import (
    FEATURE_NAMES,
    EXPECTED_FEATURE_COUNT,
    _FEATURE_INDEX,
    _NON_NEGATIVE_FEATURES,
    _TREE_MODELS_NAN_NATIVE,
    _PB_MODEL_CLASSES,
    _PB_MODEL_ORDER,
    _PB_BAYESIAN_MODELS,
    _MIN_CLASSES_FOR_TRAINING,
    _MIN_TEST_SAMPLES,
    _MIN_VAL_SAMPLES_FOR_DL,
    _CALIBRATION_STATUS_OK,
    _CALIBRATION_STATUS_NOT_CALIBRATED,
    _CALIBRATION_STATUS_SKIPPED_SMALL,
    _CALIBRATION_STATUS_SKIPPED_CLASSES,
    _CALIBRATION_STATUS_FAILED,
    _CALIBRATION_STATUS_RAW_KEPT,
    _CALIBRATION_STATUS_TEMPERATURE,
    _LL_SOURCE_WALK_FORWARD,
    _LL_SOURCE_CALIBRATION_FALLBACK,
    _MAX_ACCEPTABLE_LOGLOSS,
    _MIN_WF_FOLD_COVERAGE,
    _PROBA_ATOL,
    _EPS,
    _NAN,
    _TorchNet,
    _TorchProbaWrapper,
    _TorchTemperatureWrapper,
    _make_xgb_factory,
    _make_lgbm_factory,
    _make_catboost_factory,
    _make_rf_factory,
    _evaluate_ml_model_walk_forward,
    _evaluate_dl_walk_forward,
    _evaluate_pb_model_walk_forward,
    _fit_pb_model,
    _calibrate_or_keep,
    _compute_calibration_log_loss,
    _model_produces_three_classes,
    _proba_array_is_valid,
    _extract_model_probabilities,
    _walk_forward_splits,
    _compute_optimal_temperature,
    _compute_ensemble_probas,
    _evaluate_ensemble_on_set,
    _filter_valid_weights,
    _normalize_weights,
    _select_fallback_uniform_models,
    _apply_quality_filter_on_weights,
    process_matches,
    create_features_with_context)

try:
    from features import (
        _make_lr_factory, _make_et_factory, _make_svm_factory,
        _make_fttransformer_factory, DixonColesDynamicModel, KalmanBayesianModel,
        HierarchicalBayesianADVIModel, MonteCarloSimulator, StackingEnsemble)
except ImportError:
    pass

try:
    from features import (
        _make_skellam_factory, _make_copula_factory, _make_ngboost_factory,
        _make_hmm_factory, _make_tabm_factory, _make_realmlp_factory,
        _make_excelformer_factory, _make_limix_factory, _make_lightpfn_factory,
        SkellamModel, CopulaModel, NGBoostModel, HMMModel, TabMWrapper)
except ImportError:
    pass

from predict import (
    _load_full_history,
    _atomic_write_pickle,
    _atomic_read_pickle)

try:
    from penaltyblog_method_patch import apply_patch as _pb_patch_apply
    _pb_patch_apply()
except Exception as _pb_patch_exc:
    import logging as _plogging
    _plogging.getLogger("goangel.main").warning(
        "penaltyblog_method_patch échoué : %s", _pb_patch_exc
    )

from api import (
    CheckpointManager,
    get_historical_matches)

from pb_convergence import PBIncompleteError
from checkpoint_all import (
    save_step as ck_save_step, load_step as ck_load_step,
    has_step as ck_has_step, save_fold as ck_save_fold,
    list_folds as ck_list_folds, load_finalize_context,
    save_finalize_context, save_ml_trained, save_ml_calibrated,
    save_pb_finalized, save_finalize_partial, load_finalize_partial,
    has_finalize_partial, list_finalize_partials, save_finalize_agg,
    ml_key, pb_key, finalize_key, STEP_TRAINED, STEP_CALIBRATED,
    STEP_FINALIZED, STEP_CONTEXT, STEP_PARTIAL)

# Forcer l'application du patch bayesian au chargement du module
try:
    from pb_convergence import apply_bayesian_init_patch as _ensure_bayesian_patch
    _ensure_bayesian_patch()
except Exception as _exc:
    import logging as _l
    _l.getLogger("goangel.main").warning("Patch bayesian init échoué : %s", _exc)

logger = logging.getLogger("goangel.main")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    RandomForestClassifier = None

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV

try:
    from sklearn.frozen import FrozenEstimator
    _FROZEN_AVAILABLE = True
except ImportError:
    FrozenEstimator = None
    _FROZEN_AVAILABLE = False

RESET: str = "\033[0m"
BOLD: str = "\033[1m"
DIM: str = "\033[2m"
ITALIC: str = "\033[3m"

RED: str = "\033[1;31m"
GREEN: str = "\033[1;32m"
YELLOW: str = "\033[1;33m"
BLUE: str = "\033[1;34m"
MAGENTA: str = "\033[1;35m"
CYAN: str = "\033[1;36m"
WHITE: str = "\033[1;37m"
GREY: str = "\033[0;90m"
ORANGE: str = "\033[38;5;208m"
LIME: str = "\033[38;5;154m"
PINK: str = "\033[38;5;213m"
VIOLET: str = "\033[38;5;135m"
SKY: str = "\033[38;5;117m"
GOLD: str = "\033[38;5;220m"

_BANNER_LOGO: List[str] = [
    r"   ██████╗  ██████╗  █████╗ ███╗   ██╗ ██████╗ ███████╗██╗     ",
    r"  ██╔════╝ ██╔═══██╗██╔══██╗████╗  ██║██╔════╝ ██╔════╝██║     ",
    r"  ██║  ███╗██║   ██║███████║██╔██╗ ██║██║  ███╗█████╗  ██║     ",
    r"  ██║   ██║██║   ██║██╔══██║██║╚██╗██║██║   ██║██╔══╝  ██║     ",
    r"  ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║╚██████╔╝███████╗███████╗",
    r"   ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚══════╝",
]

_BANNER_COLORS: List[str] = [RED, ORANGE, GOLD, LIME, SKY, VIOLET]

_TITLE: str = "GOANGEL SYSTEM"
_SUBTITLE: str = "★MOTEUR DE PRÉDICTIONS FOOTBALLISTIQUES MULTI-MODÈLES ★"
_VERSION: str = f"v23 — ARCTIQUE — {EXPECTED_FEATURE_COUNT_REFERENCE} FEATURES"
_TAGLINE: str = "★ ANALYSE · STATISTIQUES · ENSEMBLE LEARNING · CALIBRATION★"

_SHUTDOWN_FLAG: Dict[str, bool] = {"requested": False}

_MIN_YEAR: int = 1990
_MAX_YEAR: int = 2100

_TUPLE_LEN_EXPECTED: int = 5

_BANNER_WIDTH_MIN: int = 68
_BANNER_WIDTH_MAX: int = 100
_SEPARATOR_LEN: int = 70

_DATE_FORMAT_ISO: str = "%Y-%m-%d"

_QUIT_COMMANDS: frozenset = frozenset({"quit", "exit", "q"})
_HELP_COMMANDS: frozenset = frozenset({"help", "h", "?"})
_COMPETITIONS_COMMANDS: frozenset = frozenset({"competitions", "comps", "c"})
_CLEAR_COMMANDS: frozenset = frozenset({"clear", "cls"})
_REFRESH_COMMANDS: frozenset = frozenset({"refresh", "reload", "r"})
_STATUS_COMMANDS: frozenset = frozenset({"status", "info"})
_TODAY_COMMANDS: frozenset = frozenset({"today", "aujourd'hui", "aujourdhui"})
_YESTERDAY_COMMANDS: frozenset = frozenset({"yesterday", "hier"})
_TOMORROW_COMMANDS: frozenset = frozenset({"tomorrow", "demain"})

_FULL_DATE_RE = re.compile(r"^\s*(\d{1,2})\s*[/\-\.]\s*(\d{1,2})\s*[/\-\.]\s*(\d{4})\s*$")
_DAY_MONTH_RE = re.compile(r"^\s*(\d{1,2})\s*[/\-\.]\s*(\d{1,2})\s*$")
_DAY_ONLY_RE = re.compile(r"^\s*(\d{1,2})\s*$")

STAGES_DIR: Path = Path(CACHE_DIR) / "stages"
STAGES_DIR.mkdir(parents=True, exist_ok=True)

DONE_DIR: Path = STAGES_DIR / "done"
DONE_DIR.mkdir(parents=True, exist_ok=True)

ATTEMPTS_DIR: Path = STAGES_DIR / "attempts"
ATTEMPTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR: Path = STAGES_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_FEATURES_FILE: Path = STAGES_DIR / "features.pkl"
_METADATA_FILE: Path = STAGES_DIR / "metadata.pkl"

def _safe_int_env(name: str, default: int) -> int:
    val = os.getenv(name, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

_MAX_ATTEMPTS_DEFAULT: int = _safe_int_env("GOANGEL_MAX_ATTEMPTS_PER_MODEL", 15)

_PB_STAGE_KEYS: Dict[str, str] = {
    "train_pb_poisson": "PoissonGoalsModel", "train_pb_dixon_coles": "DixonColesGoalModel",
    "train_pb_bivariate": "BivariatePoissonGoalModel", "train_pb_negbinomial": "NegativeBinomialGoalModel",
    "train_pb_zeroinflated": "ZeroInflatedPoissonGoalsModel", "train_pb_weibull": "WeibullCopulaGoalsModel",
    "train_dixon_coles_dynamic": "DixonColesDynamicModel", "train_kalman": "KalmanBayesianModel",
    "train_hierarchical_advi": "HierarchicalBayesianADVIModel", "train_monte_carlo": "MonteCarloSimulator",
    "train_skellam": "SkellamModel", "train_copula": "CopulaModel",
    "train_hmm": "HMMModel"}

_ML_STAGE_KEYS: Dict[str, str] = {
    "train_xgb": "XGB", "train_lgbm": "LGBM", "train_catboost": "CatBoost", "train_rf": "RF", "train_dl": "DL",
    "train_lr": "LR", "train_et": "ET", "train_svm": "SVM",
    "train_histgb": "HistGB", "train_mlp": "MLP",
    "train_tabm": "TabM",
    "train_realmlp": "RealMLP", "train_excelformer": "ExcelFormer",
    
    "train_ngboost": "NGBoost"}

_STAGES_ORDER: Tuple[str, ...] = (
    "prepare", "refresh",
    "train_xgb", "train_lgbm", "train_catboost", "train_rf", "train_dl",
    "train_lr", "train_et", "train_svm",
    "train_histgb", "train_mlp",
    "train_tabm", "train_realmlp", "train_excelformer",
    "train_ngboost", "train_hmm",
    "train_pb_poisson", "train_pb_dixon_coles", "train_pb_bivariate",
    "train_pb_negbinomial", "train_pb_zeroinflated", "train_pb_weibull",
    "train_dixon_coles_dynamic", "train_kalman",
    "train_hierarchical_advi", "train_monte_carlo", "train_skellam", "train_copula",
    "train_stacking",
    "finalize_prep",
    "finalize_xgb", "finalize_lgbm", "finalize_catboost", "finalize_rf", "finalize_dl",
    "finalize_lr", "finalize_et", "finalize_svm",
    "finalize_histgb", "finalize_mlp",
    "finalize_tabm", "finalize_realmlp", "finalize_excelformer",
    "finalize_ngboost", "finalize_hmm",
    "finalize_pb_poisson", "finalize_pb_dixon_coles", "finalize_pb_bivariate",
    "finalize_pb_negbinomial", "finalize_pb_zeroinflated", "finalize_pb_weibull",
    "finalize_dixon_coles_dynamic", "finalize_kalman",
    "finalize_hierarchical_advi", "finalize_monte_carlo", "finalize_skellam", "finalize_copula",
    "finalize_stacking",
    "finalize_agg")

_VALID_STAGES: Tuple[str, ...] = tuple(list(_STAGES_ORDER) + ["next"])

def _visible_len(text: str) -> int:
    result = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\033":
            j = text.find("m", i)
            if j == -1:
                break
            i = j + 1
            continue
        result += 1
        i += 1
    return result

def _center_colored(text: str, width: int) -> str:
    visible = _visible_len(text)
    if visible >= width:
        return text
    left = (width - visible) // 2
    right = width - visible - left
    return (" " * left) + text + (" " * right)

def _safe_terminal_width() -> int:
    try:
        w = os.get_terminal_size().columns
        if w < _BANNER_WIDTH_MIN:
            return _BANNER_WIDTH_MIN
        if w > _BANNER_WIDTH_MAX:
            return _BANNER_WIDTH_MAX
        return int(w)
    except (OSError, ValueError, AttributeError):
        return _BANNER_WIDTH_MIN + 8

def print_banner() -> None:
    try:
        banner_width = _safe_terminal_width()
        term_width = _safe_terminal_width()
        left_pad = max(0, (term_width - banner_width) // 2)
        print()
        print(" " * left_pad + f"{GREY}╔" + "═" * (banner_width - 2) + f"╗{RESET}")
        for i, line in enumerate(_BANNER_LOGO):
            color = _BANNER_COLORS[i % len(_BANNER_COLORS)]
            inner = _center_colored(f"{color}{BOLD}{line}{RESET}", banner_width - 4)
            print(" " * left_pad + f"{GREY}║{RESET} " + inner + f" {GREY}║{RESET}")
        print(" " * left_pad + f"{GREY}╠" + "═" * (banner_width - 2) + f"╣{RESET}")
        title_line = (
            f"{GOLD}{BOLD}★★ {_TITLE} ★★{RESET}  "
            f"{DIM}{GREY}({_VERSION}){RESET}"
        )
        print(" " * left_pad + f"{GREY}║{RESET} " + _center_colored(title_line, banner_width - 4) + f" {GREY}║{RESET}")
        subtitle_line = f"{CYAN}{_SUBTITLE}{RESET}"
        print(" " * left_pad + f"{GREY}║{RESET} " + _center_colored(subtitle_line, banner_width - 4) + f" {GREY}║{RESET}")
        tagline_line = f"{DIM}{ITALIC}{LIME}{_TAGLINE}{RESET}"
        print(" " * left_pad + f"{GREY}║{RESET} " + _center_colored(tagline_line, banner_width - 4) + f" {GREY}║{RESET}")
        print(" " * left_pad + f"{GREY}╠" + "═" * (banner_width - 2) + f"╣{RESET}")
        cache_label = "PERMANENT" if CACHE_NEVER_EXPIRES else "avec expiration"
        cache_color = GREEN if CACHE_NEVER_EXPIRES else YELLOW
        cache_line = (
            f"{SKY}💾 Cache {cache_color}{BOLD}{cache_label}{RESET}"
            f"{SKY} : {CACHE_DIR}{RESET}"
        )
        print(" " * left_pad + f"{GREY}║{RESET} " + _center_colored(cache_line, banner_width - 4) + f" {GREY}║{RESET}")
        comp_line = (
            f"{GOLD}🏆 🏆LIGUES ARCTIQUE 🏆🏆 : {BOLD}{len(COMPETITIONS_DETAILS)}{RESET}"
        )
        print(" " * left_pad + f"{GREY}║{RESET} " + _center_colored(comp_line, banner_width - 4) + f" {GREY}║{RESET}")
        try:
            now_str = now_local().strftime("%d/%m/%Y %H:%M:%S")
        except Exception:
            now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        time_line = f"{GOLD}★ {SKY}🕒 {now_str}{GOLD} ★{RESET}"
        print(" " * left_pad + f"{GREY}║{RESET} " + _center_colored(time_line, banner_width - 4) + f" {GREY}║{RESET}")
        print(" " * left_pad + f"{GREY}╚" + "═" * (banner_width - 2) + f"╝{RESET}")
        print()
    except Exception:
        print("GOANGEL SYSTEM")
        print()

def print_section(title: str, color: str = CYAN) -> None:
    print()
    padding = max(0, _SEPARATOR_LEN - len(title))
    print(f"{color}── {title} " + "─" * padding + f"{RESET}")

def clear_screen() -> None:
    try:
        os.system("clear" if os.name == "posix" else "cls")
    except Exception:
        pass
    print_banner()

def _now_local() -> datetime:
    return now_local()

def _iso_from_datetime(dt: datetime) -> str:
    return dt.strftime(_DATE_FORMAT_ISO)

def _try_full_date(ui: str) -> Optional[datetime]:
    m = _FULL_DATE_RE.match(ui)
    if not m:
        return None
    day = int(m.group(1))
    month = int(m.group(2))
    year = int(m.group(3))
    if year < _MIN_YEAR or year > _MAX_YEAR:
        return None
    try:
        return datetime(year, month, day)
    except ValueError:
        return None

def _try_day_month(ui: str, now: datetime) -> Optional[datetime]:
    m = _DAY_MONTH_RE.match(ui)
    if not m:
        return None
    day = int(m.group(1))
    month = int(m.group(2))
    year = now.year
    if year < _MIN_YEAR or year > _MAX_YEAR:
        return None
    try:
        return datetime(year, month, day)
    except ValueError:
        return None

def _try_day_only(ui: str, now: datetime) -> Optional[datetime]:
    m = _DAY_ONLY_RE.match(ui)
    if not m:
        return None
    day = int(m.group(1))
    month = now.month
    year = now.year
    if year < _MIN_YEAR or year > _MAX_YEAR:
        return None
    try:
        return datetime(year, month, day)
    except ValueError:
        return None

def parse_date_smart(user_input: str) -> Tuple[Optional[str], str, str]:
    if not isinstance(user_input, str):
        return None, "", "entrée invalide"
    ui = user_input.strip()
    if not ui:
        return None, "", "entrée vide"
    now = _now_local()
    today = now.date()
    if ui in _TODAY_COMMANDS:
        dt_ = datetime(today.year, today.month, today.day)
        return _iso_from_datetime(dt_), "today", "commande today"
    if ui in _YESTERDAY_COMMANDS:
        y = today - timedelta(days=1)
        dt_ = datetime(y.year, y.month, y.day)
        return _iso_from_datetime(dt_), "yesterday", "commande yesterday"
    if ui in _TOMORROW_COMMANDS:
        t = today + timedelta(days=1)
        dt_ = datetime(t.year, t.month, t.day)
        return _iso_from_datetime(dt_), "tomorrow", "commande tomorrow"
    full = _try_full_date(ui)
    if full is not None:
        return _iso_from_datetime(full), "full", f"date explicite {full.strftime('%d/%m/%Y')}"
    dm = _try_day_month(ui, now)
    if dm is not None:
        return _iso_from_datetime(dm), "day_month", f"jour+mois explicite, année={now.year}"
    do = _try_day_only(ui, now)
    if do is not None:
        return _iso_from_datetime(do), "auto", f"jour seul → {do.strftime('%d/%m/%Y')} (mois+année détectés depuis le système)"
    return None, "", "format non reconnu"

def _pretty_iso(date_iso: str) -> str:
    try:
        dt_ = datetime.strptime(date_iso, _DATE_FORMAT_ISO)
        return dt_.strftime("%d/%m/%Y")
    except Exception:
        return date_iso

def display_competitions() -> None:
    print_section(f"Ligues Arctique suivies ({len(COMPETITIONS_DETAILS)})", CYAN)
    for comp in COMPETITIONS_DETAILS:
        group_color = {
            "EUROPE_TOP5": LIME,
            "EUROPE_SECOND": GOLD,
            "EUROPE_OTHER": SKY,
            "AMERICAS": ORANGE,
            "ASIA_AFRICA": PINK,
            "OTHER": GREY}.get(comp.group, GREY)
        print(
            f"  {WHITE}{comp.league_id:>4}{RESET}  "
            f"{comp.name:<38} "
            f"{GREY}Tier {comp.tier}{RESET} — "
            f"{group_color}{comp.group}{RESET}"
        )

def display_help() -> None:
    print_section("Commandes disponibles", CYAN)
    rows = [
        ("07", "Jour seul → mois+année détectés automatiquement"),
        ("07/12", "Jour+mois explicites (année courante)"),
        ("07/12/2026", "Date complète (priorité maximale)"),
        ("today", "Matchs du jour"),
        ("yesterday", "Matchs d'hier"),
        ("tomorrow", "Matchs de demain"),
        ("refresh", "Reconstruire les données et réentraîner"),
        ("status", "État du cache et des modèles"),
        ("competitions", "Afficher les ligues suivies"),
        ("clear", "Effacer l'écran"),
        ("help", "Afficher cette aide"),
        ("quit / exit / q", "Quitter"),
    ]
    for cmd, desc in rows:
        print(f"  {YELLOW}{cmd:<16s}{RESET} {desc}")

def display_status(state: Dict[str, Any]) -> None:
    print_section("État du système", CYAN)
    ml_models = state.get("ml_models")
    pb_models = state.get("pb_models")
    df_past = state.get("df_past")
    pipeline = state.get("pipeline")
    weights = state.get("weights") or {}
    n_ml = len(ml_models) if isinstance(ml_models, dict) else 0
    n_pb = len(pb_models) if isinstance(pb_models, list) else 0
    try:
        n_rows = int(len(df_past)) if df_past is not None else 0
    except Exception:
        n_rows = 0
    print(f"  {SKY}Cache :{RESET} {CACHE_DIR}")
    print(f"  {SKY}Checkpoints :{RESET} {CHECKPOINT_DIR}")
    print(f"  {SKY}Stages :{RESET} {STAGES_DIR}")
    print(f"  {SKY}Persistance :{RESET} {'PERMANENTE' if CACHE_NEVER_EXPIRES else 'temporaire'}")
    print(f"  {SKY}Manifest :{RESET} {'présent' if MANIFEST_FILE.exists() else 'absent'}")
    print(f"  {SKY}Schéma :{RESET} {ARTIFACT_SCHEMA_VERSION}")
    print(f"  {SKY}Fingerprint :{RESET} {CONFIG_FINGERPRINT[:16]}...")
    print(f"  {SKY}Source :{RESET} Bzzoiro")
    print(f"  {SKY}Tokens :{RESET} {TOKEN_COUNT}")
    print(f"  {SKY}Workers :{RESET} {BZZOIRO_WORKERS_PER_KEY}/clé (max {BZZOIRO_MAX_WORKERS})")
    print(f"  {SKY}Saisons lookback :{RESET} {HISTORICAL_SEASONS_LOOKBACK} | xG : {'OUI' if BZZOIRO_ENRICH_XG else 'NON'}")
    print()
    print(f"  {LIME}Matchs historiques :{RESET} {n_rows}")
    print(f"  {LIME}Modèles ML :{RESET} {n_ml}")
    print(f"  {LIME}Modèles Penaltyblog :{RESET} {n_pb}")
    print(f"  {LIME}Pipeline :{RESET} {'OK' if pipeline is not None else 'absent'}")
    if isinstance(weights, dict) and weights:
        temperature = weights.get("__temperature__")
        if temperature is not None:
            try:
                print(f"  {LIME}Température :{RESET} {float(temperature):.4f}")
            except (TypeError, ValueError):
                pass
        test_ll = weights.get("__test_logloss__")
        test_acc = weights.get("__test_accuracy__")
        if test_ll is not None:
            try:
                print(f"  {LIME}Test LogLoss :{RESET} {float(test_ll):.4f}")
            except (TypeError, ValueError):
                pass
        if test_acc is not None:
            try:
                print(f"  {LIME}Test Accuracy :{RESET} {float(test_acc) * 100:.1f}%")
            except (TypeError, ValueError):
                pass
    print()
    if n_rows >= MIN_MATCHES_FOR_TRAINING and (n_ml > 0 or n_pb > 0):
        print(f"  {GREEN}Statut : prêt pour la prédiction{RESET}")
    else:
        print(f"  {YELLOW}Statut : mode dégradé{RESET}")

def _handle_sigint(signum: int, frame: Any) -> None:
    if _SHUTDOWN_FLAG["requested"]:
        print(f"\n{RED}Arrêt forcé.{RESET}")
        os._exit(130)
    _SHUTDOWN_FLAG["requested"] = True
    print(f"\n{YELLOW}Interruption détectée. Appuyez de nouveau sur Ctrl+C pour forcer, ou tapez 'quit'.{RESET}")

def _handle_sigterm(signum: int, frame: Any) -> None:
    logger.info("SIGTERM reçu, arrêt propre.")
    print(f"\n{YELLOW}Arrêt demandé.{RESET}")
    sys.exit(0)

def _install_signal_handlers() -> None:
    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except (ValueError, OSError):
        pass
    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except (ValueError, OSError, AttributeError):
        pass

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _atomic_write_json(path: Path, data: Any) -> None:
    if path.exists() and path.is_dir():
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Lecture %s échouée : %s", path, exc)
        return None

def _stage_done_path(stage: str) -> Path:
    return DONE_DIR / f"{stage}.done"

def _stage_attempts_path(stage: str) -> Path:
    return ATTEMPTS_DIR / f"{stage}.json"

def _read_attempts(stage: str) -> Dict[str, Any]:
    data = _read_json(_stage_attempts_path(stage))
    if not isinstance(data, dict):
        return {"count": 0, "history": []}
    return {
        "count": int(data.get("count", 0)),
        "history": list(data.get("history", []))}

def _increment_attempts(stage: str) -> int:
    data = _read_attempts(stage)
    count = data["count"] + 1
    history = data["history"] + [datetime.now().isoformat()]
    _atomic_write_json(
        _stage_attempts_path(stage),
        {"count": count, "history": history})
    return count

def _stage_artifacts(stage: str) -> List[Path]:
    if stage == "train_stacking":
        return [MODELS_DIR / "ml_Stacking.pkl"]
    if stage.startswith("finalize_") and stage not in ("finalize_prep", "finalize_agg"):
        _pname = _FINALIZE_STAGE_TO_PARTIAL.get(stage[len("finalize_"):]) or _FINALIZE_PB_MODELS.get(stage)
        if _pname:
            return [Path(CACHE_DIR) / "stages" / "finalize" / f"{_pname}.pkl"]
    if stage == "prepare":
        return [_FEATURES_FILE, _METADATA_FILE, _SELECTION_FILE, Path(CACHE_DIR) / "historical.pkl"]
    if stage == "refresh":
        return [Path(CACHE_DIR) / "historical.pkl"]
    if stage in _ML_STAGE_KEYS:
        return [MODELS_DIR / f"ml_{_ML_STAGE_KEYS[stage]}.pkl"]
    if stage in _PB_STAGE_KEYS:
        return [MODELS_DIR / f"pb_{_PB_STAGE_KEYS[stage]}.pkl"]
    if stage == "finalize":
        return [
            Path(CACHE_DIR) / "historical.pkl",
            Path(CACHE_DIR) / "models.pkl",
            Path(CACHE_DIR) / "scaler.pkl",
            Path(CACHE_DIR) / "weights.pkl",
        ]
    if stage == "finalize_prep":
        return [Path(CACHE_DIR) / "stages" / "finalize" / "context.pkl"]
    if stage == "finalize_agg":
        return [
            Path(CACHE_DIR) / "models.pkl",
            Path(CACHE_DIR) / "scaler.pkl",
            Path(CACHE_DIR) / "weights.pkl",
        ]
    if stage.startswith("finalize_"):
        model_key = stage[len("finalize_"):]
        mapping = {
            "xgb": "XGB", "lgbm": "LGBM", "catboost": "CatBoost", "rf": "RF", "dl": "DL",
            "dixon_coles_dynamic": "DixonColesDynamicModel", "kalman": "KalmanBayesianModel",
            "hierarchical_advi": "HierarchicalBayesianADVIModel", "monte_carlo": "MonteCarloSimulator"}
        if model_key in mapping:
            model_key = mapping[model_key]
        elif stage in _FINALIZE_PB_MODELS:
            model_key = _FINALIZE_PB_MODELS[stage]
        return [Path(CACHE_DIR) / "stages" / "finalize" / f"{model_key}.pkl"]
    return []

def _is_stage_done(stage: str) -> bool:
    done_path = _stage_done_path(stage)
    data = _read_json(done_path)
    if not isinstance(data, dict):
        return False
    if data.get("stage") != stage:
        return False
    return True

def _mark_stage_done(stage: str) -> bool:
    artifacts = _stage_artifacts(stage)
    if not artifacts:
        return False
    # Sécurité anti-dossier (IsADirectoryError)
    done_file = _stage_done_path(stage)
    if done_file.exists() and done_file.is_dir():
        import shutil
        shutil.rmtree(done_file, ignore_errors=True)

    hashes: Dict[str, str] = {}
    for path in artifacts:
        if not path.exists():
            logger.error("Artifact manquant pour %s : %s", stage, path)
            return False
        try:
            hashes[str(path)] = _sha256_file(path)
        except Exception as exc:
            logger.error("Hash échoué pour %s : %s", path, exc)
            return False
    payload = {
        "stage": stage,
        "files": hashes,
        "ts": datetime.now().isoformat(),
        "schema": ARTIFACT_SCHEMA_VERSION}
    _atomic_write_json(_stage_done_path(stage), payload)
    logger.info("Stage %s marqué done (%d fichiers)", stage, len(hashes))
    return True

def _next_stage() -> str:
    for stage in _STAGES_ORDER:
        if not _is_stage_done(stage):
            return stage
    return "DONE"

def _reset_all_progress() -> None:
    for stage in _STAGES_ORDER:
        done_p = _stage_done_path(stage)
        attempts_p = _stage_attempts_path(stage)
        for p in (done_p, attempts_p):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
    try:
        if Path(CACHE_DIR).exists():
            for f in Path(CACHE_DIR).glob("stages/*.pkl"):
                f.unlink()
    except OSError:
        pass
    logger.info("Progress complet réinitialisé")

def _safe_reload_models(state: Dict[str, Any], force_refresh: bool) -> bool:
    try:
        from predict import load_or_build_data_and_models
        result = load_or_build_data_and_models(force_refresh=force_refresh)
    except TypeError as exc:
        logger.error("Signature incompatible : %s", exc)
        print(f"{RED}Signature incompatible : {exc}{RESET}")
        return False
    except Exception as exc:
        logger.error("Échec chargement : %s", exc)
        print(f"{RED}Échec du chargement des modèles : {exc}{RESET}")
        return False
    if not isinstance(result, tuple) or len(result) != _TUPLE_LEN_EXPECTED:
        print(f"{RED}Signature de chargement invalide.{RESET}")
        return False
    ml_models, pb_models, pipeline, df_past, weights = result
    if df_past is None or getattr(df_past, "empty", True):
        print(f"{YELLOW}⚠️  Aucun historique exploitable.{RESET}")
    else:
        n_rows = int(len(df_past))
        print(f"{GREEN}📚 Historique chargé : {n_rows} matchs.{RESET}")
    if not ml_models and not pb_models:
        print(f"{YELLOW}⚠️  Aucun modèle entraîné.{RESET}")
    else:
        n_ml = len(ml_models) if isinstance(ml_models, dict) else 0
        n_pb = len(pb_models) if isinstance(pb_models, list) else 0
        print(f"{GREEN}🤖 Modèles actifs : {n_ml} ML + {n_pb} Penaltyblog.{RESET}")
    if weights is None:
        weights = {}
    state["ml_models"] = ml_models
    state["pb_models"] = pb_models
    state["pipeline"] = pipeline
    state["df_past"] = df_past
    state["weights"] = weights
    return True

def _predict_date(state: Dict[str, Any], date_iso: str) -> bool:
    try:
        from predict import predict_for_date
        predict_for_date(
            date_iso,
            state.get("ml_models"),
            state.get("pb_models"),
            state.get("pipeline"),
            state.get("df_past"),
            state.get("weights", {}))
        return True
    except Exception as exc:
        logger.error("Erreur prédiction : %s", exc)
        print(f"{RED}Erreur lors de la prédiction : {exc}{RESET}")
        return False

def _stage_save_features(df_past: pd.DataFrame) -> None:
    X, y, home_keys, away_keys, gh_all, ga_all, comps, h2h = create_features_with_context(
        df_past, strict_competition=True
    )
    if X.shape[0] < MIN_MATCHES_FOR_TRAINING:
        raise RuntimeError(f"Features insuffisantes : {X.shape[0]} < {MIN_MATCHES_FOR_TRAINING}")
    if X.shape[1] != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(f"Features shape invalide : {X.shape[1]} != {EXPECTED_FEATURE_COUNT}")
    dates = df_past["date_parsed"].tolist() if "date_parsed" in df_past.columns else []
    payload = {
        "X": X,
        "y": y,
        "home_keys": home_keys,
        "away_keys": away_keys,
        "gh_all": gh_all,
        "ga_all": ga_all,
        "comps": comps,
        "h2h": h2h,
        "dates": dates,
        "n": len(X),
        "created_at": datetime.now().isoformat()}
    _atomic_write_pickle(_FEATURES_FILE, payload)
    meta = {
        "n_matchs": len(df_past),
        "n_features": X.shape[1],
        "n_samples": len(X),
        "class_counts": np.bincount(y, minlength=3).tolist(),
        "created_at": datetime.now().isoformat()}
    _atomic_write_pickle(_METADATA_FILE, meta)
    stage_print(f"💾 Features : {len(X)} samples × {X.shape[1]} features")
    stage_print(f"   Classes : {meta['class_counts']}")

def _mark_stage_done_safe(model_name: str) -> None:
    """Marque un stage comme done sans crash si l'artifact n'est pas trouvé."""
    try:
        from pathlib import Path
        done_dir = Path(CACHE_DIR) / "stages" / "done"
        done_dir.mkdir(parents=True, exist_ok=True)
        done_file = done_dir / f"train_{model_name.lower()}.done"
        import json
        from datetime import datetime
        done_file.write_text(json.dumps({
            "stage": f"train_{model_name.lower()}",
            "skipped": True,
            "reason": "model_unavailable",
            "ts": datetime.now().isoformat()
        }), encoding="utf-8")
    except Exception:
        pass

def _apply_sample_weights(y):
    from config import DRAW_SAMPLE_WEIGHT, HOME_SAMPLE_WEIGHT, AWAY_SAMPLE_WEIGHT
    weights = np.ones(len(y), dtype=np.float64)
    weights[y == 0] = HOME_SAMPLE_WEIGHT
    weights[y == 1] = DRAW_SAMPLE_WEIGHT
    weights[y == 2] = AWAY_SAMPLE_WEIGHT
    return weights

def _stage_load_features() -> Dict[str, Any]:
    if not _FEATURES_FILE.exists():
        raise RuntimeError(f"Stage prepare manquant : {_FEATURES_FILE}")
    payload = _atomic_read_pickle(_FEATURES_FILE)
    return payload

def _stage_train_pb_model(class_name: str) -> None:
    from pb_convergence import fit_pb_attempt, PBIncompleteError

    payload = _stage_load_features()
    X = payload["X"]
    home_keys = payload["home_keys"]
    away_keys = payload["away_keys"]
    gh_all = payload["gh_all"]
    ga_all = payload["ga_all"]
    y = payload["y"]
    n = len(X)
    cut_train = int(n * TRAIN_RATIO)
    cut_train = max(1, min(cut_train, n - 4))
    gh_train = gh_all[:cut_train]
    ga_train = ga_all[:cut_train]
    th_train = home_keys[:cut_train]
    ta_train = away_keys[:cut_train]
    # dates alignées 1:1 sur les lignes de features ET tronquées au TRAIN (avant : liste complète ->
    # longueurs incohérentes avec gh_train => pondération temporelle jamais appliquée / stage en erreur)
    _dates_all = payload.get("dates")
    dates_train = list(_dates_all[:cut_train]) if _dates_all is not None and len(_dates_all) == n else None
    # ─── HMM (modèle statistique temporel + calibration isotonic) ───
    if class_name == "HMMModel":
        from features import HMMModel
        from sklearn.metrics import log_loss as _ll

        cut_val = int(n * (TRAIN_RATIO + VAL_RATIO))
        cut_calib = int(n * (TRAIN_RATIO + VAL_RATIO + CALIB_RATIO))
        cut_val = max(cut_train + 1, min(cut_val, n - 3))
        cut_calib = max(cut_val + 1, min(cut_calib, n - 2))

        gh_calib = gh_all[cut_val:cut_calib]
        ga_calib = ga_all[cut_val:cut_calib]
        th_calib = home_keys[cut_val:cut_calib]
        ta_calib = away_keys[cut_val:cut_calib]
        y_calib_pb = y[cut_val:cut_calib]

        stage_print(f"   HMM : fit sur {len(gh_train)} matchs, calibration sur {len(gh_calib)}")
        m = HMMModel()
        m.fit(gh_train, ga_train, th_train, ta_train)

        # Calibration isotonic interne
        try:
            m._fit_calibrators(th_calib, ta_calib, y_calib_pb)
            calib_status = "calibrated_isotonic"
            stage_print(f"   ✅ HMM calibré (isotonic, n={len(y_calib_pb)})")
        except Exception as cal_exc:
            calib_status = f"calibration_failed: {cal_exc}"
            stage_print(f"   ⚠️ HMM calibration échouée : {cal_exc}")

        # LogLoss sur calib
        try:
            import pandas as pd
            X_calib_df = pd.DataFrame({"home_key": th_calib, "away_key": ta_calib})
            proba = m.predict_proba(X_calib_df)
            ll_val = float(_ll(y_calib_pb, proba, labels=[0, 1, 2]))
            acc = float((proba.argmax(axis=1) == y_calib_pb).mean())
            stage_print(f"✅ HMM : LogLoss={ll_val:.4f} | Accuracy={acc*100:.1f}% (calib n={len(y_calib_pb)})")
        except Exception as exc:
            stage_print(f"⚠️ HMM : LogLoss indisponible ({exc})")

        model_path = MODELS_DIR / f"pb_{class_name}.pkl"
        _atomic_write_pickle(model_path, {"model": m, "type": class_name, "meta": {}})
        save_ml_trained(class_name, str(model_path))
        save_ml_calibrated(class_name, calib_status)
        stage_print(f"💾 {class_name} sauvegardé : {model_path}")
        return
    # ─── HMM (modèle statistique temporel + calibration isotonic) ───
    if class_name == "HMMModel":
        from features import HMMModel
        from sklearn.metrics import log_loss as _ll

        cut_val = int(n * (TRAIN_RATIO + VAL_RATIO))
        cut_calib = int(n * (TRAIN_RATIO + VAL_RATIO + CALIB_RATIO))
        cut_val = max(cut_train + 1, min(cut_val, n - 3))
        cut_calib = max(cut_val + 1, min(cut_calib, n - 2))

        gh_calib = gh_all[cut_val:cut_calib]
        ga_calib = ga_all[cut_val:cut_calib]
        th_calib = home_keys[cut_val:cut_calib]
        ta_calib = away_keys[cut_val:cut_calib]
        y_calib_pb = y[cut_val:cut_calib]

        stage_print(f"   HMM : fit sur {len(gh_train)} matchs, calibration sur {len(gh_calib)}")
        m = HMMModel()
        m.fit(gh_train, ga_train, th_train, ta_train)

        # Calibration isotonic interne
        try:
            m._fit_calibrators(th_calib, ta_calib, y_calib_pb)
            calib_status = "calibrated_isotonic"
            stage_print(f"   ✅ HMM calibré (isotonic, n={len(y_calib_pb)})")
        except Exception as cal_exc:
            calib_status = f"calibration_failed: {cal_exc}"
            stage_print(f"   ⚠️ HMM calibration échouée : {cal_exc}")

        # LogLoss sur calib
        try:
            import pandas as pd
            X_calib_df = pd.DataFrame({"home_key": th_calib, "away_key": ta_calib})
            proba = m.predict_proba(X_calib_df)
            ll_val = float(_ll(y_calib_pb, proba, labels=[0, 1, 2]))
            acc = float((proba.argmax(axis=1) == y_calib_pb).mean())
            stage_print(f"✅ HMM : LogLoss={ll_val:.4f} | Accuracy={acc*100:.1f}% (calib n={len(y_calib_pb)})")
        except Exception as exc:
            stage_print(f"⚠️ HMM : LogLoss indisponible ({exc})")

        model_path = MODELS_DIR / f"pb_{class_name}.pkl"
        _atomic_write_pickle(model_path, {"model": m, "type": class_name, "meta": {}})
        save_ml_trained(class_name, str(model_path))
        save_ml_calibrated(class_name, calib_status)
        stage_print(f"💾 {class_name} sauvegardé : {model_path}")
        return



    from features import (DixonColesDynamicModel, KalmanBayesianModel,
                          HierarchicalBayesianADVIModel, MonteCarloSimulator)
    if class_name == "SkellamModel":
        from features import SkellamModel
        m = SkellamModel()
        m.fit(gh_train, ga_train, th_train, ta_train, dates_train)
        model_path = MODELS_DIR / f"pb_{class_name}.pkl"
        _atomic_write_pickle(model_path, {"model": m, "type": class_name, "meta": {}})
        save_ml_trained(class_name, str(model_path))
        save_ml_calibrated(class_name, "built_in")
        stage_print(f"💾 {class_name} sauvegardé : {model_path}")
        return
    elif class_name == "CopulaModel":
        from features import CopulaModel
        m = CopulaModel()
        m.fit(gh_train, ga_train, th_train, ta_train, dates_train)
        model_path = MODELS_DIR / f"pb_{class_name}.pkl"
        _atomic_write_pickle(model_path, {"model": m, "type": class_name, "meta": {}})
        save_ml_trained(class_name, str(model_path))
        save_ml_calibrated(class_name, "built_in")
        stage_print(f"💾 {class_name} sauvegardé : {model_path}")
        return
    elif class_name == "HMMModel":
        from features import HMMModel
        m = HMMModel()
        m.fit(gh_train, ga_train, th_train, ta_train)
        model_path = MODELS_DIR / f"pb_{class_name}.pkl"
        _atomic_write_pickle(model_path, {"model": m, "type": class_name, "meta": {}})
        save_ml_trained(class_name, str(model_path))
        save_ml_calibrated(class_name, "built_in")
        stage_print(f"💾 {class_name} sauvegardé : {model_path}")
        return
    elif class_name == "DixonColesDynamicModel":
        m = DixonColesDynamicModel()
        m.fit(gh_train, ga_train, th_train, ta_train, dates_train)
    elif class_name == "KalmanBayesianModel":
        m = KalmanBayesianModel()
        m.fit(gh_train, ga_train, th_train, ta_train, dates_train)
    elif class_name == "HierarchicalBayesianADVIModel":
        m = HierarchicalBayesianADVIModel()
        m.fit(gh_train, ga_train, th_train, ta_train, dates_train)
    elif class_name == "MonteCarloSimulator":
        m = MonteCarloSimulator(intensity_home=float(np.mean(gh_train)), intensity_away=float(np.mean(ga_train)))
        m.fit()
    else:
        m = None

    if m is not None:
        model_path = MODELS_DIR / f"pb_{class_name}.pkl"
        _atomic_write_pickle(model_path, {"model": m, "type": class_name, "meta": {}})
        save_ml_trained(class_name, str(model_path))
        save_ml_calibrated(class_name, "built_in")
        stage_print(f"💾 {class_name} sauvegardé : {model_path}")
        return

    model_cls = _PB_MODEL_CLASSES.get(class_name)
    if model_cls is None:
        raise RuntimeError(f"Modèle penaltyblog inconnu : {class_name}")
    is_bayesian = class_name in _PB_BAYESIAN_MODELS

    stage_print(
        f"🧠 Tentative {class_name} (bayesian={is_bayesian}) "
        f"sur {len(gh_train)} matchs..."
    )
    status, model, meta = fit_pb_attempt(
        class_name=class_name,
        model_class=model_cls,
        gh=gh_train,
        ga=ga_train,
        th=th_train,
        ta=ta_train,
        dates=dates_train,
        is_bayesian=is_bayesian)
    stage_print(
        f"   → attempt={meta.get('attempt')} filter={meta.get('filter')} "
        f"n_kept={meta.get('n_kept')} status={status}"
    )

    if status == "ok" and model is not None:
        model_path = MODELS_DIR / f"pb_{class_name}.pkl"
        if model_path.exists():
            model_path.unlink()
        _atomic_write_pickle(
            model_path, {"model": model, "type": class_name, "meta": meta}
        )
        stage_print(f"💾 {class_name} sauvegardé : {model_path}")
        return

    if status == "incomplete":
        raise PBIncompleteError(
            f"{class_name} : non convergé ({meta.get('error', 'unknown')})"
        )

    raise RuntimeError(
        f"Erreur réelle sur {class_name} : {meta.get('error', 'unknown')}"
    )

# ═══════════════════════════════════════════════════════════════════════════
# FINALIZE ÉCLATÉ (15 stages indépendants)
# ═══════════════════════════════════════════════════════════════════════════

_FINALIZE_ML_MODELS = (
    "XGB", "LGBM", "CatBoost", "RF", "DL",
    "LR", "ET", "SVM", "HistGB", "MLP",
    "TabM", "RealMLP", "ExcelFormer",
    "NGBoost", "HMM")

_FINALIZE_PB_MODELS = {
    "finalize_pb_poisson": "PoissonGoalsModel", "finalize_pb_dixon_coles": "DixonColesGoalModel",
    "finalize_pb_bivariate": "BivariatePoissonGoalModel", "finalize_pb_negbinomial": "NegativeBinomialGoalModel",
    "finalize_pb_zeroinflated": "ZeroInflatedPoissonGoalsModel", "finalize_pb_weibull": "WeibullCopulaGoalsModel",
    "finalize_dixon_coles_dynamic": "DixonColesDynamicModel", "finalize_kalman": "KalmanBayesianModel",
    "finalize_hierarchical_advi": "HierarchicalBayesianADVIModel", "finalize_monte_carlo": "MonteCarloSimulator",
    "finalize_skellam": "SkellamModel", "finalize_copula": "CopulaModel"}

def _load_payload_features():
    import pickle
    payload = _stage_load_features()
    X = payload["X"]
    y = payload["y"]
    home_keys = payload["home_keys"]
    away_keys = payload["away_keys"]
    gh_all = payload["gh_all"]
    ga_all = payload["ga_all"]
    return X, y, home_keys, away_keys, gh_all, ga_all

def _compute_splits(X, y, home_keys, away_keys, gh_all, ga_all):
    n = len(X)
    cut_train = int(n * TRAIN_RATIO)
    cut_val = int(n * (TRAIN_RATIO + VAL_RATIO))
    cut_calib = int(n * (TRAIN_RATIO + VAL_RATIO + CALIB_RATIO))
    cut_train = max(1, min(cut_train, n - 4))
    cut_val = max(cut_train + 1, min(cut_val, n - 3))
    cut_calib = max(cut_val + 1, min(cut_calib, n - 2))
    return {
        "n": n,
        "cut_train": cut_train,
        "cut_val": cut_val,
        "cut_calib": cut_calib,
        "X_train_raw": X[:cut_train].astype(np.float64, copy=True),
        "X_calib_raw": X[cut_val:cut_calib].astype(np.float64, copy=True),
        "X_test_raw": X[cut_calib:].astype(np.float64, copy=True),
        "y_train": y[:cut_train],
        "y_calib": y[cut_val:cut_calib],
        "y_test": y[cut_calib:],
        "home_keys_train": home_keys[:cut_train],
        "away_keys_train": away_keys[:cut_train],
        "home_keys_calib": home_keys[cut_val:cut_calib],
        "away_keys_calib": away_keys[cut_val:cut_calib],
        "home_keys_test": home_keys[cut_calib:],
        "away_keys_test": away_keys[cut_calib:],
        "gh_train": gh_all[:cut_train],
        "ga_train": ga_all[:cut_train]}

# ═══════════════════════════════════════════════════════════════════════════
# V3 — SPLITS / PREPROCESSING / CALIBRATION / ENSEMBLE (sans fuite, dimension unique 251)
# ═══════════════════════════════════════════════════════════════════════════
_MODEL_SCHEMA: str = "goangel-core-v4"


def _cfg(name: str, default: Any) -> Any:
    import config as _c
    return getattr(_c, name, default)


_SELECTION_FILE: Path = STAGES_DIR / "feature_selection.json"


def _split_cuts(n: int) -> Tuple[int, int, int]:
    c_tr = max(1, min(int(n * TRAIN_RATIO), n - 4))
    c_va = max(c_tr + 1, min(int(n * (TRAIN_RATIO + VAL_RATIO)), n - 3))
    c_ca = max(c_va + 1, min(int(n * (TRAIN_RATIO + VAL_RATIO + CALIB_RATIO)), n - 2))
    return c_tr, c_va, c_ca


def _compute_and_save_selection() -> Dict[str, Any]:
    """Sélection des K features les plus puissantes, calculée UNE FOIS sur le TRAIN chronologique
    seul (anti-fuite) puis PERSISTÉE : tous les stages (processus séparés) et l'inférence utilisent
    exactement les mêmes colonnes. Recalculée uniquement par prepare/refresh (nouvelles features)."""
    from features import _compute_diff_features, FEATURE_NAMES_FULL
    from goangel_core import FeatureSelector

    payload = _stage_load_features()
    X = _compute_diff_features(payload["X"])
    if X.shape[1] != len(FEATURE_NAMES_FULL):
        raise RuntimeError(f"Noms de features ({len(FEATURE_NAMES_FULL)}) != colonnes ({X.shape[1]})")
    y = np.asarray(payload["y"])
    c_tr = _split_cuts(len(X))[0]
    X_raw = np.where(np.isfinite(X), X, np.nan).astype(np.float64)
    stage_print(f"🎯 SÉLECTION DES FEATURES : {X.shape[1]} candidates, train={c_tr} lignes (aucune donnée val/calib/test)")
    t0 = time.time()
    fs = FeatureSelector(
        k=int(_cfg("FEATURE_SELECTION_K", 100)), min_k=int(_cfg("FEATURE_SELECTION_MIN_K", 60)),
        min_coverage=float(_cfg("FEATURE_SELECTION_MIN_COVERAGE", 0.30)),
        corr_threshold=float(_cfg("FEATURE_SELECTION_CORR", 0.97)),
        seed=MODEL_RANDOM_SEED).fit(X_raw[:c_tr], y[:c_tr], FEATURE_NAMES_FULL)
    rep = fs.report_
    data = {"schema": _MODEL_SCHEMA, "selection_id": fs.selection_id_, "selected_idx": fs.selected_idx_,
            "selected_names": fs.selected_names_, "n_features": int(X.shape[1]), "n_samples": int(len(X)),
            "train_n": int(c_tr), "report": rep, "created_at": datetime.now().isoformat()}
    _atomic_write_json(_SELECTION_FILE, data)
    stage_print(f"   ✅ {rep['n_selected']} features retenues sur {rep['n_usable']} utilisables "
                f"({rep['n_qualified']} qualifiées, {rep['n_duplicates_removed']} doublons écartés, "
                f"{rep['n_diff_selected']} différentielles) en {time.time() - t0:.0f}s | id={fs.selection_id_}")
    if rep["n_topped_up_below_min_k"] > 0:
        stage_print(f"   ⚠️ {rep['n_topped_up_below_min_k']} features complétées sous le seuil min_k (signal faible)")
    for nm, pm_, mi_ in rep["top20"][:10]:
        stage_print(f"      {nm:<34} permutation ΔLL={pm_:+.5f}  MI={mi_:.5f}")
    return data


def _load_or_compute_selection() -> Dict[str, Any]:
    d = _read_json(_SELECTION_FILE)
    if isinstance(d, dict) and d.get("selected_idx") and d.get("schema") == _MODEL_SCHEMA:
        return d
    return _compute_and_save_selection()


def _ece(y: np.ndarray, P: np.ndarray, bins: int = 10) -> float:
    """Expected Calibration Error moyen one-vs-rest (0 = calibration parfaite)."""
    y = np.asarray(y).astype(int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    tot = 0.0
    for k in range(3):
        p = P[:, k]
        t = (y == k).astype(float)
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        e = 0.0
        for b in range(bins):
            m = idx == b
            if m.any():
                e += float(m.mean()) * abs(float(p[m].mean()) - float(t[m].mean()))
        tot += e
    return tot / 3.0


def _build_splits() -> Dict[str, Any]:
    """Splits chronologiques train|val|calib|test + préprocesseur ajusté sur le TRAIN seul.
    Déterministe : chaque stage (processus séparé) reconstruit EXACTEMENT les mêmes matrices."""
    from features import _compute_diff_features, EXPECTED_FEATURE_COUNT_FULL
    from goangel_core import GoPreprocessor

    payload = _stage_load_features()
    X = _compute_diff_features(payload["X"])          # idempotent : 172 -> 251, 251 -> 251
    if X.shape[1] != EXPECTED_FEATURE_COUNT_FULL:
        raise RuntimeError(f"Dimension features {X.shape[1]} != {EXPECTED_FEATURE_COUNT_FULL}")
    y = np.asarray(payload["y"])
    hk, ak = payload["home_keys"], payload["away_keys"]
    n = len(X)
    c_tr = max(1, min(int(n * TRAIN_RATIO), n - 4))
    c_va = max(c_tr + 1, min(int(n * (TRAIN_RATIO + VAL_RATIO)), n - 3))
    c_ca = max(c_va + 1, min(int(n * (TRAIN_RATIO + VAL_RATIO + CALIB_RATIO)), n - 2))
    X_raw = np.where(np.isfinite(X), X, np.nan).astype(np.float64)
    sel = _load_or_compute_selection()
    if sel["n_features"] != X.shape[1]:
        raise RuntimeError("feature_selection.json obsolète (dimension) — relancer prepare/refresh")
    prep = GoPreprocessor(float(FEATURE_OUTLIER_ZSCORE), selected_idx=sel["selected_idx"]).fit(X_raw[:c_tr])
    Xs = prep.transform(X_raw)                      # (n, K) : colonnes sélectionnées, standardisées
    Xr = prep.transform_raw(X_raw)                  # (n, K) : colonnes sélectionnées, NaN conservés
    sl = {"train": slice(0, c_tr), "val": slice(c_tr, c_va),
          "calib": slice(c_va, c_ca), "test": slice(c_ca, n)}
    out: Dict[str, Any] = {"n": n, "cuts": (c_tr, c_va, c_ca), "prep": prep, "selection_id": sel["selection_id"],
                           "gh_train": payload["gh_all"][:c_tr], "ga_train": payload["ga_all"][:c_tr]}
    for part, s in sl.items():
        out[f"raw_{part}"] = Xr[s]
        out[f"sc_{part}"] = Xs[s]
        out[f"y_{part}"] = y[s]
        out[f"hk_{part}"] = hk[s]
        out[f"ak_{part}"] = ak[s]
    return out


def _ml_input(model_name: str, sp: Dict[str, Any], part: str) -> np.ndarray:
    """Arbres NaN-natifs -> brut ; tous les autres -> imputé/clippé/standardisé."""
    return sp[f"raw_{part}"] if model_name in _TREE_MODELS_NAN_NATIVE else sp[f"sc_{part}"]


def _train_torch_classifier(sp: Dict[str, Any], seed: int, hidden: int):
    if not DL_ENABLED or torch is None or nn is None or _TorchNet is None or _TorchProbaWrapper is None:
        raise RuntimeError("Deep Learning indisponible")
    Xtr, ytr, Xva, yva = sp["sc_train"], sp["y_train"], sp["sc_val"], sp["y_val"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _TorchNet(Xtr.shape[1], hidden_size=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=DL_LEARNING_RATE, weight_decay=DL_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()
    Xt, yt = torch.FloatTensor(Xtr), torch.LongTensor(ytr)
    Xv, yv = torch.FloatTensor(Xva), torch.LongTensor(yva)      # early stopping sur VAL (pas calib)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=DL_BATCH_SIZE, shuffle=True,
                        drop_last=len(Xt) > DL_BATCH_SIZE, generator=gen)
    best, best_state, patience = float("inf"), None, 0
    for _ in range(DL_EPOCHS):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            crit(model(bx), by).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), DL_GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        sched.step(vl)
        if vl < best:
            best, patience = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= DL_EARLY_STOPPING_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return _TorchProbaWrapper(model)


def _stage_train_ml_model(model_name: str) -> None:
    from goangel_core import (fit_guarded_calibrator, PrefitCalibratedClassifier,
                              safe_log_loss, prior_log_loss)

    model_path = MODELS_DIR / f"ml_{model_name}.pkl"
    if ck_has_step(ml_key(model_name), STEP_TRAINED) and model_path.exists():
        try:
            _old = _atomic_read_pickle(model_path)
            if (_old.get("schema") == _MODEL_SCHEMA
                    and _old.get("selection_id") == _load_or_compute_selection()["selection_id"]):
                stage_print(f"📦 {model_name} déjà entraîné (schéma {_MODEL_SCHEMA}, mêmes features)")
                return
            stage_print(f"♻️  {model_name} : ancien schéma / autre sélection -> ré-entraînement")
        except Exception:
            pass
    if model_path.exists():
        model_path.unlink()

    sp = _build_splits()
    y_tr, y_va, y_ca, y_te = sp["y_train"], sp["y_val"], sp["y_calib"], sp["y_test"]
    stage_print(f"📊 {model_name} : train={len(y_tr)} val={len(y_va)} calib={len(y_ca)} "
                f"test={len(y_te)} | {sp['sc_train'].shape[1]} features (dimension unique)")
    X_fit = _ml_input(model_name, sp, "train")
    X_va = _ml_input(model_name, sp, "val")
    sample_w = _apply_sample_weights(y_tr)
    # patience 50 à lr=0.01 s'arrête sur le bruit du LogLoss VAL (~2k lignes) : sous-apprentissage
    _es = int(_cfg("EARLY_STOP_ROUNDS", 200))

    if model_name in ("DL", "MLP"):
        base = _train_torch_classifier(
            sp, seed=MODEL_RANDOM_SEED + (0 if model_name == "DL" else 17),
            hidden=int(_cfg("DL_HIDDEN_SIZE", 128)) * (1 if model_name == "DL" else 2))
    elif model_name == "HistGB":
        from sklearn.ensemble import HistGradientBoostingClassifier as _HGB
        from config import (HIST_N_ESTIMATORS, HIST_LEARNING_RATE,
                            HIST_MAX_LEAF_NODES, HIST_L2_REGULARIZATION)

        def _mk(it: int):
            # max_iter (et NON n_estimators). early_stopping interne = split ALÉATOIRE -> interdit.
            return _HGB(max_iter=it, learning_rate=HIST_LEARNING_RATE,
                        max_leaf_nodes=HIST_MAX_LEAF_NODES, l2_regularization=HIST_L2_REGULARIZATION,
                        max_bins=255, early_stopping=False, random_state=MODEL_RANDOM_SEED)
        probe = _mk(HIST_N_ESTIMATORS).fit(X_fit, y_tr, sample_weight=sample_w)
        lls = [safe_log_loss(y_va, P) for P in probe.staged_predict_proba(X_va)]
        best_it = int(np.argmin(lls)) + 1                      # arrêt précoce chronologique sur VAL
        stage_print(f"   HistGB : meilleure itération sur VAL = {best_it}/{HIST_N_ESTIMATORS}")
        base = _mk(best_it).fit(X_fit, y_tr, sample_weight=sample_w)
    else:
        factories = {
            "XGB": lambda: _make_xgb_factory()(), "LGBM": lambda: _make_lgbm_factory()(),
            "CatBoost": lambda: _make_catboost_factory()(), "RF": lambda: _make_rf_factory()(),
            "LR": lambda: _make_lr_factory()(), "ET": lambda: _make_et_factory()(),
            "SVM": lambda: _make_svm_factory()(), "TabM": lambda: _make_tabm_factory()(),
            "RealMLP": lambda: _make_realmlp_factory()(),
            "ExcelFormer": lambda: _make_excelformer_factory()(),
            "NGBoost": lambda: _make_ngboost_factory()(),
        }
        if model_name not in factories:
            raise ValueError(f"Modèle ML inconnu : {model_name}")
        try:
            base = factories[model_name]()
        except (RuntimeError, ImportError, NameError, TypeError) as fac_exc:
            stage_print(f"⚠️ {model_name} indisponible : {fac_exc} — stage marqué skip")
            _mark_stage_done_safe(model_name)
            return
        if base is None:
            stage_print(f"⚠️ {model_name} indisponible (factory None) — stage marqué skip")
            _mark_stage_done_safe(model_name)
            return
        if model_name == "XGB":
            base.set_params(early_stopping_rounds=_es)
            base.fit(X_fit, y_tr, sample_weight=sample_w, eval_set=[(X_va, y_va)], verbose=False)
        elif model_name == "LGBM":
            import lightgbm as _lgb
            base.fit(X_fit, y_tr, sample_weight=sample_w, eval_set=[(X_va, y_va)],
                     callbacks=[_lgb.early_stopping(_es, verbose=False)])
        elif model_name == "CatBoost":
            base.set_params(early_stopping_rounds=_es)
            base.fit(X_fit, y_tr, sample_weight=sample_w, eval_set=(X_va, y_va), verbose=False)
        else:
            try:
                base.fit(X_fit, y_tr, sample_weight=sample_w)
            except TypeError:
                base.fit(X_fit, y_tr)

    # ── Calibration : sur CALIB uniquement, garde-fou hors-échantillon ──
    X_ca = _ml_input(model_name, sp, "calib")
    X_te = _ml_input(model_name, sp, "test")
    P_ca = np.asarray(base.predict_proba(X_ca), dtype=np.float64)
    cal, status, info = fit_guarded_calibrator(
        P_ca, y_ca,
        min_isotonic_samples=int(_cfg("CALIBRATION_ISOTONIC_MIN_SAMPLES", 1000)),
        min_gain=float(_cfg("CALIBRATION_GUARD_MIN_GAIN", 5e-4)))
    model = PrefitCalibratedClassifier(base, cal)

    P_va_raw = np.asarray(base.predict_proba(X_va), dtype=np.float64)
    P_te_raw = np.asarray(base.predict_proba(X_te), dtype=np.float64)
    P_va, P_te = model.predict_proba(X_va), model.predict_proba(X_te)
    prior_te = prior_log_loss(y_tr, y_te)
    ll_te = safe_log_loss(y_te, P_te)
    stage_print(
        f"✅ {model_name} [{status}] | VAL brut {safe_log_loss(y_va, P_va_raw):.4f} → "
        f"calibré {safe_log_loss(y_va, P_va):.4f} | "
        f"TEST (jamais vu) brut {safe_log_loss(y_te, P_te_raw):.4f} → calibré {ll_te:.4f} "
        f"| acc {float((P_te.argmax(1) == y_te).mean()) * 100:.1f}% | ECE {_ece(y_te, P_te):.4f} "
        f"| constant={prior_te:.4f}")

    _atomic_write_pickle(model_path, {
        "model": model, "type": model_name, "schema": _MODEL_SCHEMA, "selection_id": sp["selection_id"], "calib_status": status,
        "calib_info": info, "prep": sp["prep"]})
    save_ml_trained(model_name, str(model_path))
    save_ml_calibrated(model_name, status)
    stage_print(f"💾 {model_name} sauvegardé : {model_path}")


def _partial_is_current(path: Path) -> bool:
    try:
        if not (path.exists() and path.stat().st_size > 100):
            return False
        pl = _atomic_read_pickle(path)
        if not (isinstance(pl, dict) and pl.get("schema") == _MODEL_SCHEMA):
            return False
        if pl.get("kind") == "ml" and pl.get("selection_id") != _load_or_compute_selection()["selection_id"]:
            return False
        return True
    except Exception:
        return False


def _stage_finalize_prep() -> None:
    """Prépare le contexte partagé : même préprocesseur (251 col.) pour entraînement ET inférence."""
    from finalize_split import save_context
    from goangel_core import prior_log_loss

    stage_print("🔧 FINALIZE PREP — contexte partagé")
    sp = _build_splits()
    c_tr, c_va, c_ca = sp["cuts"]
    stage_print(f"   Split chrono : train={c_tr} val={c_va - c_tr} calib={c_ca - c_va} test={sp['n'] - c_ca}")
    stage_print(f"   Dimension unique : {sp['prep'].n_features_in_} colonnes "
                f"(keep_empty_features=True, plus de 233/248)")
    _atomic_write_pickle(MODELS_DIR / "prep.pkl", sp["prep"])
    ctx = {
        "schema": _MODEL_SCHEMA, "n": sp["n"], "cut_train": c_tr, "cut_val": c_va, "cut_calib": c_ca,
        "n_features": int(sp["prep"].n_features_in_), "pipeline": sp["prep"],
        "prior_ll_val": prior_log_loss(sp["y_train"], sp["y_val"]),
        "prior_ll_test": prior_log_loss(sp["y_train"], sp["y_test"]),
    }
    save_context(ctx)
    save_finalize_context({k: v for k, v in ctx.items() if k != "pipeline"})
    stage_print(f"   LogLoss du prédicteur constant : VAL {ctx['prior_ll_val']:.4f} | "
                f"TEST {ctx['prior_ll_test']:.4f}  (hasard pur = 1.0986)")
    stage_print("💾 Contexte sauvegardé : cache/stages/finalize/context.pkl")


def _finalize_ml_one(model_name: str) -> None:
    from finalize_split import save_partial
    from goangel_core import safe_log_loss

    partial_path = Path(CACHE_DIR) / "stages" / "finalize" / f"{model_name}.pkl"
    if _partial_is_current(partial_path):
        stage_print(f"📦 {model_name} déjà finalisé (partial à jour)")
        return
    stage_print(f"🔧 FINALIZE ML — {model_name}")
    if model_name == "Stacking":
        # Le stacking CV précédent s'entraînait sur des probas IN-SAMPLE (fuite). Remplacé par un
        # blend convexe optimisé sur VAL dans finalize_agg.
        save_partial(model_name, {"schema": _MODEL_SCHEMA, "model_key": model_name,
                                  "skipped": True, "reason": "replaced_by_blend_in_agg"})
        return
    model_path = MODELS_DIR / f"ml_{model_name}.pkl"
    if not model_path.exists():
        stage_print(f"⚠️  {model_name} : modèle absent — skip")
        save_partial(model_name, {"schema": _MODEL_SCHEMA, "model_key": model_name,
                                  "skipped": True, "reason": "model_missing"})
        return
    pl = _atomic_read_pickle(model_path)
    sp = _build_splits()
    if pl.get("schema") != _MODEL_SCHEMA or pl.get("selection_id") != sp["selection_id"]:
        raise RuntimeError(f"{model_name} : modèle d'un ancien schéma / d'une autre sélection de features — relancer train_{model_name.lower()}")
    model = pl["model"]
    P_val = np.asarray(model.predict_proba(_ml_input(model_name, sp, "val")), dtype=np.float64)
    P_te = np.asarray(model.predict_proba(_ml_input(model_name, sp, "test")), dtype=np.float64)
    ll_va, ll_te = safe_log_loss(sp["y_val"], P_val), safe_log_loss(sp["y_test"], P_te)
    stage_print(f"✅ {model_name} : VAL {ll_va:.4f} | TEST {ll_te:.4f} | {pl.get('calib_status')}")
    save_partial(model_name, {
        "schema": _MODEL_SCHEMA, "model_key": model_name, "kind": "ml", "selection_id": sp["selection_id"],
        "val_ll": ll_va, "test_ll": ll_te, "wf_ll": ll_va, "calib_ll": ll_va,
        "P_val": P_val, "P_test": P_te, "calib_status": pl.get("calib_status")})


def _finalize_pb_one(stage_name: str, class_name: str) -> None:
    from finalize_split import save_partial
    from goangel_core import (pb_hda_batch, fit_guarded_calibrator, safe_log_loss,
                              prior_log_loss, normalize_probas)

    partial_path = Path(CACHE_DIR) / "stages" / "finalize" / f"{class_name}.pkl"
    if _partial_is_current(partial_path):
        stage_print(f"📦 {class_name} déjà finalisé (partial à jour)")
        return
    stage_print(f"🔧 FINALIZE PB — {class_name}")
    model_path = MODELS_DIR / f"pb_{class_name}.pkl"
    if not model_path.exists():
        save_partial(class_name, {"schema": _MODEL_SCHEMA, "model_key": class_name,
                                  "skipped": True, "reason": "model_missing"})
        return
    model = _atomic_read_pickle(model_path)["model"]
    sp = _build_splits()
    y_tr = sp["y_train"]
    freq = np.array([(y_tr == k).mean() for k in range(3)])

    def _predict(part: str):
        P, ok = pb_hda_batch(model, sp[f"hk_{part}"], sp[f"ak_{part}"])
        return P, ok

    P_ca, ok_ca = _predict("calib")
    P_va, ok_va = _predict("val")
    P_te, ok_te = _predict("test")
    stage_print(f"   couverture : calib {ok_ca.mean():.0%} | val {ok_va.mean():.0%} | test {ok_te.mean():.0%}")
    if ok_ca.sum() < 300 or ok_va.mean() < 0.80 or ok_te.mean() < 0.80:
        stage_print(f"⚠️  {class_name} : prédictions insuffisantes -> exclu de l'ensemble (pas d'échec dur)")
        save_partial(class_name, {"schema": _MODEL_SCHEMA, "model_key": class_name, "skipped": True,
                                  "reason": "insufficient_predictions"})
        return
    cal, status, info = fit_guarded_calibrator(
        P_ca[ok_ca], sp["y_calib"][ok_ca],
        min_isotonic_samples=int(_cfg("CALIBRATION_ISOTONIC_MIN_SAMPLES", 1000)),
        min_gain=float(_cfg("CALIBRATION_GUARD_MIN_GAIN", 5e-4)))

    def _finish(P, ok):
        P = P.copy()
        if cal is not None and ok.any():
            P[ok] = cal.transform(P[ok])
        P[~ok] = freq                      # équipe inconnue du train : on ne triche pas, prior
        return normalize_probas(P, 1e-6)

    P_va_c, P_te_c = _finish(P_va, ok_va), _finish(P_te, ok_te)
    ll_va_raw = safe_log_loss(sp["y_val"], _finish_raw(P_va, ok_va, freq))
    ll_va, ll_te = safe_log_loss(sp["y_val"], P_va_c), safe_log_loss(sp["y_test"], P_te_c)
    stage_print(f"✅ {class_name} [{status}] : VAL brut {ll_va_raw:.4f} → calibré {ll_va:.4f} | "
                f"TEST {ll_te:.4f} | constant={prior_log_loss(y_tr, sp['y_val']):.4f}")
    save_partial(class_name, {
        "schema": _MODEL_SCHEMA, "model_key": class_name, "kind": "pb", "calibrator": cal,
        "val_ll": ll_va, "test_ll": ll_te, "wf_ll": ll_va, "calib_ll": ll_va,
        "P_val": P_va_c, "P_test": P_te_c, "calib_status": status})
    save_pb_finalized(class_name, ll_va, ll_va, status, 0)


def _finish_raw(P, ok, freq):
    from goangel_core import normalize_probas
    P = P.copy()
    P[~ok] = freq
    return normalize_probas(P, 1e-6)


def _stage_train_stacking() -> None:
    """Ancien stacking CV : méta-modèle entraîné sur des probas IN-SAMPLE (modèles de base déjà
    entraînés sur ces mêmes lignes) => fuite + LogLoss optimiste. Remplacé par un blend convexe
    optimisé sur le set VAL (hors-échantillon des calibrateurs) dans finalize_agg."""
    stage_print("🧠 STACKING — remplacé par blend convexe sur VAL (finalize_agg), anti-fuite")
    path = MODELS_DIR / "ml_Stacking.pkl"
    _atomic_write_pickle(path, {"model": None, "type": "BlendInFinalizeAgg", "schema": _MODEL_SCHEMA})
    save_ml_trained("Stacking", str(path))
    save_ml_calibrated("Stacking", "blend")
    stage_print(f"💾 Marqueur sauvegardé : {path}")


def _stage_finalize_agg() -> None:
    """Gate 'zéro skill' -> poids convexes optimisés sur VAL -> évaluation sur TEST (jamais vu)."""
    from finalize_split import list_partials
    from goangel_core import (skill_gate, optimize_blend_weights, blend, safe_log_loss,
                              prior_log_loss, CalibratedPBModel)

    stage_print("🔧 FINALIZE AGG — agrégation finale")
    sp = _build_splits()
    y_tr, y_va, y_te = sp["y_train"], sp["y_val"], sp["y_test"]
    partials = list_partials()
    P_val: Dict[str, np.ndarray] = {}
    P_test: Dict[str, np.ndarray] = {}
    val_ll: Dict[str, float] = {}
    test_ll: Dict[str, float] = {}
    for key, pl in partials.items():
        if not isinstance(pl, dict) or pl.get("skipped") or "P_val" not in pl or "P_test" not in pl:
            continue
        if pl.get("schema") != _MODEL_SCHEMA:
            stage_print(f"   ⚠️ {key} : ancien schéma ignoré")
            continue
        P_val[key], P_test[key] = np.asarray(pl["P_val"]), np.asarray(pl["P_test"])
        val_ll[key], test_ll[key] = float(pl["val_ll"]), float(pl["test_ll"])
    stage_print(f"   {len(P_val)} modèles évalués")

    prior_va, prior_te = prior_log_loss(y_tr, y_va), prior_log_loss(y_tr, y_te)
    kept = skill_gate(val_ll, prior_va,
                      min_skill=float(_cfg("ENSEMBLE_MIN_SKILL", 0.005)),
                      max_gap_to_best=float(_cfg("ENSEMBLE_MAX_GAP_TO_BEST", 0.04)))
    stage_print(f"   {'modèle':<34}{'VAL':>8}{'TEST':>8}  statut")
    for k in sorted(val_ll, key=val_ll.get):
        stage_print(f"   {k:<34}{val_ll[k]:>8.4f}{test_ll[k]:>8.4f}  "
                    f"{'RETENU' if k in kept else 'exclu (pas de skill / trop loin du meilleur)'}")
    if not kept:
        raise RuntimeError("Aucun modèle ne bat le prédicteur constant sur VAL — vérifier les features")

    w = optimize_blend_weights({k: P_val[k] for k in kept}, y_va,
                               w_max=float(ENSEMBLE_MAX_WEIGHT),
                               ridge=float(_cfg("ENSEMBLE_BLEND_RIDGE", 2e-3)))
    w = {k: v for k, v in w.items() if v > 0}
    ens_va = safe_log_loss(y_va, blend({k: P_val[k] for k in w}, w))
    P_ens_te = blend({k: P_test[k] for k in w}, w)
    ens_te = safe_log_loss(y_te, P_ens_te)
    uni = {k: 1.0 / len(w) for k in w}
    uni_te = safe_log_loss(y_te, blend({k: P_test[k] for k in w}, uni))
    best_single = min(test_ll[k] for k in kept)
    stage_print("   Poids : " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(w.items(), key=lambda kv: -kv[1])))
    stage_print(f"📊 ENSEMBLE  VAL {ens_va:.4f} | TEST {ens_te:.4f} | acc {float((P_ens_te.argmax(1) == y_te).mean()) * 100:.1f}% "
                f"| ECE {_ece(y_te, P_ens_te):.4f}")
    stage_print(f"   repères TEST : constant {prior_te:.4f} | meilleur modèle seul {best_single:.4f} "
                f"| moyenne uniforme {uni_te:.4f} | hasard 1.0986")

    ml_models: Dict[str, Any] = {}
    pb_models: List[Tuple[str, Any]] = []
    for k in w:
        mp, pp = MODELS_DIR / f"ml_{k}.pkl", MODELS_DIR / f"pb_{k}.pkl"
        if mp.exists():
            ml_models[k] = _atomic_read_pickle(mp)["model"]
        elif pp.exists():
            pb_models.append((k, CalibratedPBModel(_atomic_read_pickle(pp)["model"],
                                                   partials[k].get("calibrator"))))
    weights: Dict[str, Any] = dict(w)
    weights.update({
        "__temperature__": 1.0, "__temperature_validated__": False,   # désactivé : pas de double calibration
        "__ll_source_map__": {k: "val_chronologique" for k in val_ll},
        "__calibration_status_map__": {k: partials[k].get("calib_status") for k in val_ll},
        "__walk_forward_ll_map__": dict(val_ll), "__calibration_ll_map__": dict(val_ll),
        "__test_ll_map__": dict(test_ll),
        "__feature_names__": list(FEATURE_NAMES), "__feature_count__": int(EXPECTED_FEATURE_COUNT),
        "__cache_version__": "v10", "__score_grid_size__": 12, "__data_source__": "bzzoiro",
        "__calibration_logloss__": float(ens_va),
        "__test_logloss__": float(ens_te),
        "__test_accuracy__": float((P_ens_te.argmax(1) == y_te).mean()),
        "__prior_logloss_test__": float(prior_te),
        "__feature_selection_id__": sp["selection_id"],
        "__n_selected_features__": int(sp["prep"].n_features_out_),
    })
    _atomic_write_pickle(Path(CACHE_DIR) / "models.pkl", (ml_models, pb_models))
    _atomic_write_pickle(Path(CACHE_DIR) / "scaler.pkl", sp["prep"])
    _atomic_write_pickle(Path(CACHE_DIR) / "weights.pkl", weights)
    stage_print(f"💾 Cache final sauvegardé dans {CACHE_DIR}")
    stage_print(f"   ML : {len(ml_models)} | PB : {len(pb_models)} | Poids : {len(w)}")


_FINALIZE_STAGE_TO_PARTIAL: Dict[str, str] = {
    "xgb": "XGB", "lgbm": "LGBM", "catboost": "CatBoost", "rf": "RF", "dl": "DL",
    "lr": "LR", "et": "ET", "svm": "SVM", "histgb": "HistGB", "mlp": "MLP",
    "tabm": "TabM", "realmlp": "RealMLP", "excelformer": "ExcelFormer",
    "ngboost": "NGBoost", "stacking": "Stacking", "hmm": "HMMModel"}


def _stage_finalize() -> None:
    payload = _stage_load_features()
    X = payload["X"]
    y = payload["y"]
    home_keys = payload["home_keys"]
    away_keys = payload["away_keys"]
    gh_all = payload["gh_all"]
    ga_all = payload["ga_all"]
    n = len(X)
    cut_train = int(n * TRAIN_RATIO)
    cut_val = int(n * (TRAIN_RATIO + VAL_RATIO))
    cut_calib = int(n * (TRAIN_RATIO + VAL_RATIO + CALIB_RATIO))
    cut_train = max(1, min(cut_train, n - 4))
    cut_val = max(cut_train + 1, min(cut_val, n - 3))
    cut_calib = max(cut_val + 1, min(cut_calib, n - 2))
    X_train_raw = np.where(np.isfinite(X[:cut_train]), X[:cut_train], np.nan)
    X_calib_raw = np.where(np.isfinite(X[cut_val:cut_calib]), X[cut_val:cut_calib], np.nan)
    X_test_raw = np.where(np.isfinite(X[cut_calib:]), X[cut_calib:], np.nan)
    y_train = y[:cut_train]
    y_calib = y[cut_val:cut_calib]
    y_test = y[cut_calib:]
    home_keys_calib = home_keys[cut_val:cut_calib]
    away_keys_calib = away_keys[cut_val:cut_calib]
    home_keys_test = home_keys[cut_calib:]
    away_keys_test = away_keys[cut_calib:]
    gh_train = gh_all[:cut_train]
    ga_train = ga_all[:cut_train]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_calib_imp = imputer.transform(X_calib_raw)
    X_calib_scaled = scaler.transform(X_calib_imp)
    X_test_imp = imputer.transform(X_test_raw)
    X_test_scaled = scaler.transform(X_test_imp)
    pipeline = Pipeline([("imputer", imputer), ("scaler", scaler)])

    ml_models: Dict[str, Any] = {}
    ml_ll_calib: Dict[str, float] = {}
    ml_ll_wf: Dict[str, float] = {}
    ml_status: Dict[str, str] = {}

    import pickle
    for model_name in ("XGB", "LGBM", "CatBoost", "RF", "DL"):
        model_path = MODELS_DIR / f"ml_{model_name}.pkl"
        if not model_path.exists():
            stage_print(f"⚠️ {model_name} absent, ignoré")
            continue
        payload_m = _atomic_read_pickle(model_path)
        base = payload_m["model"]
        if model_name == "DL":
            calibrated, st = _calibrate_or_keep(model_name, base, X_calib_scaled, y_calib)
            if not _model_produces_three_classes(calibrated, X_calib_scaled, model_name):
                calibrated = _TorchProbaWrapper(base)
                st = _CALIBRATION_STATUS_RAW_KEPT
            ml_models[model_name] = calibrated
            ml_ll_calib[model_name] = _compute_calibration_log_loss(calibrated, X_calib_scaled, y_calib, None, model_name)
            ml_status[model_name] = st
        else:
            if model_name in _TREE_MODELS_NAN_NATIVE:
                X_fit = X_train_raw
                X_calib_fit = X_calib_raw
            else:
                X_fit = X_train_scaled
                X_calib_fit = X_calib_scaled
            calibrated, st = _calibrate_or_keep(model_name, base, X_calib_fit, y_calib)
            if not _model_produces_three_classes(calibrated, X_calib_fit, model_name):
                calibrated = base
                st = _CALIBRATION_STATUS_RAW_KEPT
            ml_models[model_name] = calibrated
            ml_ll_calib[model_name] = _compute_calibration_log_loss(calibrated, X_calib_fit, y_calib, X_calib_fit, model_name)
            ml_status[model_name] = st
        if model_name == "XGB":
            factory = _make_xgb_factory()
        elif model_name == "LGBM":
            factory = _make_lgbm_factory()
        elif model_name == "CatBoost":
            factory = _make_catboost_factory()
        else:
            factory = _make_rf_factory()
        ll_wf, ev, at = _evaluate_ml_model_walk_forward(factory, X_train_raw, y_train, MODEL_WALK_FORWARD_FOLDS, model_name)
        ml_ll_wf[model_name] = float(ll_wf)
        stage_print(f"✅ {model_name} : CALIB {ml_ll_calib[model_name]:.4f} | WF {ll_wf:.4f} ({ev}/{at})")

    pb_models: List[Tuple[str, Any]] = []
    pb_ll_calib: Dict[str, float] = {}
    pb_ll_wf: Dict[str, float] = {}
    pb_status: Dict[str, str] = {}

    for class_name in _PB_MODEL_ORDER:
        model_path = MODELS_DIR / f"pb_{class_name}.pkl"
        if not model_path.exists():
            continue
        payload_pb = _atomic_read_pickle(model_path)
        model = payload_pb["model"]
        probas: List[List[float]] = []
        y_kept: List[int] = []
        for h, a, yv in zip(home_keys_calib, away_keys_calib, y_calib):
            try:
                grid = model.predict(h, a)
                markets = _extract_model_probabilities(grid)
                hda = markets.get("home_draw_away")
                if hda is None or len(hda) != 3:
                    continue
                arr = np.asarray(hda, dtype=np.float64)
                if arr.sum() <= 0:
                    continue
                probas.append((arr / arr.sum()).tolist())
                y_kept.append(int(yv))
            except Exception:
                continue
        if len(probas) < 30:
            stage_print(f"⚠️ {class_name} : trop peu de prédictions valides ({len(probas)})")
            continue
        probas_arr = np.array(probas, dtype=np.float64)
        y_arr = np.array(y_kept, dtype=np.int64)
        if not _proba_array_is_valid(probas_arr):
            continue
        ll_calib_pb = float(log_loss(y_arr, probas_arr, labels=[0, 1, 2]))
        ll_wf_pb, ev, at = _evaluate_pb_model_walk_forward(class_name, gh_train, ga_train, home_keys[:cut_train], away_keys[:cut_train], MODEL_WALK_FORWARD_FOLDS)
        key = f"pb_{len(pb_models)}"
        pb_models.append((class_name, model))
        pb_ll_calib[key] = ll_calib_pb
        pb_ll_wf[key] = float(ll_wf_pb) if np.isfinite(ll_wf_pb) else ll_calib_pb
        pb_status[key] = _CALIBRATION_STATUS_NOT_CALIBRATED
        stage_print(f"✅ {class_name} : CALIB {ll_calib_pb:.4f} | WF {pb_ll_wf[key]:.4f}")

    weights: Dict[str, float] = {}
    for k, ll in ml_ll_wf.items():
        weights[k] = (1.0 / (ll + 1e-6)) if np.isfinite(ll) and 0 < ll <= _MAX_ACCEPTABLE_LOGLOSS else 0.0
    for k, ll in pb_ll_wf.items():
        weights[k] = (1.0 / (ll + 1e-6)) if np.isfinite(ll) and 0 < ll <= _MAX_ACCEPTABLE_LOGLOSS else 0.0

    filtered_weights = _filter_valid_weights(weights)
    if not any(not k.startswith("__") for k in filtered_weights):
        fallback = _select_fallback_uniform_models(ml_models, pb_models, {**ml_ll_wf, **pb_ll_wf})
        if fallback:
            u = 1.0 / len(fallback)
            for k in fallback:
                filtered_weights[k] = u
    else:
        filtered_weights = _apply_quality_filter_on_weights(filtered_weights, {**ml_ll_wf, **pb_ll_wf})
    weights = _normalize_weights(filtered_weights, ENSEMBLE_MIN_WEIGHT, ENSEMBLE_MAX_WEIGHT)

    temperature_validated = False
    if TEMPERATURE_SCALING_ENABLED:
        try:
            calib_blend = _compute_ensemble_probas(ml_models, pb_models, X_calib_scaled, home_keys_calib, away_keys_calib, weights, X_calib_raw)
            if calib_blend is not None:
                temp, ok = _compute_optimal_temperature(calib_blend, y_calib)
                weights["__temperature__"] = float(temp)
                temperature_validated = bool(ok)
            else:
                weights["__temperature__"] = float(TEMPERATURE_SCALING_DEFAULT)
        except Exception:
            weights["__temperature__"] = float(TEMPERATURE_SCALING_DEFAULT)

    weights["__temperature_validated__"] = bool(temperature_validated)
    weights["__ll_source_map__"] = {k: _LL_SOURCE_WALK_FORWARD for k in list(ml_ll_wf) + list(pb_ll_wf)}
    weights["__calibration_status_map__"] = {**ml_status, **pb_status}
    weights["__walk_forward_ll_map__"] = {**ml_ll_wf, **pb_ll_wf}
    weights["__calibration_ll_map__"] = {**ml_ll_calib, **pb_ll_calib}
    weights["__feature_names__"] = list(FEATURE_NAMES)
    weights["__feature_count__"] = int(EXPECTED_FEATURE_COUNT)

    final_temperature = float(weights.get("__temperature__", 1.0))
    test_metrics = _evaluate_ensemble_on_set(ml_models, pb_models, X_test_scaled, y_test, home_keys_test, away_keys_test, weights, final_temperature, X_test_raw)
    if test_metrics:
        ll_test, acc_test = test_metrics
        stage_print(f"📊 TEST FINAL : LogLoss {ll_test:.4f} | Accuracy {acc_test * 100:.1f}%")
        weights["__test_logloss__"] = float(ll_test)
        weights["__test_accuracy__"] = float(acc_test)

    weights["__cache_version__"] = "v10"
    weights["__score_grid_size__"] = 12
    weights["__data_source__"] = "bzzoiro"

    hist_path = Path(CACHE_DIR) / "historical.pkl"
    if not hist_path.exists():
        _atomic_write_pickle(hist_path, pd.DataFrame())
    try:
        from pb_convergence import _strip_mcmc_trace
        for _mn, _mm in (ml_models or {}).items():
            _strip_mcmc_trace(_mm)
        for _cn, _mm in (pb_models or []):
            _strip_mcmc_trace(_mm)
        logger.info("Strip trace appliqué avant models.pkl")
    except Exception as _strip_exc:
        logger.warning("Strip trace models.pkl : %s", _strip_exc)

    _atomic_write_pickle(Path(CACHE_DIR) / "models.pkl", (ml_models, pb_models))
    _atomic_write_pickle(Path(CACHE_DIR) / "scaler.pkl", pipeline)
    _atomic_write_pickle(Path(CACHE_DIR) / "weights.pkl", weights)
    stage_print(f"💾 Cache final sauvegardé dans {CACHE_DIR}")
    stage_print(f"   ML : {len(ml_models)} | PB : {len(pb_models)} | Poids : {len(weights)}")

def _stage_refresh() -> None:
    """Met à jour l'historique + régénère les features.

    Étapes :
    1. update_historical.py → récupère nouveaux matchs FINISHED (fetch API)
    2. create_features_with_context → régénère les 172 features pour
       TOUS les matchs (anciens + nouveaux) → features.pkl à jour
    """
    stage_print("🔄 STAGE REFRESH — Update historique + features")

    # ── 1. Fetch incrémental des nouveaux matchs ──
    try:
        import subprocess
        import sys as _sys
        result = subprocess.run(
            [_sys.executable, "update_historical.py"],
            capture_output=False,
            text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"update_historical.py a échoué (RC={result.returncode})"
            )
        stage_print("✅ Historique mis à jour")
    except FileNotFoundError:
        stage_print("⚠️  update_historical.py introuvable — fetch ignoré")
    except Exception as exc:
        logger.error("Fetch refresh échoué : %s", exc)
        raise

    # ── 2. Régénération des features ──
    stage_print("🔧 Régénération des features (tracker sur TOUS les matchs)...")

    try:
        import pickle as _pickle
        from features import (
            create_features_with_context,
            EXPECTED_FEATURE_COUNT)

        hist_path = Path(CACHE_DIR) / "historical.pkl"
        if not hist_path.exists():
            raise RuntimeError(f"historical.pkl introuvable : {hist_path}")

        # Charger historical.pkl (gère xz + pickle brut)
        try:
            from predict import _atomic_read_pickle as _arp
            df_past = _arp(hist_path)
        except Exception:
            with open(hist_path, "rb") as f:
                df_past = _pickle.load(f)

        if df_past is None or df_past.empty:
            raise RuntimeError("historical.pkl vide après refresh")

        stage_print(f"   historical.pkl : {len(df_past)} matchs")

        # Régénérer les features
        t0 = _time.time()
        X, y, home_keys, away_keys, gh_all, ga_all, comps, h2h = (
            create_features_with_context(df_past, strict_competition=True)
        )
        duration = _time.time() - t0

        if X.shape[0] < MIN_MATCHES_FOR_TRAINING:
            raise RuntimeError(
                f"Features insuffisantes : {X.shape[0]} < {MIN_MATCHES_FOR_TRAINING}"
            )
        if X.shape[1] != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                f"Features shape invalide : {X.shape[1]} != {EXPECTED_FEATURE_COUNT}"
            )

        # Sauvegarder features.pkl (atomique)
        payload = {
            "X": X,
            "y": y,
            "home_keys": home_keys,
            "away_keys": away_keys,
            "gh_all": gh_all,
            "ga_all": ga_all,
            "comps": comps,
            "h2h": h2h,
            "dates": df_past["date_parsed"].tolist() if "date_parsed" in df_past.columns else [],
            "n": len(X),
            "created_at": _datetime.now().isoformat()}
        _atomic_write_pickle(_FEATURES_FILE, payload)

        # Métadonnées
        meta = {
            "n_matchs": len(df_past),
            "n_features": X.shape[1],
            "n_samples": len(X),
            "class_counts": np.bincount(y, minlength=3).tolist(),
            "created_at": _datetime.now().isoformat(),
            "refreshed": True}
        _atomic_write_pickle(_METADATA_FILE, meta)

        stage_print(
            f"✅ Features régénérées : {len(X)} samples × {X.shape[1]} features "
            f"en {duration:.1f}s"
        )
        stage_print(f"   Classes : {meta['class_counts']}")
        _compute_and_save_selection()
        stage_print("✅ STAGE REFRESH — Historique + features + sélection à jour")
    except Exception as exc:
        logger.error("Régénération features échouée : %s", exc)
        raise

import time as _time
from datetime import datetime as _datetime

def _stage_prepare() -> None:
    stage_print("🚀 STAGE PREPARE")
    try:
        ckpt_mgr = CheckpointManager(
            checkpoint_dir=Path(CHECKPOINT_DIR),
            save_interval=30.0,
            push_interval=300.0,
            github_enabled=os.getenv("GOANGEL_GITHUB_PUSH", "false").lower() in ("1", "true", "yes", "on"),
            github_repo=os.getenv("GOANGEL_GITHUB_REPO", "Angeloda444/GOANGEL2"),
            github_user=os.getenv("GOANGEL_GITHUB_USER", "Angeloda444"),
            github_token=os.getenv("GH_PAT", ""),
            github_branch=os.getenv("GOANGEL_GITHUB_BRANCH", "cache-auto"))
    except Exception as exc:
        logger.warning("CheckpointManager init échoué : %s", exc)
        ckpt_mgr = None
    df_past = _load_full_history(checkpoint_mgr=ckpt_mgr)
    if df_past is None or df_past.empty:
        raise RuntimeError("Historique vide après récupération")
    _atomic_write_pickle(Path(CACHE_DIR) / "historical.pkl", df_past)
    _stage_save_features(df_past)
    _compute_and_save_selection()
    if ckpt_mgr is not None:
        try:
            ckpt_mgr.finalize()
        except Exception:
            pass
    stage_print(f"✅ STAGE PREPARE terminé : {len(df_past)} matchs")

def _stage_run(stage_name: str) -> int:
    if stage_name == "next":
        nxt = _next_stage()
        print(f"NEXT:{nxt}")
        return 0

    if stage_name not in _STAGES_ORDER:
        print(f"{RED}Stage inconnu : {stage_name}{RESET}")
        return 2

    if _is_stage_done(stage_name):
        stage_print(f"⏭️  {stage_name} déjà done (skip)")
        return 0

    attempts = _read_attempts(stage_name)

    # ─── Vérification préalable : environnement valide ? ───
    # Si les fichiers critiques sont des pointeurs LFS ou absents,
    # on ne consomme PAS d'attempt.
    _env_ok = True
    _env_reasons = []
    for _f in (Path(CACHE_DIR) / "historical.pkl",
               Path(CACHE_DIR) / "stages" / "features.pkl"):
        if not _f.exists():
            _env_ok = False
            _env_reasons.append(f"absent: {_f.name}")
            continue
        try:
            with open(_f, "rb") as _fh:
                _magic = _fh.read(6)
            if _magic.startswith(b"version https://git-lfs"):
                _env_ok = False
                _env_reasons.append(f"pointeur LFS: {_f.name}")
        except OSError as _exc:
            _env_ok = False
            _env_reasons.append(f"lecture impossible: {_f.name}")

    if not _env_ok:
        print(f"{RED}❌ Environnement invalide — pas de tentative consommée{RESET}")
        for _r in _env_reasons:
            print(f"   - {_r}")
        return 1

    current_attempt = _increment_attempts(stage_name)
    stage_print(f"{CYAN}▶️  STAGE : {stage_name} (attempt {current_attempt}/{_MAX_ATTEMPTS_DEFAULT}){RESET}")

    if current_attempt > _MAX_ATTEMPTS_DEFAULT:
        print(f"{RED}❌ Stage {stage_name} : tentatives épuisées ({current_attempt}/{_MAX_ATTEMPTS_DEFAULT}){RESET}")
        return 3

    t0 = time.time()
    try:
        if stage_name == "prepare":
            _stage_prepare()
        elif stage_name == "refresh":
            _stage_refresh()
        elif stage_name in _ML_STAGE_KEYS:
            _stage_train_ml_model(_ML_STAGE_KEYS[stage_name])
        elif stage_name == "train_hmm":
            _stage_train_pb_model("HMMModel")
        elif stage_name in _PB_STAGE_KEYS:
            _stage_train_pb_model(_PB_STAGE_KEYS[stage_name])
        elif stage_name == "train_stacking":
            _stage_train_stacking()
        elif stage_name == "finalize_stacking":
            _finalize_ml_one("Stacking")
        elif stage_name == "train_stacking":
            _stage_train_stacking()
        elif stage_name == "finalize_stacking":
            _finalize_ml_one("Stacking")
        elif stage_name == "finalize":
            _stage_finalize()
        elif stage_name == "finalize_prep":
            _stage_finalize_prep()
        elif stage_name == "finalize_agg":
            _stage_finalize_agg()
        elif stage_name in _FINALIZE_PB_MODELS:
            _finalize_pb_one(stage_name, _FINALIZE_PB_MODELS[stage_name])
        elif stage_name.startswith("finalize_"):
            model_name = stage_name[len("finalize_"):]
            if model_name == "xgb":
                _finalize_ml_one("XGB")
            elif model_name == "lgbm":
                _finalize_ml_one("LGBM")
            elif model_name == "catboost":
                _finalize_ml_one("CatBoost")
            elif model_name == "rf":
                _finalize_ml_one("RF")
            elif model_name == "dl": _finalize_ml_one("DL")
            elif model_name == "histgb": _finalize_ml_one("HistGB")
            elif model_name == "mlp": _finalize_ml_one("MLP")
            elif model_name == "tabm": _finalize_ml_one("TabM")
            elif model_name == "realmlp": _finalize_ml_one("RealMLP")
            elif model_name == "excelformer": _finalize_ml_one("ExcelFormer")
            elif model_name == "ngboost": _finalize_ml_one("NGBoost")
            elif model_name == "hmm": _finalize_pb_one("finalize_hmm", "HMMModel")
            elif model_name == "lr": _finalize_ml_one("LR")
            elif model_name == "et": _finalize_ml_one("ET")
            elif model_name == "svm": _finalize_ml_one("SVM")
            else:
                print(f"{RED}Stage inconnu : {stage_name}{RESET}")
                return 2
        else:
            print(f"{RED}Stage inconnu : {stage_name}{RESET}")
            return 2
    except PBIncompleteError as exc:
        logger.warning("Stage %s incomplet : %s", stage_name, exc)
        print(
            f"{YELLOW}⏸️ Stage {stage_name} incomplet "
            f"(retry next attempt) : {exc}{RESET}"
        )
        return 42
    except Exception as exc:
        logger.exception("Stage %s échoué", stage_name)
        print(f"{RED}❌ Stage {stage_name} échoué : {exc}{RESET}")
        return 1

    ok = _mark_stage_done(stage_name) or _is_stage_done(stage_name)
    if not ok:
        print(f"{RED}❌ Stage {stage_name} : marquage done échoué{RESET}")
        return 4

    duration = time.time() - t0
    stage_print(f"{GREEN}✅ STAGE {stage_name} terminé en {duration:.1f}s (attempt {current_attempt}){RESET}")
    return 0

def run_interactive(state: Dict[str, Any]) -> int:
    stage_print(
        f"{CYAN}🤖 GOANGEL PRÊT — Tapez un jour (ex: {YELLOW}07{RESET}{CYAN}) "
        f"ou une date complète ({YELLOW}07/12/2026{RESET}{CYAN}). "
        f"Tapez {YELLOW}help{RESET}{CYAN} pour l'aide.{RESET}"
    )
    while True:
        try:
            user_input = input(f"\n{GOLD}📅 Date ou commande ({YELLOW}quit{RESET}{GOLD} pour quitter) : {RESET}")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{GREY}Au revoir.{RESET}")
            return 0
        if _SHUTDOWN_FLAG["requested"]:
            print(f"{YELLOW}Arrêt demandé.{RESET}")
            return 0
        if user_input is None:
            continue
        user_input = user_input.strip()
        if not user_input:
            continue
        ui = user_input.lower()
        if ui in _QUIT_COMMANDS:
            print(f"{GREY}Fermeture de GOANGEL.{RESET}")
            return 0
        if ui in _CLEAR_COMMANDS:
            clear_screen()
            continue
        if ui in _HELP_COMMANDS:
            display_help()
            continue
        if ui in _COMPETITIONS_COMMANDS:
            display_competitions()
            continue
        if ui in _STATUS_COMMANDS:
            display_status(state)
            continue
        if ui in _REFRESH_COMMANDS:
            print(f"{CYAN}♻️  Reconstruction complète...{RESET}")
            ok = _safe_reload_models(state, force_refresh=True)
            if ok:
                print(f"{GREEN}✅ Reconstruction terminée.{RESET}")
            continue
        date_iso, mode, reason = parse_date_smart(user_input)
        if date_iso is None:
            print(f"{RED}❌ Date non reconnue : {reason}.{RESET}")
            continue
        pretty = _pretty_iso(date_iso)
        mode_color = {
            "auto": GOLD, "day_month": SKY, "full": LIME,
            "today": PINK, "yesterday": PINK, "tomorrow": PINK}.get(mode, GREY)
        print(f"\n{CYAN}🔍 Prédictions pour le {BOLD}{pretty}{RESET} {GREY}({mode_color}{reason}{GREY}){RESET}")
        _predict_date(state, date_iso)
        print()
        print(f"{GREY}" + "═" * _SEPARATOR_LEN + f"{RESET}")

def run_single_date(date_input: str, state: Dict[str, Any]) -> int:
    date_iso, mode, reason = parse_date_smart(date_input)
    if date_iso is None:
        print(f"{RED}❌ Date non reconnue : {reason}.{RESET}")
        return 2
    pretty = _pretty_iso(date_iso)
    print(f"{CYAN}🔍 Prédictions pour le {BOLD}{pretty}{RESET} {GREY}({reason}){RESET}")
    ok = _predict_date(state, date_iso)
    return 0 if ok else 1

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="goangel",
        description="GOANGEL — Moteur de prédictions sportives multi-modèles (Bzzoiro)")
    parser.add_argument("--date", type=str, default=None, help="Date : '07', '07/12', '07/12/2026', today/tomorrow/yesterday")
    parser.add_argument("--stage", type=str, default=None, choices=list(_VALID_STAGES), help="Stage de traitement ou 'next' pour le prochain")
    parser.add_argument("--reset-progress", action="store_true", help="Réinitialise tout le progrès des stages")
    parser.add_argument("--refresh", action="store_true", help="Force la reconstruction complète du cache et des modèles")
    parser.add_argument("--status", action="store_true", help="Affiche l'état du cache et des modèles puis quitte")
    parser.add_argument("--no-banner", action="store_true", help="N'affiche pas la bannière")
    parser.add_argument("--quiet", action="store_true", help="Réduit la sortie console au strict minimum")
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    _install_signal_handlers()
    args = _parse_args(argv)

    if args.reset_progress:
        _reset_all_progress()
        print(f"{GREEN}✅ Progress réinitialisé{RESET}")
        return 0

    if args.stage:
        return _stage_run(args.stage)

    if not args.no_banner and not args.quiet:
        print_banner()
    if not args.quiet:
        stage_print(f"{CYAN}🚀 DÉMARRAGE INITIAL{RESET}")
    logger.info("Démarrage de GOANGEL (argv=%s)", argv if argv is not None else sys.argv[1:])

    state: Dict[str, Any] = {
        "ml_models": None,
        "pb_models": None,
        "pipeline": None,
        "df_past": None,
        "weights": {}}

    if not _safe_reload_models(state, force_refresh=args.refresh):
        logger.critical("Arrêt : impossible de charger les modèles.")
        if not args.quiet:
            stage_print(f"{RED}❌ Erreur fatale lors du chargement des modèles.{RESET}")
        return 1

    if args.status:
        display_status(state)
        return 0
    if args.date:
        return run_single_date(args.date, state)
    try:
        return run_interactive(state)
    except KeyboardInterrupt:
        print(f"\n{GREY}Au revoir.{RESET}")
        return 130
    except Exception as exc:
        logger.critical("Erreur non gérée : %s", exc)
        logger.debug(traceback.format_exc())
        print(f"{RED}Erreur inattendue : {exc}{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
