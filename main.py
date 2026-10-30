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

from config import COMPETITIONS_DETAILS, COMPETITIONS, CACHE_DIR, CHECKPOINT_DIR, CACHE_NEVER_EXPIRES, MIN_MATCHES_FOR_TRAINING, MANIFEST_FILE, CONFIG_FINGERPRINT, ARTIFACT_SCHEMA_VERSION, EXPECTED_FEATURE_COUNT_REFERENCE, HISTORICAL_SEASONS_LOOKBACK, BZZOIRO_ENRICH_XG, BZZOIRO_WORKERS_PER_KEY, BZZOIRO_MAX_WORKERS, TOKEN_COUNT, TRAIN_RATIO, VAL_RATIO, CALIB_RATIO, ENSEMBLE_MAX_WEIGHT, MODEL_RANDOM_SEED, DL_ENABLED, DL_LEARNING_RATE, DL_WEIGHT_DECAY, DL_BATCH_SIZE, DL_EPOCHS, DL_EARLY_STOPPING_PATIENCE, DL_GRAD_CLIP, FEATURE_OUTLIER_ZSCORE, now_local, stage_print

from features import FEATURE_NAMES, EXPECTED_FEATURE_COUNT, _TREE_MODELS_NAN_NATIVE, _PB_MODEL_CLASSES, _TorchNet, _TorchProbaWrapper, _make_xgb_factory, _make_lgbm_factory, _make_catboost_factory, _make_rf_factory, create_features_with_context

from features import (
    CopulaModel,
    DixonColesDynamicModel,
    KalmanBayesianModel,
    MonteCarloSimulator,
    SkellamModel,
    _make_et_factory,
    _make_lr_factory,
    _make_ngboost_factory,
    _make_realmlp_factory,
    _make_svm_factory,
    _make_tabm_factory)
from registry import (
    BAYESIAN_PB_MODELS,
    FINALIZE_ML_KEYS,
    FINALIZE_STAT_KEYS,
    ML_STAGE_KEYS as _ML_STAGE_KEYS,
    STAGES_ORDER as _STAGES_ORDER,
    STAT_STAGE_KEYS as _PB_STAGE_KEYS)
from stat_models import (
    BayesianDynamicStrengthModel,
    EloRatingModel,
    NegativeBinomialRegressionModel,
    OrdinalRegressionModel)

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

from api import CheckpointManager

from pb_convergence import PBIncompleteError
from checkpoint_all import has_step as ck_has_step, save_finalize_context, save_ml_trained, save_ml_calibrated, save_pb_finalized, ml_key, STEP_TRAINED

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
        ("refresh", "Recharger le cache des modèles"),
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

_LEAGUES_SIG: str = hashlib.sha1(",".join(sorted(COMPETITIONS)).encode()).hexdigest()[:12]


def _filter_leagues(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "competition_code" not in df.columns:
        return df
    kept = df[df["competition_code"].isin(set(COMPETITIONS))].reset_index(drop=True)
    if len(kept) < len(df):
        logger.info("Filtre TOP_30_LEAGUES : %d -> %d matchs", len(df), len(kept))
    return kept


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
    finalize_dir = Path(CACHE_DIR) / "stages" / "finalize"
    if stage == "train_stacking":
        return [MODELS_DIR / "ml_Stacking.pkl"]
    if stage == "prepare":
        return [_FEATURES_FILE, _METADATA_FILE, Path(CACHE_DIR) / "historical.pkl"]
    if stage == "refresh":
        return [Path(CACHE_DIR) / "historical.pkl"]
    if stage in _ML_STAGE_KEYS:
        return [MODELS_DIR / f"ml_{_ML_STAGE_KEYS[stage]}.pkl"]
    if stage in _PB_STAGE_KEYS:
        return [MODELS_DIR / f"pb_{_PB_STAGE_KEYS[stage]}.pkl"]
    if stage == "finalize_prep":
        return [finalize_dir / "context.pkl"]
    if stage == "finalize_agg":
        return [Path(CACHE_DIR) / name for name in ("models.pkl", "scaler.pkl", "weights.pkl")]
    if stage in FINALIZE_STAT_KEYS:
        return [finalize_dir / f"{FINALIZE_STAT_KEYS[stage]}.pkl"]
    if stage in FINALIZE_ML_KEYS:
        return [finalize_dir / f"{FINALIZE_ML_KEYS[stage]}.pkl"]
    return []


def _is_stage_done(stage: str) -> bool:
    done_path = _stage_done_path(stage)
    data = _read_json(done_path)
    if not isinstance(data, dict):
        return False
    if data.get("stage") != stage:
        return False
    if data.get("leagues_sig") != _LEAGUES_SIG:
        return False
    if stage != "prepare" and data.get("feature_schema") != int(EXPECTED_FEATURE_COUNT):
        return False
    if stage not in ("prepare", "refresh") and data.get("selection_id") != _resolve_selection()["selection_id"]:
        return False
    return True

def _mark_stage_done(stage: str) -> bool:
    artifacts = _stage_artifacts(stage)
    if not artifacts:
        return False
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
        "schema": ARTIFACT_SCHEMA_VERSION,
        "feature_schema": int(EXPECTED_FEATURE_COUNT),
        "leagues_sig": _LEAGUES_SIG,
        "selection_id": _resolve_selection()["selection_id"]}
    _atomic_write_json(_stage_done_path(stage), payload)
    logger.info("Stage %s marqué done (%d fichiers)", stage, len(hashes))
    return True

def _purge_stale_markers() -> int:
    n = 0
    for stage in _STAGES_ORDER:
        p = _stage_done_path(stage)
        if p.exists() and not _is_stage_done(stage):
            for q in (p, _stage_attempts_path(stage)):
                try:
                    q.unlink()
                except OSError:
                    pass
            n += 1
    return n


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

def _safe_reload_models(state: Dict[str, Any]) -> bool:
    try:
        from predict import load_data_and_models
        result = load_data_and_models()
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

_CUSTOM_STAT_MODELS = {
    "SkellamModel": SkellamModel,
    "CopulaModel": CopulaModel,
    "DixonColesDynamicModel": DixonColesDynamicModel,
    "KalmanBayesianModel": KalmanBayesianModel,
    "EloRatingModel": EloRatingModel,
    "BayesianDynamicStrengthModel": BayesianDynamicStrengthModel,
}
_BAYESIAN_MAX_INCOMPLETE_ATTEMPTS = 3


def _write_stat_model(class_name: str, model: Any, meta: Optional[Dict[str, Any]] = None, skipped: bool = False) -> None:
    path = MODELS_DIR / f"pb_{class_name}.pkl"
    if path.exists():
        path.unlink()
    _atomic_write_pickle(path, {"model": model, "type": class_name, "meta": meta or {}, "skipped": skipped})
    save_ml_trained(class_name, str(path))
    save_ml_calibrated(class_name, "skipped" if skipped else "built_in")
    stage_print(f"💾 {class_name} {'ignoré' if skipped else 'sauvegardé'} : {path}")


def _stage_train_pb_model(class_name: str) -> None:
    from pb_convergence import fit_pb_attempt, get_pb_attempt

    payload = _stage_load_features()
    n = len(payload["X"])
    cut_train = _split_cuts(n)[0]
    gh_train = payload["gh_all"][:cut_train]
    ga_train = payload["ga_all"][:cut_train]
    th_train = payload["home_keys"][:cut_train]
    ta_train = payload["away_keys"][:cut_train]
    dates_all = payload.get("dates")
    dates_train = list(dates_all[:cut_train]) if dates_all is not None and len(dates_all) == n else None

    if class_name == "MonteCarloSimulator":
        model = MonteCarloSimulator(intensity_home=float(np.mean(gh_train)), intensity_away=float(np.mean(ga_train)))
        model.fit()
        _write_stat_model(class_name, model)
        return
    if class_name in _CUSTOM_STAT_MODELS:
        model = _CUSTOM_STAT_MODELS[class_name]()
        model.fit(gh_train, ga_train, th_train, ta_train, dates_train)
        _write_stat_model(class_name, model)
        return

    model_cls = _PB_MODEL_CLASSES.get(class_name)
    is_bayesian = class_name in BAYESIAN_PB_MODELS
    if model_cls is None:
        if is_bayesian:
            stage_print(f"⚠️ {class_name} indisponible dans penaltyblog — ignoré")
            _write_stat_model(class_name, None, skipped=True)
            return
        raise RuntimeError(f"Modèle penaltyblog inconnu : {class_name}")

    stage_print(f"🧠 Tentative {class_name} (bayesian={is_bayesian}) sur {len(gh_train)} matchs...")
    status, model, meta = fit_pb_attempt(
        class_name=class_name, model_class=model_cls, gh=gh_train, ga=ga_train,
        th=th_train, ta=ta_train, dates=dates_train, is_bayesian=is_bayesian)
    stage_print(f"   → attempt={meta.get('attempt')} filter={meta.get('filter')} n_kept={meta.get('n_kept')} status={status}")
    if status == "ok" and model is not None:
        _write_stat_model(class_name, model, meta)
        return
    if status == "incomplete":
        if is_bayesian and get_pb_attempt(class_name) >= _BAYESIAN_MAX_INCOMPLETE_ATTEMPTS:
            _write_stat_model(class_name, None, meta, skipped=True)
            return
        raise PBIncompleteError(f"{class_name} : non convergé ({meta.get('error', 'unknown')})")
    raise RuntimeError(f"Erreur réelle sur {class_name} : {meta.get('error', 'unknown')}")


_MODEL_SCHEMA: str = "goangel-core-v5"


def _cfg(name: str, default: Any) -> Any:
    import config as _c
    return getattr(_c, name, default)


def _split_cuts(n: int) -> Tuple[int, int, int]:
    c_tr = max(1, min(int(n * TRAIN_RATIO), n - 4))
    c_va = max(c_tr + 1, min(int(n * (TRAIN_RATIO + VAL_RATIO)), n - 3))
    c_ca = max(c_va + 1, min(int(n * (TRAIN_RATIO + VAL_RATIO + CALIB_RATIO)), n - 2))
    return c_tr, c_va, c_ca


def _resolve_selection() -> Dict[str, Any]:
    from config import FEATURES_SELECTED
    from features import FEATURE_NAMES_FULL

    index = {name: i for i, name in enumerate(FEATURE_NAMES_FULL)}
    missing = [name for name in FEATURES_SELECTED if name not in index]
    if missing:
        raise RuntimeError(f"Features inconnues dans FEATURES_SELECTED : {missing}")
    if len(set(FEATURES_SELECTED)) != len(FEATURES_SELECTED):
        raise RuntimeError("FEATURES_SELECTED contient des doublons")
    idx = sorted(index[name] for name in FEATURES_SELECTED)
    names = [FEATURE_NAMES_FULL[i] for i in idx]
    return {
        "selected_idx": idx,
        "selected_names": names,
        "selection_id": hashlib.sha1("|".join(names).encode()).hexdigest()[:12],
        "n_features": len(FEATURE_NAMES_FULL),
    }


def _ece(y: np.ndarray, P: np.ndarray, bins: int = 10) -> float:
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
    from features import _compute_diff_features, EXPECTED_FEATURE_COUNT_FULL
    from goangel_core import GoPreprocessor

    payload = _stage_load_features()
    X = _compute_diff_features(payload["X"])
    if X.shape[1] != EXPECTED_FEATURE_COUNT_FULL:
        raise RuntimeError(f"Dimension features {X.shape[1]} != {EXPECTED_FEATURE_COUNT_FULL}")
    y = np.asarray(payload["y"])
    hk, ak = payload["home_keys"], payload["away_keys"]
    n = len(X)
    c_tr = max(1, min(int(n * TRAIN_RATIO), n - 4))
    c_va = max(c_tr + 1, min(int(n * (TRAIN_RATIO + VAL_RATIO)), n - 3))
    c_ca = max(c_va + 1, min(int(n * (TRAIN_RATIO + VAL_RATIO + CALIB_RATIO)), n - 2))
    X_raw = np.where(np.isfinite(X), X, np.nan).astype(np.float64)
    sel = _resolve_selection()
    if sel["n_features"] != X.shape[1]:
        raise RuntimeError("Dimension des features incohérente avec features.pkl — relancer refresh")
    prep = GoPreprocessor(float(FEATURE_OUTLIER_ZSCORE), selected_idx=sel["selected_idx"]).fit(X_raw[:c_tr])
    Xs = prep.transform(X_raw)
    Xr = prep.transform_raw(X_raw)
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
    Xv, yv = torch.FloatTensor(Xva), torch.LongTensor(yva)
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
                    and _old.get("selection_id") == _resolve_selection()["selection_id"]):
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
    _es = int(_cfg("EARLY_STOP_ROUNDS", 200))

    if model_name == "MLP":
        base = _train_torch_classifier(sp, seed=MODEL_RANDOM_SEED, hidden=int(_cfg("DL_HIDDEN_SIZE", 128)))
    elif model_name == "Ordinal":
        base = OrdinalRegressionModel(random_state=MODEL_RANDOM_SEED).fit(X_fit, y_tr, sample_weight=sample_w)
    elif model_name == "NegBinReg":
        base = NegativeBinomialRegressionModel().fit(X_fit, y_tr, sp["gh_train"], sp["ga_train"])
    elif model_name == "HistGB":
        from sklearn.ensemble import HistGradientBoostingClassifier as _HGB
        from config import (HIST_N_ESTIMATORS, HIST_LEARNING_RATE,
                            HIST_MAX_LEAF_NODES, HIST_L2_REGULARIZATION)

        def _mk(it: int):
            return _HGB(max_iter=it, learning_rate=HIST_LEARNING_RATE,
                        max_leaf_nodes=HIST_MAX_LEAF_NODES, l2_regularization=HIST_L2_REGULARIZATION,
                        max_bins=255, early_stopping=False, random_state=MODEL_RANDOM_SEED)
        probe = _mk(HIST_N_ESTIMATORS).fit(X_fit, y_tr, sample_weight=sample_w)
        lls = [safe_log_loss(y_va, P) for P in probe.staged_predict_proba(X_va)]
        best_it = int(np.argmin(lls)) + 1
        stage_print(f"   HistGB : meilleure itération sur VAL = {best_it}/{HIST_N_ESTIMATORS}")
        base = _mk(best_it).fit(X_fit, y_tr, sample_weight=sample_w)
    else:
        factories = {
            "XGB": lambda: _make_xgb_factory()(), "LGBM": lambda: _make_lgbm_factory()(),
            "CatBoost": lambda: _make_catboost_factory()(), "RF": lambda: _make_rf_factory()(),
            "LR": lambda: _make_lr_factory()(), "ET": lambda: _make_et_factory()(),
            "SVM": lambda: _make_svm_factory()(), "TabM": lambda: _make_tabm_factory()(),
            "RealMLP": lambda: _make_realmlp_factory()(),
            "NGBoost": lambda: _make_ngboost_factory()(),
        }
        if model_name not in factories:
            raise ValueError(f"Modèle ML inconnu : {model_name}")
        try:
            base = factories[model_name]()
        except (RuntimeError, ImportError, TypeError) as fac_exc:
            base = None
            stage_print(f"⚠️ {model_name} indisponible : {fac_exc}")
        if base is None:
            stage_print(f"⚠️ {model_name} ignoré (librairie absente) — exclu de l'ensemble")
            _atomic_write_pickle(model_path, {"model": None, "type": model_name, "schema": _MODEL_SCHEMA,
                                              "selection_id": sp["selection_id"], "skipped": True})
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
        if pl.get("kind") == "ml" and pl.get("selection_id") != _resolve_selection()["selection_id"]:
            return False
        return True
    except Exception:
        return False


def _stage_finalize_prep() -> None:
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
    if pl.get("skipped"):
        save_partial(model_name, {"schema": _MODEL_SCHEMA, "model_key": model_name, "skipped": True,
                                  "reason": "library_unavailable"})
        return
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


def _finalize_pb_one(class_name: str) -> None:
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
    stored = _atomic_read_pickle(model_path)
    model = stored.get("model")
    if stored.get("skipped") or model is None:
        save_partial(class_name, {"schema": _MODEL_SCHEMA, "model_key": class_name, "skipped": True,
                                  "reason": "model_unavailable"})
        return
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
        P[~ok] = freq
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
    stage_print("🧠 STACKING — remplacé par blend convexe sur VAL (finalize_agg), anti-fuite")
    path = MODELS_DIR / "ml_Stacking.pkl"
    _atomic_write_pickle(path, {"model": None, "type": "BlendInFinalizeAgg", "schema": _MODEL_SCHEMA})
    save_ml_trained("Stacking", str(path))
    save_ml_calibrated("Stacking", "blend")
    stage_print(f"💾 Marqueur sauvegardé : {path}")


def _stage_finalize_agg() -> None:
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
        "__temperature__": 1.0, "__temperature_validated__": False,
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


def _stage_refresh() -> None:
    stage_print("🔄 STAGE REFRESH — Update historique + features")

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

    stage_print("🔧 Régénération des features (tracker sur TOUS les matchs)...")

    try:
        import pickle as _pickle
        from features import (
            create_features_with_context,
            EXPECTED_FEATURE_COUNT)

        hist_path = Path(CACHE_DIR) / "historical.pkl"
        if not hist_path.exists():
            raise RuntimeError(f"historical.pkl introuvable : {hist_path}")

        try:
            from predict import _atomic_read_pickle as _arp
            df_past = _arp(hist_path)
        except Exception:
            with open(hist_path, "rb") as f:
                df_past = _pickle.load(f)

        df_past = _filter_leagues(df_past)
        if df_past is None or df_past.empty:
            raise RuntimeError("historical.pkl vide après refresh")
        _atomic_write_pickle(hist_path, df_past)

        stage_print(f"   historical.pkl : {len(df_past)} matchs")

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
        stage_print("✅ STAGE REFRESH — Historique + features à jour")
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
    df_past = _filter_leagues(df_past)
    _atomic_write_pickle(Path(CACHE_DIR) / "historical.pkl", df_past)
    _stage_save_features(df_past)
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

    _env_ok = True
    _env_reasons = []
    _check_files = [] if stage_name in ("prepare", "refresh") else [
        Path(CACHE_DIR) / "historical.pkl",
        Path(CACHE_DIR) / "stages" / "features.pkl"
    ]
    for _f in _check_files:
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
        elif stage_name in _PB_STAGE_KEYS:
            _stage_train_pb_model(_PB_STAGE_KEYS[stage_name])
        elif stage_name == "train_stacking":
            _stage_train_stacking()
        elif stage_name == "finalize_prep":
            _stage_finalize_prep()
        elif stage_name == "finalize_agg":
            _stage_finalize_agg()
        elif stage_name in FINALIZE_STAT_KEYS:
            _finalize_pb_one(FINALIZE_STAT_KEYS[stage_name])
        elif stage_name in FINALIZE_ML_KEYS:
            _finalize_ml_one(FINALIZE_ML_KEYS[stage_name])
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
            print(f"{CYAN}♻️  Rechargement du cache...{RESET}")
            ok = _safe_reload_models(state)
            if ok:
                print(f"{GREEN}✅ Cache rechargé.{RESET}")
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
    parser.add_argument("--purge-stale", action="store_true", help="Supprime les marqueurs de stages périmés (schéma de features)")
    parser.add_argument("--reset-progress", action="store_true", help="Réinitialise tout le progrès des stages")
    parser.add_argument("--status", action="store_true", help="Affiche l'état du cache et des modèles puis quitte")
    parser.add_argument("--no-banner", action="store_true", help="N'affiche pas la bannière")
    parser.add_argument("--quiet", action="store_true", help="Réduit la sortie console au strict minimum")
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    _install_signal_handlers()
    args = _parse_args(argv)

    if args.purge_stale:
        n_purged = _purge_stale_markers()
        print(f"🧹 {n_purged} marqueur(s) de stage périmé(s) supprimé(s) (schéma de features {EXPECTED_FEATURE_COUNT})")
        return 0

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

    if not _safe_reload_models(state):
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
