import json
import logging
import lzma
import os
import pickle
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("goangel.checkpoint_all")

_LOCK = threading.RLock()
_SCHEMA = "goangel-checkpoint-2.0"
_COMPRESS_THRESHOLD = 5 * 1024 * 1024
_ROOT = Path("./cache/stages/checkpoints")
_LEGACY_ROOTS = [
    Path("./cache/stages/checkpoints_universal"),
    Path("./cache/stages/pb_convergence/checkpoints"),
    Path("./cache/stages/pb_convergence/wf_checkpoints"),
]

from registry import FINALIZE_STAGES as FINALIZE_KEYS
from registry import ML_MODELS as _REGISTRY_ML
from registry import STAT_MODELS as _REGISTRY_STAT

ML_MODELS: Tuple[str, ...] = tuple(key for _, key, _ in _REGISTRY_ML) + ("Stacking",)
PB_MODELS: Tuple[str, ...] = tuple(key for _, key, _ in _REGISTRY_STAT)

STEP_TRAINED = "trained"
STEP_CALIBRATED = "calibrated"
STEP_WF_DONE = "wf_done"
STEP_FINALIZED = "finalized"
STEP_CONTEXT = "context"


def _safe_key(key: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(key))


def _ensure_root() -> Path:
    _ROOT.mkdir(parents=True, exist_ok=True)
    return _ROOT


def _key_dir(key: str) -> Path:
    d = _ensure_root() / _safe_key(key)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write_bytes(path: Path, data: bytes) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)
    return len(data)


def _atomic_write_pickle(path: Path, payload: Any) -> int:
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    if len(raw) >= _COMPRESS_THRESHOLD:
        data = lzma.compress(raw, preset=6)
    else:
        data = raw
    return _atomic_write_bytes(path, data)


def _atomic_read_pickle(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            magic = f.read(6)
        if magic[:6] == b"\xfd7zXZ\x00":
            with open(path, "rb") as f:
                return pickle.loads(lzma.decompress(f.read()))
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Checkpoint illisible %s : %s", path, exc)
        return None


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> int:
    data = json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    return _atomic_write_bytes(path, data)


def _atomic_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("JSON illisible %s : %s", path, exc)
        return None


def _step_path(entity_key: str, step_name: str) -> Path:
    return _key_dir(entity_key) / f"{_safe_key(step_name)}.pkl"


def _manifest_path(entity_key: str) -> Path:
    return _key_dir(entity_key) / "_manifest.json"


def _read_manifest(entity_key: str) -> Dict[str, Any]:
    data = _atomic_read_json(_manifest_path(entity_key))
    if not isinstance(data, dict):
        return {"schema": _SCHEMA, "entity_key": entity_key, "steps": {}}
    data.setdefault("steps", {})
    return data


def _write_manifest(entity_key: str, manifest: Dict[str, Any]) -> None:
    manifest["schema"] = _SCHEMA
    manifest["entity_key"] = entity_key
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(_manifest_path(entity_key), manifest)


def save_step(
    entity_key: str,
    step_name: str,
    payload: Any,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> Optional[Path]:
    with _LOCK:
        try:
            path = _step_path(entity_key, step_name)
            if path.exists() and not overwrite:
                return path
            wrapper = {
                "schema": _SCHEMA,
                "entity_key": entity_key,
                "step_name": step_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
                "metadata": metadata or {},
            }
            size = _atomic_write_pickle(path, wrapper)
            manifest = _read_manifest(entity_key)
            manifest["steps"][step_name] = {
                "file": path.name,
                "size": size,
                "ts": wrapper["timestamp"],
                "metadata": metadata or {},
            }
            _write_manifest(entity_key, manifest)
            logger.info(
                "checkpoint_save key=%s step=%s size_kb=%.2f",
                entity_key, step_name, size / 1024.0,
            )
            return path
        except Exception as exc:
            logger.error(
                "checkpoint_save_failed key=%s step=%s err=%s",
                entity_key, step_name, exc,
            )
            return None


def has_step(entity_key: str, step_name: str) -> bool:
    with _LOCK:
        return _step_path(entity_key, step_name).exists()


def list_steps(entity_key: str) -> List[str]:
    with _LOCK:
        manifest = _read_manifest(entity_key)
        return sorted(manifest.get("steps", {}).keys())


def list_all_entities() -> List[str]:
    with _LOCK:
        if not _ROOT.exists():
            return []
        return sorted([d.name for d in _ROOT.iterdir() if d.is_dir()])


def list_folds(entity_key: str) -> Dict[int, Dict[str, Any]]:
    with _LOCK:
        d = _key_dir(entity_key)
        out: Dict[int, Dict[str, Any]] = {}
        for p in sorted(d.glob("fold_*.pkl")):
            try:
                idx = int(p.stem.split("_")[1])
                wrapper = _atomic_read_pickle(p)
                if isinstance(wrapper, dict):
                    out[idx] = wrapper.get("payload", {})
            except Exception:
                continue
        return out


def ml_key(name: str) -> str:
    return f"ml_{name}"


def pb_key(name: str) -> str:
    return f"pb_{name}"


def finalize_key(name: str) -> str:
    return name if name.startswith("finalize") else f"finalize_{name}"


def get_all_model_states() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ml": {},
        "pb": {},
        "finalize": {},
        "complete": {},
    }
    for m in ML_MODELS:
        steps = list_steps(ml_key(m))
        out["ml"][m] = {
            "trained": STEP_TRAINED in steps,
            "calibrated": STEP_CALIBRATED in steps,
            "wf_done": STEP_WF_DONE in steps,
            "steps": steps,
        }
    for m in PB_MODELS:
        steps = list_steps(pb_key(m))
        out["pb"][m] = {
            "trained": STEP_TRAINED in steps,
            "finalized": STEP_FINALIZED in steps,
            "folds": sorted(list_folds(pb_key(m)).keys()),
            "steps": steps,
        }
    for k in FINALIZE_KEYS:
        steps = list_steps(finalize_key(k))
        out["finalize"][k] = {
            "done": len(steps) > 0,
            "steps": steps,
        }
    ml_done = sum(1 for m in ML_MODELS if out["ml"][m]["trained"])
    pb_done = sum(1 for m in PB_MODELS if out["pb"][m]["trained"])
    finalize_done = sum(1 for k in FINALIZE_KEYS if out["finalize"][k]["done"])
    out["complete"] = {
        "ml_done": ml_done,
        "ml_total": len(ML_MODELS),
        "pb_done": pb_done,
        "pb_total": len(PB_MODELS),
        "finalize_done": finalize_done,
        "finalize_total": len(FINALIZE_KEYS),
        "all_done": (
            ml_done == len(ML_MODELS)
            and pb_done == len(PB_MODELS)
            and finalize_done == len(FINALIZE_KEYS)
        ),
    }
    return out


def total_size_mb() -> float:
    with _LOCK:
        if not _ROOT.exists():
            return 0.0
        total = 0
        for p in _ROOT.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total / (1024.0 * 1024.0)


def get_full_report() -> Dict[str, Any]:
    with _LOCK:
        states = get_all_model_states()
        states["total_size_mb"] = round(total_size_mb(), 2)
        states["entities"] = list_all_entities()
        states["root"] = str(_ROOT)
        return states


def migrate_legacy_layout() -> Dict[str, int]:
    stats = {"migrated": 0, "skipped": 0}
    with _LOCK:
        for legacy_root in _LEGACY_ROOTS:
            if not legacy_root.exists():
                continue
            for sub in list(legacy_root.iterdir()):
                if not sub.is_dir():
                    continue
                key = _safe_key(sub.name)
                target = _key_dir(key)
                for p in sub.glob("*.pkl"):
                    dest = target / p.name
                    if dest.exists():
                        stats["skipped"] += 1
                        continue
                    try:
                        shutil.copy2(p, dest)
                        stats["migrated"] += 1
                    except OSError:
                        stats["skipped"] += 1
    if stats["migrated"] > 0:
        logger.info(
            "checkpoint_migration migrated=%d skipped=%d",
            stats["migrated"], stats["skipped"],
        )
    return stats


def save_ml_trained(
    model_name: str,
    model_path: str,
    calib_ll: Optional[float] = None,
    accuracy: Optional[float] = None,
) -> Optional[Path]:
    return save_step(
        ml_key(model_name),
        STEP_TRAINED,
        {
            "model_path": str(model_path),
            "calib_ll": calib_ll,
            "accuracy": accuracy,
        },
    )


def save_ml_calibrated(
    model_name: str,
    calib_method: str,
    calib_ll: Optional[float] = None,
) -> Optional[Path]:
    return save_step(
        ml_key(model_name),
        STEP_CALIBRATED,
        {
            "calib_method": calib_method,
            "calib_ll": calib_ll,
        },
    )


def save_pb_finalized(
    class_name: str,
    calib_ll: float,
    wf_ll: float,
    calib_status: str,
    n_kept: int,
) -> Optional[Path]:
    return save_step(
        pb_key(class_name),
        STEP_FINALIZED,
        {
            "calib_ll": float(calib_ll),
            "wf_ll": float(wf_ll),
            "calib_status": calib_status,
            "n_kept": int(n_kept),
        },
    )


def save_finalize_context(context: Dict[str, Any]) -> Optional[Path]:
    return save_step(finalize_key("finalize_prep"), STEP_CONTEXT, context)


def bootstrap() -> Dict[str, Any]:
    _ensure_root()
    stats = migrate_legacy_layout()
    report = get_full_report()
    report["migration"] = stats
    return report


bootstrap()
