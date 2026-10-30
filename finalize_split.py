import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("goangel.finalize_split")

FINALIZE_DIR: Path = Path("./cache/stages/finalize")
FINALIZE_DIR.mkdir(parents=True, exist_ok=True)
CONTEXT_FILE: Path = FINALIZE_DIR / "context.pkl"


def _atomic_write(path: Path, obj: Any) -> None:
    import lzma
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    data = lzma.compress(raw, preset=3) if len(raw) > 5 * 1024 * 1024 else raw
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _atomic_read(path: Path) -> Any:
    import lzma
    with open(path, "rb") as f:
        magic = f.read(6)
    if magic[:6] == b"\xfd7zXZ\x00":
        with open(path, "rb") as f:
            return pickle.loads(lzma.decompress(f.read()))
    with open(path, "rb") as f:
        return pickle.load(f)


def save_context(ctx: Dict[str, Any]) -> Path:
    _atomic_write(CONTEXT_FILE, ctx)
    return CONTEXT_FILE


def partial_path(model_key: str) -> Path:
    return FINALIZE_DIR / f"{model_key}.pkl"


def save_partial(model_key: str, payload: Dict[str, Any]) -> Path:
    p = partial_path(model_key)
    _atomic_write(p, payload)
    return p


def list_partials() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in sorted(FINALIZE_DIR.glob("*.pkl")):
        if p.name == "context.pkl":
            continue
        key = p.stem
        try:
            out[key] = _atomic_read(p)
        except Exception:
            continue
    return out
