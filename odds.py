import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

LINES = ("0.5", "1.5", "2.5", "3.5", "4.5")
HT_PREFIXES = ("ht", "half_time", "halftime", "first_half", "1st_half", "1h", "h1")
LABEL_KEYS = ("name", "label", "selection", "outcome", "bet", "type", "key")
PRICE_KEYS = ("odds", "price", "decimal", "coef", "value")
MARKET_KEYS = ("market", "market_name", "market_key")
MIN_PRICE = 1.01
MAX_PRICE = 1000.0


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _price(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v < MIN_PRICE or v > MAX_PRICE:
        return None
    return v


def _line_tokens(line: str) -> Tuple[str, str]:
    us = line.replace(".", "_")
    return us, us.replace("_", "")


def _build_aliases() -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {
        "home": ["home_win", "home", "1", "h2h_home", "1x2_home", "match_winner_home", "full_time_result_home", "win_home", "moneyline_home"],
        "draw": ["draw", "x", "h2h_draw", "1x2_draw", "match_winner_draw", "full_time_result_draw", "moneyline_draw"],
        "away": ["away_win", "away", "2", "h2h_away", "1x2_away", "match_winner_away", "full_time_result_away", "win_away", "moneyline_away"],
        "1x": ["1x", "home_win_or_draw", "home_or_draw", "double_chance_1x", "dc_1x", "home_draw", "1_x"],
        "x2": ["x2", "away_win_or_draw", "draw_or_away", "away_or_draw", "double_chance_x2", "dc_x2", "draw_away", "away_draw", "x_2"],
        "12": ["12", "home_or_away", "home_win_or_away_win", "double_chance_12", "dc_12", "home_away", "1_2"],
        "dnb_home": ["draw_no_bet_home", "dnb_home", "dnb_1"],
        "dnb_away": ["draw_no_bet_away", "dnb_away", "dnb_2"],
        "btts_yes": ["btts_yes", "both_teams_to_score_yes", "both_teams_score_yes", "btts_gg", "gg", "btts_y"],
        "btts_no": ["btts_no", "both_teams_to_score_no", "both_teams_score_no", "btts_ng", "ng", "btts_n"],
    }
    for line in LINES:
        us, c = _line_tokens(line)
        for side in ("over", "under"):
            aliases[f"{side}_{us}"] = [f"{side}_{us}", f"{side}_{c}", f"{side}_{us}_goals", f"{side}_{c}_goals", f"{us}_{side}", f"{c}_{side}", f"total_{side}_{us}", f"goals_{side}_{us}"]
    return aliases


def _build_ht_aliases(base: Dict[str, List[str]]) -> Dict[str, List[str]]:
    ht: Dict[str, List[str]] = {}
    for key in ("home", "draw", "away", "1x", "x2", "12", "btts_yes", "btts_no"):
        ht[f"ht_{key}"] = [f"{p}_{a}" for p in HT_PREFIXES for a in base[key]]
    for line in LINES[:3]:
        us, _ = _line_tokens(line)
        for side in ("over", "under"):
            ht[f"ht_{side}_{us}"] = [f"{p}_{a}" for p in HT_PREFIXES for a in base[f"{side}_{us}"]]
    return ht


FT_ALIASES = _build_aliases()
HT_ALIASES = _build_ht_aliases(FT_ALIASES)
_HT_MARK = re.compile(r"(^|_)(ht|half_time|halftime|first_half|1st_half|1h|h1)(_|$)")


def _flatten(obj: Any, path: str = "") -> Iterator[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{path}_{_norm(k)}" if path else _norm(k))
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                label = next((item[k] for k in LABEL_KEYS if isinstance(item.get(k), str)), None)
                price = next((item[k] for k in PRICE_KEYS if _price(item.get(k)) is not None), None)
                market = next((item[k] for k in MARKET_KEYS if isinstance(item.get(k), str)), None)
                if label is not None and price is not None:
                    prefix = f"{path}_{_norm(market)}" if market else path
                    yield f"{prefix}_{_norm(label)}" if prefix else _norm(label), item[next(k for k in PRICE_KEYS if _price(item.get(k)) is not None)]
                    continue
            yield from _flatten(item, path)
    else:
        yield path, obj


def _match(path: str) -> Optional[str]:
    is_ht = bool(_HT_MARK.search(path))
    table = HT_ALIASES if is_ht else FT_ALIASES
    best_key, best_len = None, 0
    for key, names in table.items():
        for name in names:
            if (path == name or path.endswith("_" + name)) and len(name) > best_len:
                best_key, best_len = key, len(name)
    return best_key


def extract_markets(*payloads: Any) -> Dict[str, float]:
    found: Dict[str, List[float]] = {}
    for payload in payloads:
        if payload is None:
            continue
        for path, value in _flatten(payload):
            price = _price(value)
            if price is None or not path:
                continue
            key = _match(path)
            if key is None and "_" in path:
                key = _match(path.rsplit("_", 1)[0])
            if key is not None:
                found.setdefault(key, []).append(price)
    return {k: max(v) for k, v in found.items()}


def sample_paths(payload: Any, limit: int = 25) -> List[str]:
    out: List[str] = []
    for path, value in _flatten(payload):
        if _price(value) is not None:
            out.append(f"{path}={value}")
        if len(out) >= limit:
            break
    return out
