from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from goangel_core import blend_with_market, devig

MIN_EV = 0.04
MAX_EV = 0.30
MIN_EV_PROBABLE = -0.06
BLEND_W_MODEL = 0.40
MAX_GAP_VALUE = 0.15
MAX_GAP_PROBABLE = 0.22
KELLY_FRACTION = 0.25
MAX_KELLY = 0.05
SURE_MIN_PROB = 0.50
FLOORS = {"result": 0.40, "double_chance": 0.62, "goals": 0.55, "ht": 0.58}
TYPICAL = {
    "home": 0.45, "away": 0.30, "draw": 0.26, "1x": 0.70, "x2": 0.55, "12": 0.74,
    "dnb_home": 0.62, "dnb_away": 0.42, "over_1_5": 0.75, "under_1_5": 0.25,
    "over_2_5": 0.52, "under_2_5": 0.48, "over_3_5": 0.30, "under_3_5": 0.70,
    "btts_yes": 0.52, "btts_no": 0.48,
}
PAIRS = {
    "over_1_5": "under_1_5", "under_1_5": "over_1_5",
    "over_2_5": "under_2_5", "under_2_5": "over_2_5",
    "over_3_5": "under_3_5", "under_3_5": "over_3_5",
    "btts_yes": "btts_no", "btts_no": "btts_yes",
    "ht_over_0_5": "ht_under_0_5", "ht_under_0_5": "ht_over_0_5",
    "ht_over_1_5": "ht_under_1_5", "ht_under_1_5": "ht_over_1_5",
}
TIER_LABELS = {0: "VALUE", 1: "PROBABLE", 2: "STATISTIQUE", 3: "ISSUE LA PLUS PROBABLE"}


def expected_value(prob: float, price: float) -> float:
    return float(prob * price - 1.0)


def kelly_stake(prob: float, price: float) -> float:
    b = price - 1.0
    if b <= 0 or prob <= 0:
        return 0.0
    k = (b * prob - (1.0 - prob)) / b
    return float(np.clip(k * KELLY_FRACTION, 0.0, MAX_KELLY)) if k > 0 else 0.0


def _kind(key: str) -> str:
    if key.startswith("ht_"):
        return "ht"
    if key in ("home", "draw", "away"):
        return "result"
    if key in ("1x", "x2", "12", "dnb_home", "dnb_away"):
        return "double_chance"
    return "goals"


def _family(key: str) -> str:
    if key.startswith("ht_"):
        return "ht"
    if key in ("home", "draw", "away", "dnb_home", "dnb_away"):
        return "result"
    if key in ("1x", "x2", "12"):
        return "double_chance"
    if key.startswith("btts"):
        return "btts"
    return "goals"


def _derived(p3: Sequence[float], prefix: str) -> Dict[str, float]:
    h, d, a = float(p3[0]), float(p3[1]), float(p3[2])
    out = {f"{prefix}home": h, f"{prefix}draw": d, f"{prefix}away": a,
           f"{prefix}1x": h + d, f"{prefix}x2": d + a, f"{prefix}12": h + a}
    if not prefix and h + a > 0:
        out["dnb_home"], out["dnb_away"] = h / (h + a), a / (h + a)
    return out


def _blend2(p: float, m: float) -> float:
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    m = float(np.clip(m, 1e-4, 1 - 1e-4))
    a = BLEND_W_MODEL * np.log(p) + (1 - BLEND_W_MODEL) * np.log(m)
    b = BLEND_W_MODEL * np.log(1 - p) + (1 - BLEND_W_MODEL) * np.log(1 - m)
    return float(1.0 / (1.0 + np.exp(b - a)))


def _triple_reference(model3: Sequence[float], odds: Dict[str, float], prefix: str):
    keys = [f"{prefix}home", f"{prefix}draw", f"{prefix}away"]
    if not all(k in odds for k in keys):
        return _derived(model3, prefix), None
    market3 = devig([odds[k] for k in keys])
    adjusted = blend_with_market(model3, market3, BLEND_W_MODEL)
    return _derived(adjusted, prefix), _derived(market3, prefix)


def build_candidates(
    proba: Sequence[float],
    home: str,
    away: str,
    over15: float,
    over25: float,
    btts_yes: float,
    over35: Optional[float] = None,
    ht_hda: Optional[Sequence[float]] = None,
    ht_over05: Optional[float] = None,
    ht_over15: Optional[float] = None,
) -> List[Dict[str, Any]]:
    d = _derived(proba, "")
    rows = [
        ("home", f"Victoire {home}", d["home"]),
        ("draw", "Match Nul", d["draw"]),
        ("away", f"Victoire {away}", d["away"]),
        ("1x", f"Double chance 1X ({home} ou Nul)", d["1x"]),
        ("x2", f"Double chance X2 (Nul ou {away})", d["x2"]),
        ("12", f"Double chance 12 ({home} ou {away})", d["12"]),
        ("dnb_home", f"Draw No Bet {home}", d.get("dnb_home", 0.0)),
        ("dnb_away", f"Draw No Bet {away}", d.get("dnb_away", 0.0)),
        ("over_1_5", "+1.5 buts", over15), ("under_1_5", "-1.5 buts", 1.0 - over15),
        ("over_2_5", "+2.5 buts", over25), ("under_2_5", "-2.5 buts", 1.0 - over25),
        ("btts_yes", "BTTS — Les deux marquent (Oui)", btts_yes),
        ("btts_no", "BTTS — Les deux marquent (Non)", 1.0 - btts_yes),
    ]
    if over35 is not None:
        rows += [("over_3_5", "+3.5 buts", over35), ("under_3_5", "-3.5 buts", 1.0 - over35)]
    if ht_hda is not None and len(ht_hda) == 3:
        h = _derived(ht_hda, "ht_")
        rows += [
            ("ht_home", f"Victoire {home} (1ère MT)", h["ht_home"]),
            ("ht_draw", "Match Nul (1ère MT)", h["ht_draw"]),
            ("ht_away", f"Victoire {away} (1ère MT)", h["ht_away"]),
            ("ht_1x", f"Double chance 1X ({home} ou Nul) (1ère MT)", h["ht_1x"]),
            ("ht_x2", f"Double chance X2 (Nul ou {away}) (1ère MT)", h["ht_x2"]),
        ]
    if ht_over05 is not None:
        rows += [("ht_over_0_5", "+0.5 buts (1ère MT)", ht_over05), ("ht_under_0_5", "-0.5 buts (1ère MT)", 1.0 - ht_over05)]
    if ht_over15 is not None:
        rows += [("ht_over_1_5", "+1.5 buts (1ère MT)", ht_over15), ("ht_under_1_5", "-1.5 buts (1ère MT)", 1.0 - ht_over15)]
    out = []
    for key, name, p in rows:
        p = float(p)
        if np.isfinite(p) and 0.0 <= p <= 1.0:
            out.append({"key": key, "name": name, "p_model": p, "kind": _kind(key), "family": _family(key), "is_ht": key.startswith("ht_")})
    return out


def _score_rows(cands, proba, odds, ht_hda):
    ft_adj, ft_mkt = _triple_reference(proba, odds, "")
    ht_adj, ht_mkt = _triple_reference(ht_hda, odds, "ht_") if ht_hda is not None and len(ht_hda) == 3 else ({}, None)
    rows = []
    for c in cands:
        key, p_model = c["key"], c["p_model"]
        p, ref = p_model, None
        if key in ft_adj:
            p = ft_adj[key]
            ref = ft_mkt.get(key) if ft_mkt else None
        elif key in ht_adj:
            p = ht_adj[key]
            ref = ht_mkt.get(key) if ht_mkt else None
        elif key in PAIRS and key in odds and PAIRS[key] in odds:
            ref = float(devig([odds[key], odds[PAIRS[key]]])[0])
            p = _blend2(p_model, ref)
        price = odds.get(key)
        ev = expected_value(p, price) if price else None
        rows.append(dict(c, p=float(p), price=price, ev=ev, gap=abs(p_model - ref) if ref is not None else 0.0,
                         kelly=kelly_stake(p, price) if price else 0.0))
    return rows


def _rank(r):
    priced = r["price"] is not None
    if priced and MIN_EV <= r["ev"] <= MAX_EV and r["p"] >= FLOORS[r["kind"]] and r["gap"] <= MAX_GAP_VALUE:
        return 0, r["kelly"]
    if priced and r["p"] >= FLOORS[r["kind"]] and r["gap"] <= MAX_GAP_PROBABLE and r["ev"] >= MIN_EV_PROBABLE:
        return 1, r["ev"] + 0.1 * r["p"]
    if not r["is_ht"] and r["p"] >= SURE_MIN_PROB:
        return 2, r["p"] - TYPICAL.get(r["key"], 0.5)
    if r["p"] >= SURE_MIN_PROB:
        return 2, r["p"] - 0.5 - 0.2
    return 3, r["p"]


def recommend(
    cands: List[Dict[str, Any]],
    proba: Sequence[float],
    odds: Optional[Dict[str, float]],
    ht_hda: Optional[Sequence[float]] = None,
    n_alternatives: int = 2,
) -> Dict[str, Any]:
    odds = odds or {}
    rows = _score_rows(cands, proba, odds, ht_hda)
    ranked = sorted(rows, key=lambda r: (_rank(r)[0], -_rank(r)[1]))
    best = ranked[0]
    tier = _rank(best)[0]
    alternatives, used = [], {best["family"]}
    for r in ranked[1:]:
        if r["family"] in used:
            continue
        used.add(r["family"])
        alternatives.append(dict(r, tier=_rank(r)[0], tier_label=TIER_LABELS[_rank(r)[0]]))
        if len(alternatives) >= n_alternatives:
            break
    return {
        "pick": dict(best, tier=tier, tier_label=TIER_LABELS[tier]),
        "alternatives": alternatives,
        "n_priced": sum(1 for r in rows if r["price"] is not None),
        "n_markets": len(rows),
    }
