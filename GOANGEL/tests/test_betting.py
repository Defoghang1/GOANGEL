import sys
sys.path[:0] = ["."]
from betting import build_candidates, recommend

def cands(p, o15=.75, o25=.55, btts=.55, o35=.3, ht=None):
    return build_candidates(p, "H", "A", o15, o25, btts, over35=o35, ht_hda=ht, ht_over05=.7, ht_over15=.3)

def test_always_returns_a_pick_without_any_odds():
    for p in ([0.28, 0.25, 0.47], [0.34, 0.32, 0.34], [0.62, 0.22, 0.16]):
        r = recommend(cands(p), p, None)
        assert r["pick"]["name"] and r["n_priced"] == 0

def test_small_odds_are_used_not_ignored():
    p = [0.70, 0.19, 0.11]
    odds = {"home": 1.30, "draw": 5.5, "away": 9.0, "1x": 1.08, "over_1_5": 1.18, "under_1_5": 4.6}
    r = recommend(cands(p, o15=.82), p, odds)
    print(r["pick"]["name"], r["pick"]["tier_label"], r["pick"]["price"], r["pick"]["ev"])
    assert r["pick"]["price"] is not None and r["pick"]["price"] < 1.5
    assert r["n_priced"] >= 4

def test_value_bet_detected_with_ev_and_kelly():
    p = [0.60, 0.23, 0.17]
    odds = {"home": 2.10, "draw": 3.4, "away": 4.4}
    r = recommend(cands(p), p, odds)
    pk = r["pick"]
    print(pk["name"], pk["tier_label"], round(pk["ev"], 3), round(pk["kelly"], 4))
    assert pk["tier_label"] == "VALUE" and pk["ev"] >= 0.04 and pk["kelly"] > 0

def test_longshot_not_recommended_as_value():
    p = [0.159, 0.186, 0.655]
    odds = {"home": 9.30, "draw": 5.6, "away": 1.22}
    r = recommend(cands(p), p, odds)
    assert r["pick"]["key"] != "home"

def test_alternatives_are_other_families():
    p = [0.45, 0.27, 0.28]
    r = recommend(cands(p), p, None)
    fams = [r["pick"]["family"]] + [a["family"] for a in r["alternatives"]]
    assert len(set(fams)) == len(fams) and len(r["alternatives"]) == 2

def test_half_time_markets_priced_when_odds_exist():
    p = [0.40, 0.30, 0.30]
    ht = [0.30, 0.45, 0.25]
    r = recommend(cands(p, ht=ht), p, {"ht_draw": 2.4, "ht_home": 3.4, "ht_away": 4.2}, ht_hda=ht)
    assert r["n_priced"] == 3
