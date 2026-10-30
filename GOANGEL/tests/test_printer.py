import os, sys, io, contextlib
os.environ["BZZOIRO_TOKENS"] = "tok_aaaaaaaaaaaa"
sys.path[:0] = ["tests/stubs", "."]
import pandas as pd
import predict as P
from odds import extract_markets

def render(odds):
    m = P._build_output_from_lambdas(1.8, 1.1, 0.45)
    m["bookmaker_odds"] = odds
    m["model_count_ml"], m["model_count_pb"], m["model_count_total"] = 5, 5, 10
    m["fallback"] = False
    row = pd.Series({"home_raw": "Alpha FC", "away_raw": "Beta FC"})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        P._print_match_prediction(row, m, None, None)
    return buf.getvalue()

def test_prediction_without_odds_still_gives_a_pick():
    out = render({})
    assert "CONSEILLER" in out and "Aucun pari conseillé" not in out and "indisponible" in out

def test_prediction_with_small_odds_uses_them():
    odds = extract_markets({"odds_home": 1.42, "odds_draw": 4.6, "odds_away": 7.0, "odds_over_15": 1.20, "odds_under_15": 4.4})
    out = render(odds)
    print(out[-600:])
    assert "CONSEILLER" in out and "cotes disponibles : 5/" in out and "EV" in out

def test_partial_odds_only_one_market():
    out = render(extract_markets({"odds_btts_yes": 1.7}))
    assert "cotes disponibles : 1/" in out and "CONSEILLER" in out
