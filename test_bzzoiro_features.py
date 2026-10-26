import os
import json
import requests
from datetime import datetime, timedelta

TOKEN = os.getenv("BZZOIRO_TOKEN") or os.getenv("BZZOIRO_API_KEY")
if not TOKEN:
    # Essayer de lire depuis .env (BZZOIRO_TOKENS au pluriel)
    if os.path.exists(".env"):
        for line in open(".env"):
            if line.startswith("BZZOIRO_TOKENS=") or line.startswith("BZZOIRO_TOKEN=") or line.startswith("BZZOIRO_API_KEY="):
                raw = line.split("=", 1)[1].strip().strip('"').strip("'")
                # Prendre le premier token si séparé par des virgules
                TOKEN = raw.split(",")[0].strip()
                if TOKEN:
                    break

if not TOKEN:
    print("❌ Aucun token Bzzoiro trouvé. Définir BZZOIRO_TOKEN.")
    exit(1)

BASE = "https://sports.bzzoiro.com/api/v2"
HEADERS = {"Authorization": f"Token {TOKEN}"}

def get(path, params=None):
    try:
        r = requests.get(f"{BASE}{path}", headers=HEADERS, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        return {"error": r.status_code, "detail": r.text[:200]}
    except Exception as e:
        return {"error": str(e)}

# 1. Prendre un match récent (terminé)
yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
today = datetime.utcnow().strftime("%Y-%m-%d")

print("=" * 70)
print("  TEST DES FONCTIONNALITÉS BZZOIRO")
print("=" * 70)

events = get("/events/", {"date_from": yesterday, "date_to": today, "status": "finished", "limit": 1})
if "results" not in events or not events["results"]:
    print("❌ Aucun match trouvé. Essayer avec une autre date.")
    exit(1)

match = events["results"][0]
match_id = match["id"]
print(f"\n📅 Match test : {match['home_team']} vs {match['away_team']} (ID: {match_id})")

# 2. Tester chaque endpoint
endpoints = [
    ("/events/{id}/", "Match détail"),
    ("/events/{id}/stats/", "Statistiques"),
    ("/events/{id}/incidents/", "Incidents"),
    ("/events/{id}/lineups/", "Compositions"),
    ("/events/{id}/h2h/", "H2H"),
    ("/events/{id}/odds/", "Cotes"),
    ("/events/{id}/prediction/", "Prédictions Bzzoiro"),
    ("/events/{id}/player-stats/", "Stats joueurs"),
    ("/events/{id}/availability/", "Disponibilité"),
    ("/events/{id}/odds/comparison/", "Cotes par bookmaker"),
]

print("\n── Endpoints match ──")
for ep, name in endpoints:
    path = ep.format(id=match_id)
    data = get(path)
    if "error" in data:
        print(f"  ❌ {name:35s} : {data['error']} (inaccessible)")
    else:
        keys = list(data.keys())[:5]
        print(f"  ✅ {name:35s} : {len(keys)} clés (ex: {keys})")

# 3. Tester les endpoints équipes
home_id = match.get("home_team_id")
away_id = match.get("away_team_id")
print("\n── Endpoints équipes ──")
for team_id, team_name in [(home_id, match.get("home_team")), (away_id, match.get("away_team"))]:
    if not team_id:
        continue
    for ep, name in [
        (f"/teams/{team_id}/", "Profil équipe"),
        (f"/teams/{team_id}/form/", "Forme"),
        (f"/teams/{team_id}/squad/", "Effectif"),
        (f"/teams/{team_id}/fixtures/", "Calendrier"),
    ]:
        data = get(ep, {"last": 10} if "form" in ep else None)
        if "error" in data:
            print(f"  ❌ {name:20s} ({team_name[:15]:15s}) : {data['error']}")
        else:
            print(f"  ✅ {name:20s} ({team_name[:15]:15s}) : OK")

# 4. Tester les endpoints globaux
print("\n── Endpoints globaux ──")
for ep, name, params in [
    ("/leagues/", "Ligues", {"limit": 5}),
    ("/predictions/", "Prédictions globales", {"limit": 5}),
    ("/odds/", "Cotes globales", {"limit": 5}),
    ("/referees/", "Arbitres", {"limit": 5}),
    ("/managers/", "Entraîneurs", {"limit": 5}),
    ("/venues/", "Stades", {"limit": 5}),
    ("/coverage/", "Couverture", None),
]:
    data = get(ep, params)
    if "error" in data:
        print(f"  ❌ {name:25s} : {data['error']}")
    else:
        count = data.get("count", "?") if isinstance(data, dict) else "?"
        print(f"  ✅ {name:25s} : {count} résultats")

# 5. Afficher les champs de /prediction/
print("\n── Champs de /prediction/ (exemple) ──")
pred = get(f"/events/{match_id}/prediction/")
if "error" not in pred:
    for k, v in pred.items():
        if isinstance(v, (str, int, float, bool)):
            print(f"    {k}: {v}")
        elif isinstance(v, dict):
            print(f"    {k}: {list(v.keys())[:3]}...")
else:
    print(f"  ❌ Inaccessible : {pred['error']}")

print("\n" + "=" * 70)
print("  FIN DU TEST")
print("=" * 70)
