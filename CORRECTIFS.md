# GOANGEL — correctifs v4

Remplacer : goangel_core.py, main.py, predict.py, api.py, features.py, config.py (tests/ et stubs/ sont optionnels).
Puis relancer `prepare` (ou `refresh`) et TOUS les stages train_* / finalize_* : le schéma passe à `goangel-core-v4`, les anciens modèles sont détectés périmés.

## Causes réelles trouvées dans le code
1. predict.py : le 1X2 final = 50 % ensemble validé + 50 % matrice Poisson/ELO heuristique jamais calibrée -> biais favoris/outsiders, LogLoss réel pire que celui mesuré à l'entraînement. Supprimé ; les λ sont maintenant ajustés pour reproduire le 1X2 de l'ensemble (fit_lambdas_to_hda).
2. Calibration : le garde-fou testait un gain de 5e-4 sur ~900 points (erreur-type ≈ 0.015) -> décision quasi aléatoire, « identity » presque toujours. Remplacé par validation forward-chaining + test apparié (gain ≥ 5e-4 ET t ≥ 1). Test synthétique : détecte la sur-confiance (-0.13 LogLoss) et rejette un modèle déjà calibré.
3. Sélection de features : n'existait pas dans le code fourni (le « n_blocks_ok: 0 » vient d'un autre fichier). Ajout FeatureSelector : train seul, permutation sur split chronologique interne + information mutuelle, filtre couverture, dé-duplication (différentielles préférées), jamais de bruit ajouté pour « faire le quota », max 100. Persistée dans cache/stages/feature_selection.json (même colonnes pour tous les stages et l'inférence).
4. predict.py passait 251 colonnes brutes aux arbres : corrigé (transform_raw avec sélection).
5. Early stopping à 50 tours avec lr=0.01 : arrêt sur le bruit du LogLoss VAL -> 200.
6. Dates PB : liste complète passée à des modèles entraînés sur le train (longueurs incohérentes, pondération temporelle jamais appliquée). DixonColesDynamic : `.dt` sur un DatetimeIndex (AttributeError). ADVI : indexation d'un InferenceData. Corrigés.
7. Mode interactif : l'état ELO/forme était reconstruit sur tout l'historique pour CHAQUE match. Construit une seule fois ; filtre TOP_60_LEAGUES à la source (api.py) ; messages de progression.
8. Décision : proba modèle fusionnée avec le marché (sans marge), abstention si écart modèle/marché > 15 pts, proba minimale par type (1X2 40 %, DC 62 %, buts/BTTS 52 %), EV dans [4 %, 30 %], pas de conseil sans cote, Over/Under 3.5 et BTTS ajoutés, DC valorisée au bon prix. Scores exacts et ordre d'affichage inchangés.

## Problèmes du rapport
- Corrigés + testés (synthétique) : 01, 02, 03, 05, 06, 07, 08, 09, 10, 11, 13, 15, 24, 25, 27.
- Corrigés, non testables ici (modules absents) : 14 (partiel : early stopping), 26.
- Non corrigés : 04 (l'API ne fournit pas de cotes mi-temps : affichés, jamais conseillés), 12 (la clé « draw_no_bet » contient en réalité les cotes 1X/X2 de l'API : le calcul était cohérent, simple nom trompeur), 16 (pas de recherche d'hyperparamètres), 17 (cotes historiques : ~23 000 appels API, à décider), 18, 19, 20-23 (mémoire longue/ELO de saison/EWMA), 28-32 (infrastructure/GitHub Actions).

## Limites
- Testé uniquement sur données synthétiques (RF + HistGB). xgboost, lightgbm, catboost, penaltyblog, pymc non installés ici : jamais exécutés. Le correctif ADVI n'est pas testé.
- Aucun LogLoss réel n'est prédit. Sans cotes en entrée, ~0.99-1.00 est réaliste ; 0.94-0.96 est le niveau du marché.
- Lancer : python3 tests/run.py tests.test_core ; tests.test_decision ; python3 tests/e2e.py
