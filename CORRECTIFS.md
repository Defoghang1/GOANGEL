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

## Correctif « 0 ML + 0 PB » (prédictions toutes en mode dégradé)
Cause reproduite : un préprocesseur (scaler.pkl) créé AVANT l'ajout de la sélection de features n'a pas l'attribut `selected_idx` ; `transform()` levait AttributeError, l'erreur n'allait que dans le fichier de log et chaque match retombait sur le mode dégradé.
- goangel_core.py : accès tolérant (getattr) -> un ancien cache cohérent refonctionne (testé).
- predict.py : la cause exacte est affichée sur chaque match (« cause : … ») et un diagnostic des modèles s'affiche avant les prédictions.
- Cache MÉLANGÉ (nouveaux modèles + ancien scaler, typique d'une restauration Drive avec --ignore-existing) : impossible à réparer à la volée ; le diagnostic le signale -> relancer finalize_agg (ou tous les stages) pour régénérer scaler.pkl.
- Test : python3 tests/full_chain.py (historique synthétique -> features -> entraînement -> cache -> prédiction, cas neuf / ancien cohérent / mélangé).

## V5 — LogLoss : nouvelles features de force + faille temporelle corrigée
Constat sur tes vrais logs : XGB/LGBM/CatBoost VAL ≈ 1.021, TEST ≈ 1.026 (constant 1.077), calibration « sans gain » car déjà bien calibrés (ECE ≈ 0.018) -> le plafond vient du signal des features, pas de la calibration.
- FAILLE TEMPORELLE (features.py) : les valeurs par défaut des équipes sans historique (`*_gf_medium`, `*_ga_*`, `points_rate`…) étaient calculées sur les 65 % premiers matchs, donc sur des matchs FUTURS pour les lignes précoces. Remplacées par une moyenne courante (cumul des matchs déjà joués). Test : changer le résultat du match i ne modifie plus aucune feature des matchs ≤ i.
- 18 nouvelles features (190 de base, 275 avec différentielles, ajoutées EN FIN : les 172 premières colonnes sont inchangées) : ratings Poisson attaque/défense en ligne + probabilités 1X2 Poisson, ELO global par équipe avec régression de saison (ELO_SEASON_REGRESSION n'était jamais utilisé), formes à décroissance exponentielle (points / buts pour / buts contre). Les différentielles diff_* correspondantes sont générées automatiquement.
- Mesure (monde synthétique, 9 000 matchs) : LogLoss TEST 0.9430 -> 0.9358 (constant 1.066). Gain modeste ; sur données réelles il peut être plus faible ou plus fort : seul le prochain run le dira.
- Schéma des artefacts -> v5 ; marqueurs .done d'un ancien jeu de features invalidés automatiquement (run_stage.sh appelle `main.py --purge-stale`). `prepare` n'est pas refait ; `refresh` régénère features + sélection.
- Observations sur ton run (non modifiées) : « Token ignoré (trop long 847 > 512) » = des jetons collés sans séparateur dans le secret BZZOIRO_TOKENS ; « Tentative 1/15 » (bash) puis « attempt 2/15 » (python) : chaque stage consomme 2 tentatives.

## V6 — Un conseil sur CHAQUE match (fin de l'abstention)
Constat (tes logs du 27-30/10) : les cotes étaient bien récupérées, mais aucun marché n'atteignait EV ≥ 4 % après fusion avec le marché -> « Aucun marché valide » sur presque tous les matchs.
Nouvelle sélection en 3 niveaux, toujours étiquetée :
1. [valeur] : EV dans [4 %, 30 %], proba minimale par marché, écart modèle/marché ≤ 15 pts (comme avant).
2. [sûr] : sinon, parmi les marchés cotés avec proba ≥ 50 %, la meilleure espérance (proba × cote). Les petites cotes sont acceptées. L'EV réelle est affichée, même négative, avec « pas d'avantage statistique » et mise Kelly 0 %.
3. [sans cote] : si aucune cote exploitable, le marché dont la proba dépasse le plus son niveau habituel (≥ 50 %, hors 1ère MT).
Les anciens refus (modèles insuffisants, mode dégradé, calibration non validée, désaccord entre modèles) deviennent des avertissements ⚠️ affichés sous le conseil.
Attention : les niveaux [sûr] et [sans cote] ne garantissent aucun avantage statistique ; seul [valeur] correspond à une EV positive mesurée.

## V7 — EV et mise Kelly désactivées
- Plus aucune ligne EV / MISE KELLY à l'affichage, et elles ne servent plus à choisir le pari.
- Choix 100 % statistique (predict.py, `_select_best_bet_statistical`) : parmi les marchés du match, probabilité modèle ≥ 50 %, puis le plus fort écart de probabilité par rapport au niveau habituel du marché. Un conseil sur chaque match, avec l'indicateur de fiabilité existant.
- La cote reste affichée à titre d'information (« indisponible » si l'API n'en fournit pas pour ce marché) ; elle ne modifie pas le choix (testé).
- Les anomalies (modèles insuffisants, mode dégradé, calibration, désaccord) restent des avertissements ⚠️.

## V6 — cotes, prédiction systématique, 141 features, 30 ligues, 30 modèles, code nettoyé
- Cotes : `odds.py` extrait toutes les cotes trouvées (champs plats, blocs imbriqués, listes de sélections, dictionnaires par bookmaker ; meilleure cote retenue), quelle que soit leur valeur : 1X2, double chance, DNB, Over/Under 0.5 à 4.5, BTTS, mi-temps. L'ancien code abandonnait tout si une des trois cotes 1X2 manquait. Sources essayées dans l'ordre : champs de l'événement, `/odds/`, `/odds/comparison/`. Récupération en parallèle ; résumé affiché ; si aucune cote n'est reconnue, les champs numériques vus sont affichés.
- Décision : `betting.py`. Un pari concret est TOUJOURS proposé, avec son niveau : VALUE (EV 4 à 30 %, proba et accord modèle/marché OK, mise Kelly), PROBABLE (cote disponible, même petite, EV affichée même négative), STATISTIQUE (sans cote : marché le plus au-dessus de sa fréquence habituelle), ISSUE LA PLUS PROBABLE. Jusqu'à 2 alternatives d'autres familles de marchés. Les avertissements (modèles insuffisants, calibration, désaccord) s'affichent sans bloquer.
- Features : liste figée de 141 (108 + 33) dans `config.FEATURES_SELECTED`, plus aucun tri automatique. `diff_h2h_win_rate` n'existait pas : ajoutée. Les 18 features de force de la V5 sont toujours calculées mais NON utilisées (conformément à la consigne) ; pour les tester, ajouter leurs noms à `FEATURES_SELECTED`.
- Ligues : `TOP_30_LEAGUES` partout (collecte, historique, mode interactif). Six ligues sont nouvelles (7, 32, 33, 94, 96, 100) : relancer `prepare`. Marqueurs de stages invalidés automatiquement (signature des ligues, des features, de la sélection).
- Modèles : `registry.py` = MODELS_30, source unique pour main.py, checkpoints et train.yml (régénéré par `.github/scripts/gen_train_yml.py`, 64 stages). Nouveaux : EloRatingModel, OrdinalRegressionModel, NegativeBinomialRegression, BayesianDynamicStrengthModel (`stat_models.py`, testés). Retirés : ExcelFormer, HMM, HierarchicalADVI, DL (remplacé par MLP_PyTorch).
- Bug corrigé : un bloc d'import de main.py échouait en silence (`_make_limix_factory` inexistant) et faisait SAUTER TabM, RealMLP, NGBoost et Skellam sans erreur.
- Mode interactif : plus d'entraînement en ligne si le cache est invalide (il utilisait l'ancien code à 172 features) ; message clair à la place.
- Code : tous commentaires et docstrings supprimés (py, sh, yml), 150+ fonctions/constantes mortes, 200+ imports inutiles, doublons de config, fichiers inutilisés (training_runner.py, checkpoint.py, confg_loader.py, config_loader.py, pb_retry_check.sh, restore-cache.yml qui appelait un script inexistant).

## Non fait / à vérifier
- Absences et blessures : non disponibles dans l'historique ; non implémentées. Handicaps : non implémentés (format de cotes inconnu).
- BayesianGoalModel / BayesianHierarchicalModel (MCMC penaltyblog) : branchés, non testables ici ; après 3 tentatives incomplètes ils sont ignorés au lieu de bloquer la chaîne.
- Structure des cotes de l'API Bzzoiro non vérifiée : lancer `python3 test_bzzoiro_features.py` et, si « Cotes récupérées » reste faible, envoyer les champs affichés.
- Tests : tests/run.py tests.test_odds | test_betting | test_printer | test_stat_models | test_core | test_ratings ; tests/e2e.py ; tests/full_chain.py.
