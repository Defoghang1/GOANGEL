MODELS_30 = [
    "CatBoost",
    "XGBoost",
    "LightGBM",
    "LogisticRegression",
    "NGBoost",
    "RandomForest",
    "ExtraTrees",
    "HistGradientBoosting",
    "MLP_PyTorch",
    "TabM",
    "RealMLP",
    "PoissonGoalsModel",
    "DixonColesGoalModel",
    "BivariatePoissonGoalModel",
    "NegativeBinomialGoalModel",
    "ZeroInflatedPoissonGoalsModel",
    "WeibullCopulaGoalsModel",
    "DixonColesDynamicModel",
    "BayesianGoalModel",
    "BayesianHierarchicalModel",
    "EloRatingModel",
    "OrdinalRegressionModel",
    "SkellamGoalDifferenceModel",
    "BayesianDynamicStrengthModel",
    "KalmanBayesianModel",
    "SupportVectorMachine",
    "NegativeBinomialRegression",
    "CopulaModel",
    "MonteCarloSimulator",
    "StackingEnsemble",
]

ML_MODELS = (
    ("xgb", "XGB", "XGBoost"),
    ("lgbm", "LGBM", "LightGBM"),
    ("catboost", "CatBoost", "CatBoost"),
    ("lr", "LR", "LogisticRegression"),
    ("ngboost", "NGBoost", "NGBoost"),
    ("rf", "RF", "RandomForest"),
    ("et", "ET", "ExtraTrees"),
    ("histgb", "HistGB", "HistGradientBoosting"),
    ("mlp", "MLP", "MLP_PyTorch"),
    ("tabm", "TabM", "TabM"),
    ("realmlp", "RealMLP", "RealMLP"),
    ("svm", "SVM", "SupportVectorMachine"),
    ("ordinal", "Ordinal", "OrdinalRegressionModel"),
    ("negbin_reg", "NegBinReg", "NegativeBinomialRegression"),
)

STAT_MODELS = (
    ("pb_poisson", "PoissonGoalsModel", "PoissonGoalsModel"),
    ("pb_dixon_coles", "DixonColesGoalModel", "DixonColesGoalModel"),
    ("pb_bivariate", "BivariatePoissonGoalModel", "BivariatePoissonGoalModel"),
    ("pb_negbinomial", "NegativeBinomialGoalModel", "NegativeBinomialGoalModel"),
    ("pb_zeroinflated", "ZeroInflatedPoissonGoalsModel", "ZeroInflatedPoissonGoalsModel"),
    ("pb_weibull", "WeibullCopulaGoalsModel", "WeibullCopulaGoalsModel"),
    ("dixon_coles_dynamic", "DixonColesDynamicModel", "DixonColesDynamicModel"),
    ("bayesian_goal", "BayesianGoalModel", "BayesianGoalModel"),
    ("bayesian_hier", "HierarchicalBayesianGoalModel", "BayesianHierarchicalModel"),
    ("elo", "EloRatingModel", "EloRatingModel"),
    ("skellam", "SkellamModel", "SkellamGoalDifferenceModel"),
    ("bayes_dynamic", "BayesianDynamicStrengthModel", "BayesianDynamicStrengthModel"),
    ("kalman", "KalmanBayesianModel", "KalmanBayesianModel"),
    ("copula", "CopulaModel", "CopulaModel"),
    ("monte_carlo", "MonteCarloSimulator", "MonteCarloSimulator"),
)

BAYESIAN_PB_MODELS = ("BayesianGoalModel", "HierarchicalBayesianGoalModel")
STACKING_STAGE = "stacking"
STACKING_KEY = "Stacking"

ML_STAGE_KEYS = {f"train_{s}": k for s, k, _ in ML_MODELS}
STAT_STAGE_KEYS = {f"train_{s}": k for s, k, _ in STAT_MODELS}
FINALIZE_ML_KEYS = {f"finalize_{s}": k for s, k, _ in ML_MODELS}
FINALIZE_STAT_KEYS = {f"finalize_{s}": k for s, k, _ in STAT_MODELS}
FINALIZE_ML_KEYS[f"finalize_{STACKING_STAGE}"] = STACKING_KEY

TRAIN_STAGES = tuple(f"train_{s}" for s, _, _ in ML_MODELS + STAT_MODELS) + (f"train_{STACKING_STAGE}",)
FINALIZE_STAGES = ("finalize_prep",) + tuple(f"finalize_{s}" for s, _, _ in ML_MODELS + STAT_MODELS) + (f"finalize_{STACKING_STAGE}", "finalize_agg")
STAGES_ORDER = ("prepare", "refresh") + TRAIN_STAGES + FINALIZE_STAGES

DISPLAY_NAMES = [d for _, _, d in ML_MODELS + STAT_MODELS] + ["StackingEnsemble"]
assert sorted(DISPLAY_NAMES) == sorted(MODELS_30), "Le registre doit correspondre exactement à MODELS_30"
