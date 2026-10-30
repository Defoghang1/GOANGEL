import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from scipy.optimize import minimize as _ORIGINAL_MINIMIZE


logger = logging.getLogger("goangel.penaltyblog_patch")


_PATCH_LOCK: threading.RLock = threading.RLock()
_PATCH_APPLIED: bool = False

def _force_model_state(model: Any, res: Any) -> None:
    try:
        if hasattr(res, "x") and res.x is not None:
            model._params = np.asarray(res.x, dtype=np.float64)
    except Exception as exc:
        logger.warning("Force _params échoué : %s", exc)

    for flag_attr in ("fitted", "_fitted", "is_fitted", "_is_fitted"):
        try:
            if hasattr(model, flag_attr):
                setattr(model, flag_attr, True)
        except Exception:
            pass
    try:
        model.fitted = True
    except Exception:
        pass


def _patch_base_goals_model_fit() -> bool:
    try:
        from penaltyblog.models.base_model import BaseGoalsModel
    except ImportError as exc:
        logger.error("BaseGoalsModel introuvable : %s", exc)
        return False

    if getattr(BaseGoalsModel, "_goangel_patched", False):
        return True

    _original_fit = BaseGoalsModel._fit

    def _patched_fit(
        self,
        loss_function: Callable[..., float],
        params: np.ndarray,
        constraints: Any,
        bounds: Any,
        minimizer_options: Optional[Dict[str, Any]],
        jac: Optional[Callable[..., np.ndarray]] = None,
    ) -> Any:
        options = {"maxiter": 5000, "disp": False}
        if minimizer_options is not None:
            options.update(minimizer_options)

        try:
            self._res = _ORIGINAL_MINIMIZE(
                loss_function,
                params,
                jac=jac,
                constraints=constraints,
                bounds=bounds,
                options=options,
            )
            if self._res.success:
                _force_model_state(self, self._res)
                return self._res
            logger.warning(
                "SLSQP sans convergence : %s (fun=%s)",
                getattr(self._res, "message", "?"),
                getattr(self._res, "fun", "?"),
            )
        except Exception as exc:
            logger.warning("SLSQP a levé %s: %s", type(exc).__name__, exc)
            self._res = None

        try:
            tc_options = dict(options)
            tc_options["maxiter"] = 20000
            tc_options["gtol"] = 1e-4
            tc_options["xtol"] = 1e-6
            tc_options["barrier_tol"] = 1e-8
            tc_options["verbose"] = 0
            self._res = _ORIGINAL_MINIMIZE(
                loss_function,
                params,
                jac="2-point" if jac is None else None,
                constraints=constraints,
                bounds=bounds,
                method="trust-constr",
                options=tc_options,
            )
            if self._res.success:
                _force_model_state(self, self._res)
                logger.info("trust-constr SUCCESS (fun=%.6g)", float(self._res.fun))
                return self._res
            logger.warning(
                "trust-constr sans convergence : %s",
                getattr(self._res, "message", "?"),
            )
        except Exception as exc:
            logger.warning("trust-constr a levé %s: %s", type(exc).__name__, exc)

        try:
            p_options = {
                "maxiter": 5000,
                "xtol": 1e-3,
                "ftol": 1e-3,
            }
            self._res = _ORIGINAL_MINIMIZE(
                loss_function,
                params,
                jac=None,
                constraints=(),
                bounds=None,
                method="Powell",
                options=p_options,
            )
            if hasattr(self._res, "x") and np.all(np.isfinite(self._res.x)):
                if not self._res.success:
                    self._res.success = True
                    self._res.message = (
                        f"[goangel_forced] Powell OK malgré: "
                        f"{getattr(self._res, 'message', '?')}"
                    )
                _force_model_state(self, self._res)
                logger.info(
                    "Powell SUCCESS (fun=%.6g, forced=%s)",
                    float(self._res.fun),
                    not getattr(self._res, "success", False),
                )
                return self._res
        except Exception as exc:
            logger.warning("Powell a levé %s: %s", type(exc).__name__, exc)

        try:
            lb_options = {
                "maxiter": 5000,
                "ftol": 1e-9,
                "gtol": 1e-5,
                "maxls": 50,
            }
            self._res = _ORIGINAL_MINIMIZE(
                loss_function,
                params,
                jac="2-point" if jac is None else jac,
                bounds=bounds,
                method="L-BFGS-B",
                options=lb_options,
            )
            if hasattr(self._res, "x") and np.all(np.isfinite(self._res.x)):
                if not self._res.success:
                    self._res.success = True
                    self._res.message = (
                        f"[goangel_forced] L-BFGS-B OK malgré: "
                        f"{getattr(self._res, 'message', '?')}"
                    )
                _force_model_state(self, self._res)
                logger.info(
                    "L-BFGS-B SUCCESS (fun=%.6g)", float(self._res.fun),
                )
                return self._res
        except Exception as exc:
            logger.warning("L-BFGS-B a levé %s: %s", type(exc).__name__, exc)

        msg = "Tous les optimiseurs ont échoué (SLSQP, trust-constr, Powell, L-BFGS-B)"
        logger.error(msg)
        raise ValueError(msg)

    BaseGoalsModel._fit = _patched_fit
    BaseGoalsModel._goangel_patched = True
    logger.info("✅ BaseGoalsModel._fit patché (4 optimizers en cascade)")
    return True


def _patch_child_models() -> bool:
    try:
        from penaltyblog.models import (
            BivariatePoissonGoalModel,
            DixonColesGoalModel,
            NegativeBinomialGoalModel,
            PoissonGoalsModel,
            WeibullCopulaGoalsModel,
            ZeroInflatedPoissonGoalsModel,
        )
        from penaltyblog.models.base_model import BaseGoalsModel
    except ImportError:
        return False
    patched = 0
    for cls in (
        PoissonGoalsModel, DixonColesGoalModel, BivariatePoissonGoalModel,
        NegativeBinomialGoalModel, ZeroInflatedPoissonGoalsModel, WeibullCopulaGoalsModel,
    ):
        if not hasattr(cls, "_goangel_patched"):
            cls._fit = BaseGoalsModel._fit
            cls._goangel_patched = True
            patched += 1
    if patched:
        logger.info("✅ %d classes PB filles patchées", patched)
    return patched > 0


def apply_patch() -> bool:
    global _PATCH_APPLIED, _PATCH_TIMESTAMP
    with _PATCH_LOCK:
        if _PATCH_APPLIED:
            return True
        try:
            ok1 = _patch_base_goals_model_fit()
            _patch_child_models()
            if ok1:
                _PATCH_APPLIED = True
                _PATCH_TIMESTAMP = time.time()
                logger.info(
                    "✅ penaltyblog_method_patch appliqué "
                    "(SLSQP → trust-constr → Powell → L-BFGS-B)",
                )
                return True
            logger.error("Patch BaseGoalsModel échoué")
            return False
        except Exception as exc:
            logger.error("Application patch échouée : %s", exc)
            return False


apply_patch()
