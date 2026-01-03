#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:11:16 2025

@author: akshayjacobthomas

Some comments on the GP in this code:
    - The priors are set in the log space of the paramters
    - I use MAP estimate, however with very weak priors. Precision very low
    - To change to MLE estimate only return nll in the ok(_) function
    - I still have to write code to add other priors, but an LLM can help you.
    - Now, if you dont like working in the log space, just work with Gamma priors
    to work woth gamma priors you need to write code
    - Its not always necessary to fit using log_y = True. That was how the data
    from Pedro was behaving.
"""

from scipy.stats import qmc
import pandas as pd
from math import ceil, log2

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Union, Dict
from pathlib import Path
from dataclasses import dataclass
from tqdm.auto import tqdm
import jax
import jax.scipy as jsp
from typing import Tuple
from jax import lax
from jaxopt import LBFGS
import json
from flax import serialization as sz
jax.config.update('jax_platform_name', 'cpu')
ArrayLike = Union[float, int, Sequence[float], np.ndarray]

_SERIALIZERS_REGISTERED = False

def _register_all_serializers_once():
    """Register custom Flax serializers exactly once per process."""
    global _SERIALIZERS_REGISTERED
    if _SERIALIZERS_REGISTERED:
        return
    # If the notebook auto-reloads and registration already happened, re-registrations
    # may throw. We ignore those specific errors to keep it idempotent.
    def _safe_register(cls, to_state, from_state):
        try:
            sz.register_serialization_state(cls, to_state, from_state)
        except Exception as e:
            # Flax raises if already registered; ignore those, re-raise others.
            msg = str(e).lower()
            if "already registered" in msg or "has been registered" in msg:
                pass
            else:
                raise

    _safe_register(RBF, _kernel_to_state, _kernel_from_state)
    _safe_register(Matern32, _kernel_to_state, _kernel_from_state)
    _safe_register(GaussianLikelihood, _lik_to_state, _lik_from_state)
    _safe_register(ZeroMean, _zeromean_to_state, _zeromean_from_state)
    _safe_register(GPParams, _gpparams_to_state, _gpparams_from_state)
    _safe_register(Standardizer, _std_to_state, _std_from_state)

    _SERIALIZERS_REGISTERED = True

class BayesianFLow:
    """
    
    class that handles data creation and surrogate modeling
    """

    def __init__(self, dim: int,n_train_samples: int, n_test_samples: int): # xcould add a boudna file next time
        
        self.dim = dim
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        
        self.lb = np.array([0.34, 1e3, 1e-3, 1e-2, 1.5e-7])
        self.ub = np.array([0.9, 1e4, 10, 12, 1e-6])
        
      
    def _sobol_exact(self, n: int, d: int, scramble: bool = True) -> np.ndarray:
        """Return exactly `n` Sobol points in [0,1]^d."""
        if n <= 0:
            raise ValueError("n must be positive.")
        sampler = qmc.Sobol(d=d, scramble=scramble)
        m = int(ceil(log2(n)))
        X = sampler.random_base2(m=m)  # 2**m points
        
        return X[:n]
        
    def create_training_set(self, scramble_=True) -> np.ndarray:
        """
        create the training set
        """
        X_= self._sobol_exact(self.n_train_samples, self.dim, scramble=scramble_)
        X = qmc.scale(X_, self.lb, self.ub)
        
        
        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(self.dim)])
        df.to_csv("train_set.csv", index=False)
        print(f"Training set saved to train_set.csv with shape {df.shape}")

    def create_test_set(self, scramble_=True) -> np.ndarray:

        X_ = self._sobol_exact(self.n_test_samples, self.dim, scramble=scramble_)
        X = qmc.scale(X_, self.lb, self.ub)
        
        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(self.dim)])
        df.to_csv("test_set.csv", index=False)
        print(f"Test set saved to test_set.csv with shape {df.shape}")
        

def softplus(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    return jax.nn.softplus(x) + eps

def _stable_cholesky(K: jnp.ndarray, base_jitter: float = 1e-6) -> jnp.ndarray:
    # Symmetrize, eigen-lift, then Cholesky
    K = 0.5 * (K + K.T)
    eigvals = jnp.linalg.eigvalsh(K)  # <-- O(N^3) and very expensive
    diag_mean = jnp.mean(jnp.diag(K))
    scale = jnp.maximum(diag_mean, 1.0)
    floor = base_jitter * scale
    min_ev = jnp.min(eigvals)
    lift = jnp.maximum(0.0, floor - min_ev)
    K_spd = K + lift * jnp.eye(K.shape[0], dtype=K.dtype)
    L = jsp.linalg.cholesky(K_spd, lower=True, check_finite=False)
    return L


# -------------------------------
# Mean
# -------------------------------

@dataclass
class ZeroMean:
    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((X.shape[0],), dtype=X.dtype)

# -------------------------------
# Kernels
# -------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass
class RBF:
    log_amp: jnp.ndarray
    log_length: jnp.ndarray
    def __call__(self, X: jnp.ndarray, Z: jnp.ndarray) -> jnp.ndarray:
        amp = softplus(self.log_amp)
        ell = softplus(self.log_length)
        diff = (X[:, None, :] - Z[None, :, :]) / ell
        sq = jnp.sum(diff * diff, axis=-1)
        return (amp ** 2) * jnp.exp(-0.5 * sq)
    def tree_flatten(self): return (self.log_amp, self.log_length), ()
    @classmethod
    def tree_unflatten(cls, aux, children):
        log_amp, log_length = children
        return cls(log_amp=log_amp, log_length=log_length)

@jax.tree_util.register_pytree_node_class
@dataclass
class Matern32:
    log_amp: jnp.ndarray
    log_length: jnp.ndarray
    def __call__(self, X: jnp.ndarray, Z: jnp.ndarray) -> jnp.ndarray:
        amp = softplus(self.log_amp)
        ell = softplus(self.log_length)
        diff = (X[:, None, :] - Z[None, :, :]) / ell
        r = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-16)
        sqrt3_r = jnp.sqrt(3.0) * r
        return (amp ** 2) * (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r)
    def tree_flatten(self): return (self.log_amp, self.log_length), ()
    @classmethod
    def tree_unflatten(cls, aux, children):
        log_amp, log_length = children
        return cls(log_amp=log_amp, log_length=log_length)

# -------------------------------
# Likelihood
# -------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass
class GaussianLikelihood:
    log_noise: Optional[jnp.ndarray]   # None => fixed
    fixed_noise: Optional[float] = None
    def noise_var(self) -> jnp.ndarray:
        if self.log_noise is None:
            return jnp.array(self.fixed_noise if self.fixed_noise is not None else 1e-3, dtype=jnp.float32)
        # lower-bound noise ~ 1e-6 via softplus base
        return softplus(self.log_noise) + 1e-6
    def tree_flatten(self):
        if self.log_noise is None:
            return (), ("fixed", float(self.fixed_noise) if self.fixed_noise is not None else 1e-3)
        else:
            return (self.log_noise,), ("learnable", None)
    @classmethod
    def tree_unflatten(cls, aux, children):
        mode, fixed = aux
        if mode == "fixed":
            return cls(log_noise=None, fixed_noise=fixed)
        (log_noise,) = children
        return cls(log_noise=log_noise, fixed_noise=None)

# -------------------------------
# Params / Prior / Posterior
# -------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass
class GPParams:
    kernel: Union[RBF, Matern32]
    likelihood: GaussianLikelihood
    mean_fn: ZeroMean
    def tree_flatten(self):
        return (self.kernel, self.likelihood), (self.mean_fn,)
    @classmethod
    def tree_unflatten(cls, aux, children):
        (mean_fn,) = aux
        kernel, likelihood = children
        return cls(kernel=kernel, likelihood=likelihood, mean_fn=mean_fn)

@dataclass
class Prior:
    X: jnp.ndarray
    params: GPParams
    def K(self) -> jnp.ndarray:
        K = self.params.kernel(self.X, self.X)
        noise = self.params.likelihood.noise_var()
        return K + noise * jnp.eye(self.X.shape[0], dtype=self.X.dtype)
    def m(self) -> jnp.ndarray:
        return self.params.mean_fn(self.X)

@dataclass
class Posterior:
    X: jnp.ndarray
    y: jnp.ndarray
    params: GPParams
    L: jnp.ndarray
    alpha: jnp.ndarray
    @classmethod
    def from_prior(cls, prior: Prior, y: jnp.ndarray, jitter: float = 1e-6) -> "Posterior":
        K = prior.K()
        m = prior.m()
        y_c = y - m
        L = _stable_cholesky(K, base_jitter=jitter)
        alpha = jsp.linalg.cho_solve((L, True), y_c)
        return cls(X=prior.X, y=y, params=prior.params, L=L, alpha=alpha)
    def predict(self, Xnew: jnp.ndarray, include_noise: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        k_star = self.params.kernel(Xnew, self.X)
        m_star = self.params.mean_fn(Xnew)
        v = jsp.linalg.cho_solve((self.L, True), k_star.T)
        mean = m_star + k_star @ self.alpha
        k_ss = self.params.kernel(Xnew, Xnew)
        cov_lat = k_ss - k_star @ v
        var_lat = jnp.clip(jnp.diag(cov_lat), 0.0)
        var = var_lat + self.params.likelihood.noise_var() if include_noise else var_lat
        return mean, jnp.sqrt(jnp.maximum(var, 0.0))


# marginal likelihood

def negative_log_marginal_likelihood(
    params: GPParams, X: jnp.ndarray, y: jnp.ndarray, jitter: float = 1e-6
) -> jnp.ndarray:
    K = params.kernel(X, X) + params.likelihood.noise_var() * jnp.eye(X.shape[0], dtype=X.dtype)
    m = params.mean_fn(X)
    y_c = y - m

    any_bad = (~jnp.isfinite(K)).any() | (~jnp.isfinite(y_c)).any()

    def penalized(_):
        return jnp.array(1e12, dtype=jnp.float32)

    def ok(_):
        L = _stable_cholesky(K, base_jitter=jitter)
        alpha = jsp.linalg.cho_solve((L, True), y_c)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.clip(jnp.diag(L), 1e-30)))
        nll = 0.5 * (y_c @ alpha) + 0.5 * logdet + 0.5 * X.shape[0] * jnp.log(2.0 * jnp.pi)

        # ---- Weak priors (on standardized X, y) ----
        amp = softplus(params.kernel.log_amp)
        ell = softplus(params.kernel.log_length)
        log_ell = jnp.log(ell)
        reg_amp = 1e-3 * (jnp.log(amp) - 0.0) ** 2
        reg_len = 1e-3 * jnp.sum((log_ell - 0.0) ** 2)

        if params.likelihood.log_noise is not None:
            sig2 = softplus(params.likelihood.log_noise) + 1e-6
            reg_noise = 5e-4 * (jnp.log(sig2) - jnp.log(1e-2)) ** 2
        else:
            reg_noise = 0.0

        reg_l2 = 1e-5 * (
            jnp.sum(params.kernel.log_length**2) + params.kernel.log_amp**2 +
            (0.0 if params.likelihood.log_noise is None else params.likelihood.log_noise**2)
        )
        
###------------------------------------------
## TO HAVE PURE MLE only return nll abnd commnet out the rest of the paramters
#########
        return nll + reg_amp + reg_len + reg_noise + reg_l2

    return jax.lax.cond(any_bad, penalized, ok, operand=None)

# -------------------------------
# Training
# -------------------------------

def _init_params(
    dim: int,
    kernel_type: str = "matern32",
    fixed_noise: Optional[float] = None,   # None => learn noise
    seed: int = 0,
) -> GPParams:
    # Start at amp ≈ 1, ℓ ≈ 1 on standardized X
    log_amp = jnp.array(np.log(np.expm1(1.0)), dtype=jnp.float32)
    log_length = jnp.array(np.log(np.expm1(1.0)) * np.ones(dim), dtype=jnp.float32)

    kernel = RBF(log_amp, log_length) if kernel_type.lower()=="rbf" else Matern32(log_amp, log_length)

    if fixed_noise is None:
        # learnable noise initialized near 1e-2
        log_noise = jnp.array(np.log(np.expm1(1e-2)), dtype=jnp.float32)
        like = GaussianLikelihood(log_noise=log_noise, fixed_noise=None)
    else:
        like = GaussianLikelihood(log_noise=None, fixed_noise=float(fixed_noise))

    return GPParams(kernel=kernel, likelihood=like, mean_fn=ZeroMean())

def train_lbfgs(
    X: jnp.ndarray,
    y: jnp.ndarray,
    kernel: str = "matern32",
    fixed_noise: Optional[float] = None,  # None => learn noise
    lbfgs_max_iter: int = 300,
    lbfgs_tol: float = 1e-7,
    num_restarts: int = 3,
    seed: int = 0,
    jitter: float = 1e-6,
) -> Tuple[GPParams, float]:
    best_params: Optional[GPParams] = None
    best_val = np.inf

    for r in range(num_restarts):
        init = _init_params(X.shape[1], kernel_type=kernel, fixed_noise=fixed_noise, seed=seed + r)
        def obj(p: GPParams):
            return negative_log_marginal_likelihood(p, X, y, jitter=jitter)
        solver = LBFGS(fun=obj, maxiter=lbfgs_max_iter, tol=lbfgs_tol, linesearch="backtracking")
        res = solver.run(init)
        try:
            val = float(res.state.value)
        except Exception:
            val = float(obj(res.params))
        if val < best_val:
            best_val, best_params = val, res.params
        print(f"[restart {r}] nLML = {val:.6f}")

    if best_params is None:
        raise RuntimeError("All LBFGS restarts failed.")
    print(f"[ok] Best nLML = {best_val:.6f}")
    return best_params, best_val

# -------------------------------
# Standardization
# -------------------------------

@dataclass
class Standardizer:
    x_mean: Optional[jnp.ndarray] = None
    x_std: Optional[jnp.ndarray] = None
    y_mean: Optional[float] = None
    y_std: Optional[float] = None
    def fit(self, X: jnp.ndarray, y: jnp.ndarray, standardize_x: bool = True, standardize_y: bool = True):
        if standardize_x:
            xm = jnp.mean(X, axis=0)
            xs = jnp.std(X, axis=0)
            xs = jnp.where(xs > 0, xs, 1.0)
            self.x_mean, self.x_std = xm, xs
        if standardize_y:
            ym = jnp.mean(y); ys = jnp.std(y)
            ys = jnp.where(ys > 0, ys, 1.0)
            self.y_mean, self.y_std = float(ym), float(ys)
        return self
    def transform_X(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.x_mean is None: return X
        return (X - self.x_mean) / self.x_std
    def transform_y(self, y: jnp.ndarray) -> jnp.ndarray:
        if self.y_mean is None: return y
        return (y - self.y_mean) / self.y_std
    def inverse_y(self, y: jnp.ndarray, std: Optional[jnp.ndarray] = None):
        if self.y_mean is None:
            return (y, std) if std is not None else y
        y_ = y * self.y_std + self.y_mean
        if std is None: return y_
        return y_, std * self.y_std

# -------------------------------
# CSV loader (raw only — no transforms)
# -------------------------------

def load_xy_from_csv(train_csv: Union[str, Path], test_csv: Union[str, Path], dim: int):
    """Load raw X, y from CSVs. Drop rows containing NaNs in any used column."""
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)

    cols = [f"x{i+1}" for i in range(dim)]
    if "y" not in tr.columns:
        raise ValueError("Training CSV must contain a 'y' column.")

    # Ensure all required columns exist
    for c in cols:
        if c not in tr.columns or c not in te.columns:
            raise ValueError(f"Missing column {c} in CSVs.")

    # Columns that must be non-NaN
    req_cols_tr = cols + ["y"]
    req_cols_te = cols + (["y"] if "y" in te.columns else [])

    # Drop rows with ANY NaN in required columns
    tr = tr.dropna(subset=req_cols_tr)
    te = te.dropna(subset=req_cols_te)

    # Convert to jax arrays
    Xtr = jnp.asarray(tr[cols].values.astype(np.float32))
    ytr = jnp.asarray(tr["y"].values.astype(np.float32))

    Xte = jnp.asarray(te[cols].values.astype(np.float32))
    yte = jnp.asarray(te["y"].values.astype(np.float32)) if "y" in te.columns else None

    return Xtr, ytr, Xte, yte


    
def _kernel_to_state(k):
    if isinstance(k, RBF):
        return {"type": "rbf",
                "log_amp": np.asarray(k.log_amp),
                "log_length": np.asarray(k.log_length)}
    elif isinstance(k, Matern32):
        return {"type": "matern32",
                "log_amp": np.asarray(k.log_amp),
                "log_length": np.asarray(k.log_length)}
    else:
        raise TypeError(f"Unknown kernel class: {type(k)}")

def _kernel_from_state(target, state):
     t = state["type"]
     log_amp = jnp.asarray(state["log_amp"])
     log_length = jnp.asarray(state["log_length"])
     if t == "rbf":
         return RBF(log_amp, log_length)
     elif t == "matern32":
         return Matern32(log_amp, log_length)
     else:
         raise ValueError(f"Unknown kernel type: {t}")

def _lik_to_state(like: GaussianLikelihood):
    if like.log_noise is None:
        return {"mode": "fixed", "fixed_noise": float(like.fixed_noise if like.fixed_noise is not None else 1e-3)}
    else:
        return {"mode": "learnable", "log_noise": np.asarray(like.log_noise)}


def _lik_from_state(target, state):
     if state["mode"] == "fixed":
         return GaussianLikelihood(log_noise=None, fixed_noise=float(state["fixed_noise"]))
     else:
         return GaussianLikelihood(log_noise=jnp.asarray(state["log_noise"]), fixed_noise=None)


def _zeromean_to_state(m: ZeroMean):
    return {"type": "zero"}

def _zeromean_from_state(target, state):
     return ZeroMean()

def _gpparams_to_state(p: GPParams):
    return {
        "kernel": _kernel_to_state(p.kernel),
        "likelihood": _lik_to_state(p.likelihood),
        "mean": _zeromean_to_state(p.mean_fn),
    }

def _gpparams_from_state(target, state):
     return GPParams(
         kernel=_kernel_from_state(None, state["kernel"]),
         likelihood=_lik_from_state(None, state["likelihood"]),
         mean_fn=_zeromean_from_state(None, state["mean"]),
     )

def _std_to_state(s: Standardizer):
    return {
        "x_mean": None if s.x_mean is None else np.asarray(s.x_mean),
        "x_std":  None if s.x_std  is None else np.asarray(s.x_std),
        "y_mean": s.y_mean,
        "y_std":  s.y_std,
    }

def _std_from_state(target, state):
     return Standardizer(
         x_mean=None if state["x_mean"] is None else jnp.asarray(state["x_mean"]),
         x_std=None  if state["x_std"]  is None else jnp.asarray(state["x_std"]),
         y_mean=state["y_mean"],
         y_std=state["y_std"],
     )



# -------------------------------
# -------------------------------
# End-to-end class
# -------------------------------

@dataclass
class MiniGPJax:
    dim: int
    kernel: str = "matern32"
    fixed_noise: Optional[float] = None   # None => learn noise by default
    standardize_x: bool = True
    standardize_y: bool = True
    log_y: bool = False
    jitter: float = 1e-6

    params: Optional[GPParams] = None
    std: Optional[Standardizer] = None
    X_train: Optional[jnp.ndarray] = None
    y_train: Optional[jnp.ndarray] = None
    X_test: Optional[jnp.ndarray] = None
    y_test: Optional[jnp.ndarray] = None
    posterior: Optional[Posterior] = None

    def load(self, train_csv: Union[str, Path], test_csv: Union[str, Path]):
        # 1) Load RAW data (no transforms)
        Xtr_raw, ytr_raw, Xte_raw, yte_raw = load_xy_from_csv(train_csv, test_csv, self.dim)

        # 2) LOG transform (if requested) BEFORE standardization
        if self.log_y:
            if jnp.any(ytr_raw <= 0) or (yte_raw is not None and jnp.any(yte_raw <= 0)):
                raise ValueError("log_y=True but some targets are <= 0.")
            ytr = jnp.log(ytr_raw)
            yte = jnp.log(yte_raw) if yte_raw is not None else None
        else:
            ytr, yte = ytr_raw, yte_raw

        # 3) Fit standardizer on TRAIN ONLY (log-space y if log_y=True)
        self.std = Standardizer().fit(Xtr_raw, ytr, self.standardize_x, self.standardize_y)

        # 4) Transform X and y using train stats
        self.X_train = self.std.transform_X(Xtr_raw)
        self.y_train = self.std.transform_y(ytr)
        self.X_test  = self.std.transform_X(Xte_raw)
        self.y_test  = self.std.transform_y(yte) if yte is not None else None

        print(f"[ok] Loaded train: X={self.X_train.shape}, y={self.y_train.shape}{' (log→std)' if self.log_y else ' (std)'}")
        print(f"[ok] Loaded test:  X={self.X_test.shape}, y={'missing' if self.y_test is None else str(self.y_test.shape) + (' (log→std)' if self.log_y else ' (std)')}")

    def fit(self, lbfgs_max_iter: int = 300, lbfgs_tol: float = 1e-7, num_restarts: int = 3, seed: int = 0):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Call load() first.")
        params, nll = train_lbfgs(
            self.X_train, self.y_train,
            kernel=self.kernel,
            fixed_noise=self.fixed_noise,
            lbfgs_max_iter=lbfgs_max_iter,
            lbfgs_tol=lbfgs_tol,
            num_restarts=num_restarts,
            seed=seed,
            jitter=self.jitter,
        )
        self.params = params
        # Diagnostics
        amp = float(softplus(self.params.kernel.log_amp))
        ell = np.asarray(softplus(self.params.kernel.log_length))
        noise = float(self.params.likelihood.noise_var())
        print(f"[fit] amp={amp:.4g} | noise_var={noise:.4g} | ell={np.round(ell,3)}")

        # Train fit quick check (z-space)
        prior = Prior(self.X_train, self.params)
        self.posterior = Posterior.from_prior(prior, self.y_train, jitter=self.jitter)
        mtr, _ = self.posterior.predict(self.X_train, include_noise=True)
        ytr_true = self.y_train
        tr_resid = np.asarray(mtr) - np.asarray(ytr_true)
        tr_rmse = float(np.sqrt(np.mean(tr_resid**2)))
        tr_r2 = float(1.0 - np.sum(tr_resid**2) / (np.sum((np.asarray(ytr_true) - float(np.mean(ytr_true)))**2) + 1e-12))
        print(f"[fit] train RMSE(z)={tr_rmse:.4f} | R²(z)={tr_r2:.4f}  (z = standardized)")
    
    def predict(self, Xnew: Union[np.ndarray, jnp.ndarray], include_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Predict in original y units. If log_y=True, uses log-normal moment correction."""
        if self.posterior is None:
            raise RuntimeError("Call fit() first.")
        Xn = jnp.asarray(Xnew, dtype=jnp.float32)
        Xn = self.std.transform_X(Xn)

        # Posterior in z-space (standardized log(y) if log_y=True; else standardized y)
        mean_z, std_z = self.posterior.predict(Xn, include_noise=include_noise)
        # Undo standardization (still in log space if log_y=True)
        mean_z, std_z = self.std.inverse_y(mean_z, std_z)

        if self.log_y:
            # Z ~ N(mu, sigma^2) ; Y = exp(Z)
            mu    = mean_z
            sigma = std_z
            y_mean = jnp.exp(mu + 0.5 * (sigma ** 2))                                  # E[Y]
            y_std  = jnp.sqrt((jnp.exp(sigma ** 2) - 1.0) * jnp.exp(2 * mu + sigma**2)) # std(Y)
            mean, std = y_mean, y_std
        else:
            mean, std = mean_z, std_z

        return np.asarray(mean), np.asarray(std)

    def predict_log(self, Xnew: Union[np.ndarray, jnp.ndarray], include_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict in natural log units. Only meaningful if log_y=True.
        Returns (mean_logy, std_logy). No exp back-transform.
        """
        if not self.log_y:
            raise RuntimeError("predict_log() requires MiniGPJax(log_y=True).")
        if self.posterior is None:
            raise RuntimeError("Call fit() first.")
        Xn = jnp.asarray(Xnew, dtype=jnp.float32)
        #Xn = self.std.transform_X(Xn)

        # Posterior in training z-space (standardized). Include obs noise if requested.
        mu_z, std_z = self.posterior.predict(Xn, include_noise=include_noise)
        # Undo standardization -> natural log units
        mu_log, std_log = self.std.inverse_y(mu_z, std_z)
        return np.asarray(mu_log), np.asarray(std_log)

    def _predict_latent_z(self, Xnew: Union[np.ndarray, jnp.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mu_lat_z, std_lat_z) in TRAINING SPACE z (standardized log(y) if log_y=True,
        or standardized y if log_y=False)."""
        if self.posterior is None:
            raise RuntimeError("Call fit() first.")
        Xn = jnp.asarray(Xnew, dtype=jnp.float32)
        Xn = self.std.transform_X(Xn)
        mu_lat_z, std_lat_z = self.posterior.predict(Xn, include_noise=False)
        return np.asarray(mu_lat_z), np.asarray(std_lat_z)

    def evaluate_and_plot(self, outdir: Union[str, Path] = "gp_outputs", metric_space: str = "original") -> Dict[str, float]:
        """
        metric_space:
          - "original": metrics/plots in original y units
          - "log": metrics/plots in natural log units (log(y))  [only if log_y=True]
          - "logz": metrics in standardized training space z     [always available]
        """
        # Prefer evaluating in log space by default when trained with log targets
        if self.log_y and (metric_space == "original"):
            metric_space = "log"

        out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
        metrics: Dict[str, float] = {}

        # --- z-space metrics (always available) ---
        if metric_space.lower() == "logz":
            mu_z, _ = self._predict_latent_z(self.X_test)          # z-pred
            y_true_z = np.asarray(self.y_test)                      # z-true (already standardized in load)
            resid = mu_z - y_true_z
            rmse = float(np.sqrt(np.mean(resid**2)))
            denom = float(np.sum((y_true_z - y_true_z.mean())**2))
            r2 = float("nan") if denom < 1e-20 else float(1.0 - np.sum(resid**2) / denom)
            mae = float(np.mean(np.abs(resid)))
            metrics.update({"rmse(z)": rmse, "mae(z)": mae, "r2(z)": r2})
            print(f"[eval:z] RMSE(z)={rmse:.4f} | MAE(z)={mae:.4f} | R²(z)={r2:.4f}")

            plt.figure(figsize=(6,6))
            lo = float(min(y_true_z.min(), mu_z.min())); hi = float(max(y_true_z.max(), mu_z.max()))
            plt.plot([lo,hi], [lo,hi], "k--", lw=1, label="ideal")
            plt.plot(y_true_z, mu_z, "o", ms=4, alpha=0.7, label="pred")
            plt.xlabel("z (true)"); plt.ylabel("z (pred)")
            plt.title("Predictions vs Truth (z-space)")
            plt.legend(); plt.tight_layout(); plt.savefig(out / "scatter_pred_vs_true_z.pdf"); plt.close()
            return metrics

        # --- log-space metrics (preferred when log_y=True) ---
        if metric_space.lower() == "log":
            if not self.log_y:
                raise ValueError("metric_space='log' requires MiniGPJax(log_y=True).")

            mu_log, _ = self.predict_log(self.X_test, include_noise=True)
            y_true_log = np.asarray(self.std.inverse_y(self.y_test))  # undo standardization -> log(y)

            resid = mu_log - y_true_log
            rmse = float(np.sqrt(np.mean(resid**2)))
            mae  = float(np.mean(np.abs(resid)))
            denom = float(np.sum((y_true_log - y_true_log.mean())**2))
            r2 = float("nan") if denom < 1e-20 else float(1.0 - np.sum(resid**2) / denom)
            metrics.update({"rmse_log": rmse, "mae_log": mae, "r2_log": r2})
            print(f"[eval:log] RMSE(log)={rmse:.6f} | MAE(log)={mae:.6f} | R²(log)={r2:.6f}")

            # Plot in log units
            plt.figure(figsize=(6,6))
            lo = float(min(y_true_log.min(), mu_log.min())); hi = float(max(y_true_log.max(), mu_log.max()))
            plt.plot([lo,hi], [lo,hi], "k--", lw=1, label="ideal")
            plt.plot(y_true_log, mu_log, "o", ms=4, alpha=0.7, label="pred")
            plt.xlabel("log(y) true"); plt.ylabel("log(y) pred")
            plt.title("Predictions vs Truth (log space)")
            plt.legend(); plt.tight_layout(); plt.savefig(out / "scatter_pred_vs_true_log.pdf"); plt.close()

            # Dump a CSV in log space
            df = pd.DataFrame(self.X_test, columns=[f"x{i+1}" for i in range(self.dim)])
            df["y_true_log"] = y_true_log
            df["y_pred_log"] = mu_log
            df.to_csv(out / "test_predictions_log.csv", index=False)
            
            return metrics

        # --- default: original space (only if explicitly requested or log_y=False) ---
        y_pred, y_std = self.predict(self.X_test, include_noise=True)

        if self.y_test is not None:
            # Build y_true in original units regardless of log_y
            y_true_base = np.asarray(self.std.inverse_y(self.y_test))
            y_true = np.exp(y_true_base) if self.log_y else y_true_base

            resid = y_pred - y_true
            rmse = float(np.sqrt(np.mean(resid**2)))
            mae  = float(np.mean(np.abs(resid)))
            denom = float(np.sum((y_true - y_true.mean())**2))
            r2 = float("nan") if denom < 1e-20 else float(1.0 - np.sum(resid**2) / denom)
            metrics.update({"rmse": rmse, "mae": mae, "r2": r2})
            print(f"[eval] RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")
        else:
            print("[eval] Test set has no 'y'; skipping metrics and scatter.")

        # Scatter in original space
        if self.y_test is not None:
            plt.figure(figsize=(6,6))
            lo = float(min(y_true.min(), y_pred.min())); hi = float(max(y_true.max(), y_pred.max()))
            plt.plot([lo,hi], [lo,hi], "k--", lw=1, label="ideal")
            plt.errorbar(y_true, y_pred, yerr=1.96*y_std, fmt="o", ms=4, alpha=0.7, label="pred ± 95% CI")
            plt.xlabel("y (true)"); plt.ylabel("y (pred)")
            plt.title("Predictions vs Truth (original space)")
            plt.legend(); plt.tight_layout(); plt.savefig(out / "scatter_pred_vs_true.pdf"); plt.close()

        # Uncertainty vs index (original-space std)
        plt.figure(figsize=(7,4))
        plt.plot(y_std, lw=1.8)
        plt.xlabel("test index"); plt.ylabel("pred std")
        plt.title("Predictive Uncertainty (std)")
        plt.tight_layout(); plt.savefig(out / "pred_uncertainty_vs_index.pdf"); plt.close()

        # Dump predictions in the requested evaluation space
        df = pd.DataFrame(self.X_test, columns=[f"x{i+1}" for i in range(self.dim)])
        df["y_pred"], df["y_std"] = (y_pred if 'y_pred' in locals() else np.nan), (y_std if 'y_std' in locals() else np.nan)
        if self.y_test is not None:
            df["y_true"] = y_true if 'y_true' in locals() else np.nan
        df.to_csv(out / "test_predictions.csv", index=False)
        return metrics
    
    def save_checkpoint(self, ckpt_dir: Union[str, Path]) -> None:
        """
        Saves:
          - config.json
          - params.msgpack     (GPParams)
          - standardizer.msgpack
          - (optional) You can also save X_train / y_train if desired.
        """
        _register_all_serializers_once()
        if self.params is None or self.std is None:
            raise RuntimeError("Nothing to save: fit() and load() must have run so params and std exist.")
        out = Path(ckpt_dir); out.mkdir(parents=True, exist_ok=True)

        cfg = {
            "dim":            int(self.dim),
            "kernel":         str(self.kernel),
            "fixed_noise":    (None if self.fixed_noise is None else float(self.fixed_noise)),
            "standardize_x":  bool(self.standardize_x),
            "standardize_y":  bool(self.standardize_y),
            "log_y":          bool(self.log_y),
            "jitter":         float(self.jitter),
        }
        (out / "config.json").write_text(json.dumps(cfg, indent=2))

        (out / "params.msgpack").write_bytes(sz.to_bytes(self.params))
        (out / "standardizer.msgpack").write_bytes(sz.to_bytes(self.std))

        # Optional: persist train arrays to reconstruct posterior later without reloading CSVs
        # np.save(out / "X_train.npy", np.asarray(self.X_train) if self.X_train is not None else None)
        # np.save(out / "y_train.npy", np.asarray(self.y_train) if self.y_train is not None else None)

        print(f"[save] Wrote checkpoint to: {out.resolve()}")

    @classmethod
    def load_checkpoint(cls, ckpt_dir: Union[str, Path]) -> "MiniGPJax":
        """
        Loads a model with GP hyperparameters and Standardizer.
        Posterior is NOT auto-rebuilt because it depends on X_train/y_train in memory.
        After calling this, either:
          - call `model.load(train_csv, test_csv)` then `model.rebuild_posterior()`, or
          - set `model.X_train`, `model.y_train` (or load from .npy) and call `rebuild_posterior()`.
        """
        _register_all_serializers_once()
        src = Path(ckpt_dir)
        cfg = json.loads((src / "config.json").read_text())
        model = cls(
            dim=cfg["dim"],
            kernel=cfg["kernel"],
            fixed_noise=cfg["fixed_noise"],
            standardize_x=cfg["standardize_x"],
            standardize_y=cfg["standardize_y"],
            log_y=cfg["log_y"],
            jitter=cfg["jitter"],
        )

        tmpl_params = _init_params(dim=cfg["dim"], kernel_type=cfg["kernel"], fixed_noise=cfg["fixed_noise"])
        model.params = sz.from_bytes(tmpl_params, (src / "params.msgpack").read_bytes())
        model.std    = sz.from_bytes(Standardizer(), (src / "standardizer.msgpack").read_bytes())

        # Optional: if you saved these
        # xt_path, yt_path = src / "X_train.npy", src / "y_train.npy"
        # if xt_path.exists() and yt_path.exists():
        #     model.X_train = jnp.asarray(np.load(xt_path, allow_pickle=True))
        #     model.y_train = jnp.asarray(np.load(yt_path, allow_pickle=True))

        print(f"[load] Loaded config+params from: {src.resolve()}")
        return model

    def rebuild_posterior(self) -> None:
        if self.params is None:
            raise RuntimeError("params is None; load checkpoint or fit() first.")
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("X_train/y_train are missing; call load(...), or set them, then rebuild.")
        prior = Prior(self.X_train, self.params)
        self.posterior = Posterior.from_prior(prior, self.y_train, jitter=self.jitter)
        print("[posterior] Rebuilt training posterior.")

    def attach_training_from_csv(self, train_csv: Union[str, Path]) -> None:
        """
        Read RAW training CSV, apply log transform if self.log_y, and then
        apply the *loaded* standardizer to populate X_train / y_train.
        """
        if self.std is None:
            raise RuntimeError("Standardizer is missing. Load checkpoint first (or call load(...)).")

        tr = pd.read_csv(train_csv)
        cols = [f"x{i+1}" for i in range(self.dim)]
        if "y" not in tr.columns:
            raise ValueError("Training CSV must contain a 'y' column.")

        Xtr_raw = jnp.asarray(tr[cols].values.astype(np.float32))
        ytr_raw = jnp.asarray(tr["y"].values.astype(np.float32))

        if self.log_y:
            if jnp.any(ytr_raw <= 0):
                raise ValueError("Model expects log_y=True but training y has non-positive values.")
            ytr = jnp.log(ytr_raw)
        else:
            ytr = ytr_raw

        # IMPORTANT: use the *loaded* standardizer stats
        self.X_train = self.std.transform_X(Xtr_raw)
        self.y_train = self.std.transform_y(ytr)

    def attach_test_from_csv(self, test_csv: Union[str, Path]) -> None:
        """
        Read RAW test CSV and populate X_test / y_test using the *loaded* standardizer.
        Works whether 'y' exists in test CSV or not.
        """
        if self.std is None:
            raise RuntimeError("Standardizer is missing. Load checkpoint first (or call load(...)).")

        te = pd.read_csv(test_csv)
        cols = [f"x{i+1}" for i in range(self.dim)]
        Xte_raw = jnp.asarray(te[cols].values.astype(np.float32))
        yte = jnp.asarray(te["y"].values.astype(np.float32)) if "y" in te.columns else None

        self.X_test = self.std.transform_X(Xte_raw)
        if yte is None:
            self.y_test = None
        else:
            if self.log_y:
                if jnp.any(yte <= 0):
                    raise ValueError("Model expects log_y=True but test y has non-positive values.")
                self.y_test = self.std.transform_y(jnp.log(yte))
            else:
                self.y_test = self.std.transform_y(yte)

    @classmethod
    def ready_from_checkpoint(
        cls,
        ckpt_dir: Union[str, Path],
        train_csv: Optional[Union[str, Path]] = None,
        test_csv: Optional[Union[str, Path]] = None,
        rebuild: bool = True,
    ) -> "MiniGPJax":
        """
        One-call loader that returns a *ready-to-use* GP.

        Steps:
          1) load checkpoint (config + GP params + standardizer)
          2) if train_csv is provided: attach training design with loaded std
          3) if rebuild=True and training is present: rebuild posterior
          4) if test_csv is provided: attach test design
        """
        _register_all_serializers_once()
        model = cls.load_checkpoint(ckpt_dir)

        if train_csv is not None:
            model.attach_training_from_csv(train_csv)
            if rebuild:
                model.rebuild_posterior()

        if test_csv is not None:
            model.attach_test_from_csv(test_csv)

        return model
# -------------------------------
# Demo
# -------------------------------

def _demo():
    TRAIN_CSV = "11_13_25_train_set.csv"
    TEST_CSV  = "11_13_25_test_set.csv"
    dim = 5
    gp = MiniGPJax(
        dim=dim,
        kernel="matern32",            # or "matern32"
        fixed_noise=None,        # consider None to learn noise for better generalization
        standardize_x=True,
        standardize_y=True,
        log_y=True,              # ensures: log -> standardize (train stats) -> fit
        jitter=1e-6,
    )
    gp.load(TRAIN_CSV, TEST_CSV)
    gp.fit(lbfgs_max_iter=1, lbfgs_tol=1e-7, num_restarts=2, seed=42)
    gp.evaluate_and_plot(outdir="11_16")  # defaults to "log" when log_y=True

#if __name__ == "__main__":
#    _demo()

    
    
    

def main():
    # Instantiate and run creating data\
        
    flow = BayesianFLow(dim=5, n_train_samples=1024, n_test_samples=256)
    flow.create_training_set()
    flow.create_test_set()


if __name__ == "__main__":
    main()