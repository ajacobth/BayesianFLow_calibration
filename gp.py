#!/usr/bin/env python3
# Exact GP (ARD RBF) with:
#   - MAP hyperparameters via jaxopt.LBFGS (no SVI)
#   - Optional NUTS sampling over hyperparameters (NumPyro)
# Uses jax.lax.cond in _predict_impl to avoid TracerBoolConversionError.

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC, NUTS,
    init_to_median, init_to_feasible, init_to_sample, init_to_uniform, init_to_value,
)

from jaxopt import LBFGS


# -----------------------------
# Kernel
def kernel_base(X, Z, var, length):
    """
    ARD RBF kernel (no noise/jitter added).
    X: (N, D), Z: (M, D)
    var: scalar > 0
    length: (D,) or scalar > 0
    returns: (N, M)
    """
    length = jnp.atleast_1d(length)
    if length.size == 1 and X.shape[1] > 1:
        length = jnp.full((X.shape[1],), length[0])
    diff = (X[:, None, :] - Z[None, :, :]) / length
    sq = jnp.sum(diff ** 2, axis=-1)
    return var * jnp.exp(-0.5 * sq)

# -----------------------------
# NumPyro model (for MCMC mode)
def model_numpyro(X, Y, jitter=1e-6, prior_scale=0.5, prior_loc_noise=-2.3):
    D = X.shape[1]
    var    = numpyro.sample("kernel_var",   dist.LogNormal(0.0, prior_scale))
    noise  = numpyro.sample("kernel_noise", dist.LogNormal(prior_loc_noise, prior_scale))
    length = numpyro.sample("kernel_length",
                            dist.LogNormal(jnp.zeros(D), prior_scale * jnp.ones(D)))
    K = kernel_base(X, X, var, length) + (noise + jitter) * jnp.eye(X.shape[0])
    numpyro.sample("Y", dist.MultivariateNormal(jnp.zeros(X.shape[0]), K), obs=Y)

# -----------------------------
# Predictive utilities (Option 1: lax.cond)
@jax.jit
def _predict_impl(X, Y, X_test, var, length, noise, jitter, use_cholesky):
    k_pp = kernel_base(X_test, X_test, var, length)
    k_pX = kernel_base(X_test, X, var, length)
    K_xx = kernel_base(X, X, var, length) + (noise + jitter) * jnp.eye(X.shape[0])

    def cholesky_path(carry):
        K_xx, k_pX, k_pp, Y = carry
        cho   = jax.scipy.linalg.cho_factor(K_xx)
        alpha = jax.scipy.linalg.cho_solve(cho, Y)
        mean  = k_pX @ alpha
        V     = jax.scipy.linalg.cho_solve(cho, k_pX.T)
        cov   = k_pp - k_pX @ V
        std   = jnp.sqrt(jnp.clip(jnp.diag(cov), 0.0))
        return mean, std

    def inv_path(carry):
        K_xx, k_pX, k_pp, Y = carry
        K_inv = jnp.linalg.inv(K_xx)
        mean  = k_pX @ (K_inv @ Y)
        cov   = k_pp - k_pX @ (K_inv @ k_pX.T)
        std   = jnp.sqrt(jnp.clip(jnp.diag(cov), 0.0))
        return mean, std

    mean, std = lax.cond(
        use_cholesky,
        cholesky_path,
        inv_path,
        operand=(K_xx, k_pX, k_pp, Y),
    )
    return mean, std

def predict(X, Y, X_test, var, length, noise, jitter=1e-6, use_cholesky=True, alpha_ci=0.90):
    mean, std = _predict_impl(
        X, Y, X_test, var, length, noise, jitter,
        bool(use_cholesky)  # ensure a Python bool for jit-compiled lax.cond
    )
    z = 1.6448536269514722 if abs(alpha_ci - 0.90) < 1e-9 else float(
        dist.Normal(0,1).icdf(jnp.array([(1+alpha_ci)/2]))[0]
    )
    lo = mean - z * std
    hi = mean + z * std
    return mean, lo, hi

# -----------------------------
# Data
def get_data(N=25, D=1, sigma_obs=0.15, N_test=400, standardize_x=True, standardize_y=True, seed=0):
    rng = np.random.RandomState(seed)
    x1 = jnp.linspace(-1, 1, N)
    y = x1 + 0.2 * jnp.power(x1, 3.0) + 0.5 * jnp.power(0.5 + x1, 2.0) * jnp.sin(4.0 * x1)
    y = y + sigma_obs * jnp.asarray(rng.randn(N))

    if D == 1:
        X = x1[:, None]
        Xt = jnp.linspace(-1.3, 1.3, N_test)[:, None]
    else:
        feats = [x1]
        for d in range(1, D):
            feats.append(jnp.sin((d + 1) * x1))
        X = jnp.stack(feats, axis=1)
        x1t = jnp.linspace(-1.3, 1.3, N_test)
        feats_t = [x1t]
        for d in range(1, D):
            feats_t.append(jnp.sin((d + 1) * x1t))
        Xt = jnp.stack(feats_t, axis=1)

    if standardize_x:
        mu = jnp.mean(X, axis=0)
        sd = jnp.std(X, axis=0)
        X = (X - mu) / sd
        Xt = (Xt - mu) / sd

    if standardize_y:
        y = (y - jnp.mean(y)) / jnp.std(y)

    return X, y, Xt

# -----------------------------
# MAP objective (no SVI): maximize exact log posterior
def make_map_objective(X, Y, prior_scale=0.5, prior_loc_noise=-2.3, jitter=1e-6):
    N, D = X.shape

    def unpack(params):
        z_var   = params["z_var"]          # scalar
        z_len   = params["z_len"]          # (D,)
        z_noise = params["z_noise"]        # scalar
        var     = jnp.exp(z_var)
        length  = jnp.exp(z_len)
        noise   = jnp.exp(z_noise)
        return var, length, noise, z_var, z_len, z_noise

    def neg_log_posterior(params):
        var, length, noise, z_var, z_len, z_noise = unpack(params)

        # Kernel and Cholesky
        K = kernel_base(X, X, var, length) + (noise + jitter) * jnp.eye(N)
        cho = jax.scipy.linalg.cho_factor(K)
        alpha = jax.scipy.linalg.cho_solve(cho, Y)

        # Log marginal likelihood
        term1 = -0.5 * (Y @ alpha)
        term2 = -jnp.sum(jnp.log(jnp.diag(cho[0])))  # -sum(log diag(L)) = -0.5 log|K|
        term3 = -0.5 * N * jnp.log(2.0 * jnp.pi)
        log_marginal = term1 + term2 + term3

        # LogNormal priors in log-space: z ~ Normal(mu, sigma)
        def log_normal_prior(z, mu, sigma):
            return -0.5 * ((z - mu) / sigma) ** 2 - jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi)

        lp_var   = log_normal_prior(z_var,   0.0, prior_scale)
        lp_noise = log_normal_prior(z_noise, prior_loc_noise, prior_scale)
        lp_len   = jnp.sum(log_normal_prior(z_len, 0.0, prior_scale))

        log_post = log_marginal + lp_var + lp_noise + lp_len
        return -log_post  # minimize negative log posterior

    return neg_log_posterior

def run_map_lbfgs(args, X, Y):
    neg_log_post = make_map_objective(
        X, Y,
        prior_scale=args.prior_scale,
        prior_loc_noise=args.prior_loc_noise,
        jitter=args.jitter,
    )
    # Good starting point in z-space
    init_params = {
        "z_var":   jnp.array(0.0),              # log 1.0
        "z_len":   jnp.zeros((X.shape[1],)),    # log 1.0 per-dim
        "z_noise": jnp.array(-2.3),             # log ~ 0.1
    }

    solver = LBFGS(fun=neg_log_post, maxiter=args.lbfgs_max_iter, tol=args.lbfgs_tol)
    res = solver.run(init_params)
    z_var, z_len, z_noise = res.params["z_var"], res.params["z_len"], res.params["z_noise"]
    var, length, noise = jnp.exp(z_var), jnp.exp(z_len), jnp.exp(z_noise)
    return {"kernel_var": var, "kernel_length": length, "kernel_noise": noise}

# -----------------------------
# MCMC (NUTS) over hyperparameters
def run_mcmc(args, X, Y):
    def fixed_model(X_, Y_):
        return model_numpyro(
            X_, Y_,
            jitter=args.jitter,
            prior_scale=args.prior_scale,
            prior_loc_noise=args.prior_loc_noise,
        )

    # init strategy
    if args.init_strategy == "value":
        init_strategy = init_to_value(
            values={"kernel_var": 1.0, "kernel_noise": 0.1, "kernel_length": jnp.ones(X.shape[1])}
        )
    elif args.init_strategy == "median":
        init_strategy = init_to_median(num_samples=10)
    elif args.init_strategy == "feasible":
        init_strategy = init_to_feasible()
    elif args.init_strategy == "sample":
        init_strategy = init_to_sample()
    elif args.init_strategy == "uniform":
        init_strategy = init_to_uniform(radius=1.0)
    else:
        init_strategy = init_to_median(num_samples=10)

    kernel = NUTS(fixed_model, init_strategy=init_strategy, target_accept=args.target_accept)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thinning,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    rng = random.PRNGKey(args.seed)
    mcmc.run(rng, X, Y)
    mcmc.print_summary()
    return mcmc.get_samples()

# -----------------------------
def main(args):
    numpyro.set_platform(args.device)
    X, Y, X_test = get_data(
        N=args.num_data, D=args.dim, N_test=args.num_test,
        standardize_x=args.standardize_x, standardize_y=args.standardize_y, seed=args.seed
    )

    if args.inference == "map":
        t0 = time.time()
        post = run_map_lbfgs(args, X, Y)
        print(f"[MAP] finished in {time.time()-t0:.3f}s")
        var, length, noise = post["kernel_var"], post["kernel_length"], post["kernel_noise"]

        mean, lo, hi = predict(
            X, Y, X_test, var, length, noise,
            jitter=args.jitter, use_cholesky=not args.no_cholesky, alpha_ci=args.alpha_ci
        )
        mean_prediction = np.asarray(mean)
        percentiles = np.stack([np.asarray(lo), np.asarray(hi)], axis=0)
        ls_for_report = np.asarray(length)

    elif args.inference == "mcmc":
        t0 = time.time()
        samples = run_mcmc(args, X, Y)
        print(f"[MCMC] finished in {time.time()-t0:.3f}s")
        def one_pred(var, length, noise):
            m, _, _ = predict(
                X, Y, X_test, var, length, noise,
                jitter=args.jitter, use_cholesky=not args.no_cholesky, alpha_ci=args.alpha_ci
            )
            return m
        means = jax.vmap(one_pred)(
            samples["kernel_var"], samples["kernel_length"], samples["kernel_noise"]
        )  # (S, N*)
        mean_prediction = np.mean(np.asarray(means), axis=0)
        percentiles = np.percentile(np.asarray(means), [5.0, 95.0], axis=0)
        ls_for_report = np.asarray(jnp.median(samples["kernel_length"], axis=0))
    else:
        raise ValueError("Unknown --inference choice")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if args.dim == 1:
        ax.plot(np.asarray(X[:, 0]), np.asarray(Y), "kx", label="train")
        ax.fill_between(np.asarray(X_test[:, 0]), percentiles[0, :], percentiles[1, :], alpha=0.3,
                        label=f"{int(args.alpha_ci*100)}% band")
        ax.plot(np.asarray(X_test[:, 0]), mean_prediction, lw=2.0, label=f"mean ({args.inference.upper()})")
        ax.set(xlabel="X", ylabel="Y", title=f"GP ({args.inference.upper()})")
        ax.legend()
    else:
        ax.plot(np.asarray(X[:, 0]), np.asarray(Y), "kx", label="train (vs x1)")
        ax.plot(np.asarray(X_test[:, 0]), mean_prediction, lw=2.0, label=f"mean ({args.inference.upper()} vs x1)")
        ax.set(xlabel="x1", ylabel="Y", title=f"GP ({args.inference.upper()}) mean vs x1 (D={args.dim})")
        ax.legend()

    out = f"gp_{args.inference}.pdf"
    plt.savefig(out)
    print(f"[ok] Plot saved to {out}")
    print("Lengthscales (MAP or posterior median):", ls_for_report)
    print("Heuristic relevance ~ 1/length^2:", 1.0 / (ls_for_report ** 2))

# -----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exact GP (ARD) — MAP (LBFGS) or NUTS, JIT-safe predict")
    # Data
    p.add_argument("--num-data", type=int, default=25)
    p.add_argument("--num-test", type=int, default=400)
    p.add_argument("--dim", type=int, default=1)
    p.add_argument("--standardize-x", action="store_true", default=True)
    p.add_argument("--standardize-y", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)

    # Inference choice
    p.add_argument("--inference", type=str, choices=["map", "mcmc"], default="map")

    # MAP (LBFGS) controls
    p.add_argument("--lbfgs-max-iter", type=int, default=200)
    p.add_argument("--lbfgs-tol", type=float, default=1e-7)

    # MCMC controls
    p.add_argument("-n", "--num-samples", default=1000, type=int)
    p.add_argument("--num-warmup", default=800, type=int)
    p.add_argument("--num-chains", default=1, type=int)
    p.add_argument("--thinning", default=1, type=int)
    p.add_argument("--target-accept", type=float, default=0.8)
    p.add_argument("--init-strategy", default="median",
                   choices=["median", "feasible", "value", "uniform", "sample"])

    # Numerics / priors / backend
    p.add_argument("--device", type=str, default="cpu")  # "cpu" or "gpu"
    p.add_argument("--no-cholesky", action="store_true", default=False)
    p.add_argument("--jitter", type=float, default=1e-6)
    p.add_argument("--alpha-ci", type=float, default=0.90)
    p.add_argument("--prior-scale", type=float, default=0.5)
    p.add_argument("--prior-loc-noise", type=float, default=-2.3)

    args = p.parse_args()
    main(args)
