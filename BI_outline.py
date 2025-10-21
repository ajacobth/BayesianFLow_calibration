#!/usr/bin/env python3
# Bayesian Simple Linear Regression (NumPyro) + full diagnostics
# Running on CPU + NUTS, with convergence diagnostics.

# ---- platform/env (CPU) ----
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"]     = "cpu"
# You can remove the following two lines on CPU to speed things up:
os.environ["JAX_DISABLE_JIT"]     = "1"
os.environ["NUMPYRO_DISABLE_JIT"] = "1"
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import jax
jax.config.update('jax_platform_name', 'cpu')
print("JAX backend:", jax.default_backend())

# -----------------------------
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.diagnostics import effective_sample_size, gelman_rubin

# -----------------------------
# Data generation
def make_data(n=100, w_true=2.0, b_true=0.5, sigma_true=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, 1)).astype(np.float32)
    f = w_true * X[:, 0] + b_true
    y = (f + sigma_true * rng.standard_normal(size=n)).astype(np.float32)
    return X, y, float(w_true), float(b_true), float(sigma_true)

# -----------------------------
# NumPyro model: y ~ Normal(w x + b, sigma)
def model(X, y=None):
    w = numpyro.sample("w", dist.Normal(0.0, 5.0))        # slope
    b = numpyro.sample("b", dist.Normal(0.0, 5.0))        # intercept
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0)) # noise std
    mu = w * X[:, 0] + b
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

# -----------------------------
# Diagnostics printer
def print_diagnostics(samples_by_chain, extra_by_chain, max_tree_depth):
    """
    Prints: R-hat, ESS (bulk/tail if available), divergence counts by chain, and tree-depth saturation.
    Works with older NumPyro versions (no `method=` argument).
    """
    # --- compatibility shim for older NumPyro ---
    def _ess(v, mode=None):
        try:
            return np.asarray(effective_sample_size(v, method=mode))
        except TypeError:
            return np.asarray(effective_sample_size(v))

    rhat = {k: np.asarray(gelman_rubin(v)) for k, v in samples_by_chain.items()}
    ess_bulk = {k: _ess(v, "bulk") for k, v in samples_by_chain.items()}
    ess_tail = {k: _ess(v, "tail") for k, v in samples_by_chain.items()}

    print("\n=== Convergence diagnostics ===")
    def _fmt(x):
        arr = np.atleast_1d(x)
        if arr.ndim == 1 and arr.size == 1:
            return f"{arr.item():.3f}"
        return np.array2string(arr, precision=3, floatmode="fixed")

    for name in ["w", "b", "sigma"]:
        if name in rhat:
            print(f"R-hat[{name}]     :", _fmt(rhat[name]))
            print(f"ESS_bulk[{name}] :", _fmt(ess_bulk[name]))
            print(f"ESS_tail[{name}] :", _fmt(ess_tail[name]))

    # Divergences per chain
    div = np.asarray(extra_by_chain["diverging"])
    div_counts = div.sum(axis=1)
    print("\nDivergences per chain:", div_counts.tolist(), f"(total={div_counts.sum()})")

    if "step_size" in extra_by_chain:
        ss = np.asarray(extra_by_chain["step_size"])
        if ss.ndim == 2:
            ss = ss[:, -1]
        print("Final step_size per chain:", np.round(ss, 6).tolist())

    if "accept_prob" in extra_by_chain:
        ap = np.asarray(extra_by_chain["accept_prob"])
        print("Mean accept_prob per chain:", np.round(ap.mean(axis=1), 3).tolist())

    if "num_steps" in extra_by_chain:
        ns = np.asarray(extra_by_chain["num_steps"])
        cap = 2 ** max_tree_depth
        sat = (ns >= cap).sum(axis=1)
        print(f"Tree depth cap hits per chain (>= 2**{max_tree_depth} = {cap} steps):", sat.tolist())


# -----------------------------
# Inference (multi-chain NUTS on CPU)
def run_inference(
    X, y,
    sampler="nuts",
    num_warmup=800,
    num_samples=1000,
    chains=4,
    seed=0,
    target_accept_prob=0.8,
    max_tree_depth=10,
    chain_method="sequential"
):
    if sampler != "nuts":
        raise ValueError("This helper is for NUTS. Use sampler='nuts'.")

    kernel = NUTS(
        model,
        init_strategy=init_to_median(),
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=chains,
        chain_method=chain_method,  # 'sequential' is safe on a single CPU device
        progress_bar=True,
    )

    mcmc.run(random.PRNGKey(seed), X, y)
    mcmc.print_summary()

    # Grouped by chain for diagnostics/inspection
    samples_by_chain = mcmc.get_samples(group_by_chain=True)  # shapes: (C, S, ...)
    extra_by_chain   = mcmc.get_extra_fields(group_by_chain=True)
    return samples_by_chain, extra_by_chain, mcmc

# -----------------------------
# Posterior predictive utilities
def from_alpha_to_z(alpha):
    if abs(alpha - 0.90) < 1e-12:
        return 1.6448536269514722
    if abs(alpha - 0.95) < 1e-12:
        return 1.959963984540054
    return float(dist.Normal(0, 1).icdf(jnp.array([(1 + alpha) / 2]))[0])

def posterior_predictive_grid(samples, X, grid=None, alpha_ci=0.90, n_lines=50, seed=0):
    if grid is None:
        x_min = float(np.min(X[:, 0])); x_max = float(np.max(X[:, 0]))
        grid = np.linspace(x_min, x_max, 300).astype(np.float32)[:, None]

    S = samples["w"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(S, size=min(n_lines, S), replace=False)

    w = np.asarray(samples["w"]); b = np.asarray(samples["b"]); sigma = np.asarray(samples["sigma"])
    means = (w[:, None] * grid[:, 0][None, :] + b[:, None])  # (S, N*)
    z = from_alpha_to_z(alpha_ci)

    mean_curve = np.mean(means, axis=0)
    std_curve = np.sqrt(np.var(means, axis=0) + np.mean(sigma**2))
    lo_curve = mean_curve - z * std_curve
    hi_curve = mean_curve + z * std_curve

    lines = []
    for s in idx:
        eps = rng.standard_normal(size=grid.shape[0]).astype(np.float32) * sigma[s]
        lines.append(means[s] + eps)
    lines = np.stack(lines, axis=0) if len(lines) else np.empty((0, grid.shape[0]))
    return grid, mean_curve, lo_curve, hi_curve, lines

def posterior_predictive_yrep(samples, X, seed=0):
    rng = np.random.default_rng(seed)
    S = samples["w"].shape[0]
    s = rng.integers(0, S)
    w, b, sigma = float(samples["w"][s]), float(samples["b"][s]), float(samples["sigma"][s])
    mu = w * X[:, 0] + b
    return (mu + sigma * rng.standard_normal(size=X.shape[0])).astype(np.float32)

# -----------------------------
# Diagnostics & plots
def plot_all(X, y, samples, alpha_ci=0.90, out_prefix="linreg_cpu"):
    print("JAX backend:", jax.default_backend())
    import os as _os
    _os.makedirs("figs", exist_ok=True)

    # 1) Data + predictive mean/CI + random y_rep lines
    grid, mean_curve, lo_curve, hi_curve, lines = posterior_predictive_grid(samples, X, alpha_ci=alpha_ci, n_lines=40)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(np.asarray(X[:,0]), np.asarray(y), s=15, c="k", alpha=0.7, label="data")
    ax.fill_between(grid[:,0], lo_curve, hi_curve, alpha=0.25, label=f"{int(alpha_ci*100)}% predictive CI")
    for k in range(lines.shape[0]):
        ax.plot(grid[:,0], lines[k], lw=0.8, alpha=0.3)
    ax.plot(grid[:,0], mean_curve, lw=2.0, label="posterior mean")
    ax.set(xlabel="x", ylabel="y", title="Posterior predictive")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"figs/{out_prefix}_predictive.pdf"); plt.close(fig)

    # 2) Trace-like plots (flattened draws)
    fig, axes = plt.subplots(3, 1, figsize=(8,6), sharex=True)
    axes[0].plot(np.asarray(samples["w"]), lw=0.8);     axes[0].set_ylabel("w")
    axes[1].plot(np.asarray(samples["b"]), lw=0.8);     axes[1].set_ylabel("b")
    axes[2].plot(np.asarray(samples["sigma"]), lw=0.8); axes[2].set_ylabel("sigma")
    axes[2].set_xlabel("draw")
    fig.suptitle("Trace (flattened)", y=0.98)
    fig.tight_layout(); fig.savefig(f"figs/{out_prefix}_trace.pdf"); plt.close(fig)

    # 3) Posterior histograms
    fig, axes = plt.subplots(1, 3, figsize=(12,3.5))
    axes[0].hist(np.asarray(samples["w"]), bins=40, alpha=0.7);     axes[0].set_title("posterior w")
    axes[1].hist(np.asarray(samples["b"]), bins=40, alpha=0.7);     axes[1].set_title("posterior b")
    axes[2].hist(np.asarray(samples["sigma"]), bins=40, alpha=0.7); axes[2].set_title("posterior sigma")
    fig.tight_layout(); fig.savefig(f"figs/{out_prefix}_posteriors.pdf"); plt.close(fig)

    # 4) Residuals vs fitted (using posterior means)
    w_hat = float(np.mean(np.asarray(samples["w"])))
    b_hat = float(np.mean(np.asarray(samples["b"])))
    mu_hat = w_hat * X[:,0] + b_hat
    resid = y - mu_hat
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(np.asarray(mu_hat), np.asarray(resid), s=15, alpha=0.7)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set(xlabel="fitted (posterior mean)", ylabel="residual", title="Residuals vs Fitted")
    fig.tight_layout(); fig.savefig(f"figs/{out_prefix}_residuals.pdf"); plt.close(fig)

    # 5) QQ-plot of residuals (vs Normal)
    r = np.sort(np.asarray(resid))
    n = r.size
    probs = (np.arange(1, n+1) - 0.5) / n
    qn = jsp.stats.norm.ppf(probs)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(qn, r, s=15, alpha=0.7)
    slope = np.std(r, ddof=1); intercept = np.mean(r)
    ax.plot(qn, intercept + slope * qn, lw=1.0)
    ax.set(xlabel="Normal quantiles", ylabel="Residual quantiles", title="QQ-plot of residuals")
    fig.tight_layout(); fig.savefig(f"figs/{out_prefix}_qq.pdf"); plt.close(fig)

    # 6) Posterior Predictive Check (hist overlay)
    y_rep = posterior_predictive_yrep(samples, X, seed=123)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(np.asarray(y), bins=30, alpha=0.5, label="observed")
    ax.hist(np.asarray(y_rep), bins=30, alpha=0.5, label="y_rep (one draw)")
    ax.set(title="Posterior Predictive Check (hist overlay)", xlabel="y", ylabel="count")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"figs/{out_prefix}_ppc_hist.pdf"); plt.close(fig)

    print("[ok] saved diagnostics to figs/*.pdf")

# -----------------------------
def main(args):
    print("JAX backend:", jax.default_backend())  # should report 'cpu'
    X, y, w_true, b_true, sigma_true = make_data(
        n=args.n, w_true=args.w_true, b_true=args.b_true, sigma_true=args.sigma_true, seed=args.seed
    )
    print(f"True params: w={w_true:.3f}, b={b_true:.3f}, sigma={sigma_true:.3f}")

    samples_by_chain, extra_by_chain, mcmc = run_inference(
        X, y,
        sampler="nuts",
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        chains=args.chains,
        seed=args.seed,
        target_accept_prob=args.target_accept_prob,
        max_tree_depth=args.max_tree_depth,
        chain_method="sequential",
    )

    # Print convergence diagnostics
    print_diagnostics(samples_by_chain, extra_by_chain, max_tree_depth=args.max_tree_depth)

    # Flatten chains for plotting (C, S, ...) -> (C*S, ...)
    samples = {k: np.reshape(np.asarray(v), (-1,) + v.shape[2:]) for k, v in samples_by_chain.items()}

    plot_all(X, y, samples, alpha_ci=args.alpha_ci, out_prefix=args.out_prefix)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bayesian Simple Linear Regression (NumPyro, CPU)")
    ap.add_argument("--n", type=int, default=100, help="number of data points")
    ap.add_argument("--w-true", type=float, default=2.0)
    ap.add_argument("--b-true", type=float, default=0.5)
    ap.add_argument("--sigma-true", type=float, default=0.3)
    ap.add_argument("--num-warmup", type=int, default=800)
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)          # multiple chains by default
    ap.add_argument("--alpha-ci", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", type=str, default="linreg_cpu")
    ap.add_argument("--target-accept_prob", type=float, default=0.8)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    args = ap.parse_args()
    main(args)
