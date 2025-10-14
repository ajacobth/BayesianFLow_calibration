import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")  # keep non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)

# -----------------------------
# ARD squared exponential (RBF) kernel WITHOUT branching
def kernel_base(X, Z, var, length):
    """
    ARD RBF kernel (no noise/jitter added here).
    X: (N, D)
    Z: (M, D)
    var: scalar
    length: (D,) or scalar
    returns: (N, M)
    """
    length = jnp.atleast_1d(length)
    diff = (X[:, None, :] - Z[None, :, :]) / length     # (N, M, D)
    sq = jnp.sum(diff**2, axis=-1)                      # (N, M)
    return var * jnp.exp(-0.5 * sq)


def model(X, Y, jitter=1e-6):
    """
    GP with ARD lengthscales:
      y ~ N(0, K(X,X) + sigma^2 I)
      var ~ LogNormal(0, 10)
      length ~ LogNormal(0, 10) per-dimension
      noise ~ LogNormal(0, 10)
    """
    D = X.shape[1]
    var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
    noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
    length = numpyro.sample("kernel_length", dist.LogNormal(jnp.zeros(D), 10.0 * jnp.ones(D)))

    K_xx = kernel_base(X, X, var, length)
    # Add diagonal noise ONLY for K_xx here
    K_xx = K_xx + (noise + jitter) * jnp.eye(X.shape[0])

    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=K_xx),
        obs=Y,
    )


def run_inference(model, args, rng_key, X, Y):
    start = time.time()
    # init strategies
    if args.init_strategy == "value":
        init_strategy = init_to_value(
            values={
                "kernel_var": 1.0,
                "kernel_noise": 0.05,
                "kernel_length": jnp.ones(X.shape[1]) * 0.5,
            }
        )
    elif args.init_strategy == "median":
        init_strategy = init_to_median(num_samples=10)
    elif args.init_strategy == "feasible":
        init_strategy = init_to_feasible()
    elif args.init_strategy == "sample":
        init_strategy = init_to_sample()
    elif args.init_strategy == "uniform":
        init_strategy = init_to_uniform(radius=1)
    else:
        init_strategy = init_to_median(num_samples=10)

    hmc_kernel = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(
        hmc_kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thinning,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


def predict(rng_key, X, Y, X_test, var, length, noise, use_cholesky=True, jitter=1e-6):
    """
    GP posterior predictive:
      mean_* = K_*X (K_XX)^{-1} Y
      cov_*  = K_** - K_*X (K_XX)^{-1} K_X*
    Returns (mean, mean + eps) where eps ~ N(0, diag(cov_*))
    """
    k_pp = kernel_base(X_test, X_test, var, length)      # no noise
    k_pX = kernel_base(X_test, X, var, length)           # no noise
    k_XX = kernel_base(X, X, var, length) + (noise + jitter) * jnp.eye(X.shape[0])

    if use_cholesky:
        K_xx_cho = jax.scipy.linalg.cho_factor(k_XX)
        mean = jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, Y))  # (N*,)
        V = jax.scipy.linalg.cho_solve(K_xx_cho, k_pX.T)                  # (N, N*)
        cov = k_pp - jnp.matmul(k_pX, V)                                   # (N*, N*)
    else:
        K_xx_inv = jnp.linalg.inv(k_XX)
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, k_pX.T))

    std = jnp.sqrt(jnp.clip(jnp.diag(cov), 0.0))
    sigma_noise = std * jax.random.normal(rng_key, X_test.shape[:1])

    return mean, mean + sigma_noise


def get_data(N=25, D=1, sigma_obs=0.15, N_test=400):
    """
    Synthetic regression data.
    Returns:
      X: (N, D)
      Y: (N,)
      X_test: (N_test, D)
    """
    np.random.seed(0)

    # Base 1D coordinate
    x1 = jnp.linspace(-1, 1, N)
    y = x1 + 0.2 * jnp.power(x1, 3.0) + 0.5 * jnp.power(0.5 + x1, 2.0) * jnp.sin(4.0 * x1)
    y = y + sigma_obs * jnp.asarray(np.random.randn(N))
    y = (y - jnp.mean(y)) / jnp.std(y)

    if D == 1:
        X = x1[:, None]  # (N, 1)
        X_test = jnp.linspace(-1.3, 1.3, N_test)[:, None]
    else:
        # Build simple multi-D features from x1
        feats = [x1]
        for d in range(1, D):
            feats.append(jnp.sin((d + 1) * x1))
        X = jnp.stack(feats, axis=1)  # (N, D)

        x1t = jnp.linspace(-1.3, 1.3, N_test)
        feats_t = [x1t]
        for d in range(1, D):
            feats_t.append(jnp.sin((d + 1) * x1t))
        X_test = jnp.stack(feats_t, axis=1)  # (N_test, D)

    return X, y, X_test


def main(args):
    X, Y, X_test = get_data(N=args.num_data, D=args.dim)

    # Inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng_key, X, Y)

    # Prediction
    vmap_args = (
        random.split(rng_key_predict, samples["kernel_var"].shape[0]),
        samples["kernel_var"],            # (S,)
        samples["kernel_length"],         # (S, D)
        samples["kernel_noise"],          # (S,)
    )

    means, predictions = vmap(
        lambda rng_key, var, length, noise: predict(
            rng_key, X, Y, X_test, var, length, noise, use_cholesky=args.use_cholesky
        )
    )(*vmap_args)

    mean_prediction = np.mean(means, axis=0)
    percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if args.dim == 1:
        ax.plot(np.asarray(X[:, 0]), np.asarray(Y), "kx", label="train")
        ax.fill_between(
            np.asarray(X_test[:, 0]),
            percentiles[0, :],
            percentiles[1, :],
            color="lightblue",
            label="90% CI",
        )
        ax.plot(np.asarray(X_test[:, 0]), mean_prediction, "blue", ls="solid", lw=2.0, label="mean")
        ax.set(xlabel="X", ylabel="Y", title="GP (ARD) Mean predictions with 90% CI")
        ax.legend()
    else:
        # For D>1, show Y vs first feature for a quick sanity check
        ax.plot(np.asarray(X[:, 0]), np.asarray(Y), "kx", label="train (vs x1)")
        ax.plot(np.asarray(X_test[:, 0]), mean_prediction, "blue", lw=2.0, label="mean (vs x1)")
        ax.set(xlabel="x1", ylabel="Y", title=f"GP (ARD) mean vs x1 (D={args.dim})")
        ax.legend()

    plt.savefig("gp_plot.pdf")
    print("[ok] Plot saved to gp_plot.pdf")

    # ARD summary
    med_ls = np.asarray(jnp.median(samples["kernel_length"], axis=0))
    rel = 1.0 / (med_ls**2)
    print("Median lengthscales (per feature):", med_ls)
    print("Heuristic relevance ~ 1/length^2:", rel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian Process (ARD) example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--thinning", nargs="?", default=2, type=int)
    parser.add_argument("--num-data", nargs="?", default=25, type=int)
    parser.add_argument("--dim", nargs="?", default=1, type=int, help="number of input features (ARD)")
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--init-strategy",
        default="median",
        type=str,
        choices=["median", "feasible", "value", "uniform", "sample"],
    )
    parser.add_argument("--no-cholesky", dest="use_cholesky", action="store_false")
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
