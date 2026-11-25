"Generate figures for the README."

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.bs import bs_price, bs_vega
from src.solver import implied_vol


def figure_newton_convergence() -> None:
    """Show quadratic convergence of Newton-Raphson."""
    S, K, T, r, true_sigma = 100, 100, 1.0, 0.05, 0.3
    mkt_price = bs_price(S, K, T, r, true_sigma, "call")

    sigma = 0.5
    errors = []
    for _ in range(12):
        price = bs_price(S, K, T, r, sigma, "call")
        v = bs_vega(S, K, T, r, sigma)
        err = abs(sigma - true_sigma)
        errors.append(err)
        if abs(v) < 1e-12:
            break
        sigma = sigma - (price - mkt_price) / v

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(1, len(errors) + 1), errors, "o-", markersize=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("|σ − σ_true|")
    ax.set_title("Newton-Raphson convergence (quadratic: error squares each step)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path("assets").mkdir(exist_ok=True)
    fig.savefig("assets/newton_convergence.png", dpi=120)
    print("Saved assets/newton_convergence.png")


def figure_vol_smile() -> None:
    """Recover a synthetic volatility smile from option prices."""
    S, T, r = 100, 0.5, 0.05
    strikes = np.linspace(70, 130, 25)

    true_vols = 0.20 + 0.002 * (strikes - 100) ** 2 / 100

    prices = [bs_price(S, K, T, r, sig, "call") for K, sig in zip(strikes, true_vols)]

    recovered = []
    for K, price in zip(strikes, prices):
        result = implied_vol(price, S, K, T, r, "call")
        recovered.append(result.sigma if result.converged else float("nan"))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(strikes, true_vols * 100, "o-", label="True vol (input)", markersize=6)
    ax.plot(
        strikes,
        np.array(recovered) * 100,
        "s--",
        label="Recovered implied vol",
        markersize=5,
        alpha=0.8,
    )
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title("Volatility smile: true vs recovered")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path("assets").mkdir(exist_ok=True)
    fig.savefig("assets/results.png", dpi=120)
    Path("results").mkdir(exist_ok=True)
    Path("results/smile.json").write_text(
        json.dumps(
            {
                "kind": "Synthetic-input solver diagnostic",
                "S": S,
                "T": T,
                "r": r,
                "strikes": strikes.tolist(),
                "input_vols": true_vols.tolist(),
                "prices": prices,
                "recovered_vols": recovered,
                "max_absolute_error": float(
                    np.max(np.abs(np.array(recovered) - true_vols))
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print("Saved assets/results.png")


if __name__ == "__main__":
    figure_newton_convergence()
    figure_vol_smile()
