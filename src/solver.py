"""Implied volatility solver.

Given a market price and observable parameters (S, K, T, r), finds the
volatility sigma such that BS(S, K, T, r, sigma) = market_price.

Strategy:
  1. Try Newton-Raphson (fast, quadratic convergence) using vega as the
     Jacobian. If it converges in max_iter iterations, return the result.
  2. If Newton fails (vega too small, sigma goes negative, non-convergence),
     fall back to Brent's method (slower but guaranteed to converge given
     valid brackets).

Run `python -m src.solver` for a demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .bs import bs_price, bs_vega


@dataclass
class ImpliedVolResult:
    """Result of an implied-vol computation."""
    sigma: float           # the implied volatility
    price_error: float     # |BS(sigma) - market_price|, should be ~0
    iterations: int        # how many iterations the solver took
    method: str            # "newton" or "brent"
    converged: bool


def implied_vol_newton(
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: str = "call",
    initial_guess: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> ImpliedVolResult:
    """Find implied vol via Newton-Raphson.

    Newton's update rule:
        sigma_{n+1} = sigma_n - (BS(sigma_n) - market_price) / vega(sigma_n)

    Convergence is quadratic when vega is well-behaved (away from deep
    OTM/ITM and away from expiry). Fails when vega ≈ 0 because the
    denominator vanishes.
    """
    sigma = initial_guess

    for i in range(1, max_iter + 1):
        price = bs_price(S, K, T, r, sigma, option_type)
        v = bs_vega(S, K, T, r, sigma)

        error = price - market_price

        if abs(error) < tol:
            return ImpliedVolResult(
                sigma=sigma, price_error=abs(error),
                iterations=i, method="newton", converged=True
            )

        if abs(v) < 1e-12:
            # Vega is essentially zero — Newton can't make progress.
            # This happens for deep OTM options or very short T.
            break

        sigma = sigma - error / v

        # Guard: sigma must stay positive.
        if sigma <= 0:
            break

    return ImpliedVolResult(
        sigma=sigma, price_error=abs(bs_price(S, K, T, r, sigma, option_type) - market_price),
        iterations=max_iter, method="newton", converged=False
    )


def implied_vol_brent(
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: str = "call",
    sigma_low: float = 1e-6,
    sigma_high: float = 5.0,
    tol: float = 1e-8,
) -> ImpliedVolResult:
    """Find implied vol via Brent's method (bracketed root-finding).

    Brent's method is guaranteed to converge as long as the objective
    function changes sign over [sigma_low, sigma_high]. For Black-Scholes
    this is guaranteed because BS(sigma_low) ≈ intrinsic < market_price
    and BS(sigma_high) >> market_price for any sensible sigma_high.

    Slower than Newton (linear convergence) but robust to all the edge
    cases that break Newton.
    """
    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    # Verify the brackets are valid (objective changes sign).
    f_low = objective(sigma_low)
    f_high = objective(sigma_high)

    if f_low * f_high > 0:
        # No sign change — market price is outside the achievable range.
        # Either the price is below intrinsic (arbitrage) or above the
        # upper bound at sigma=5.0 (unrealistic).
        return ImpliedVolResult(
            sigma=float("nan"), price_error=abs(f_low),
            iterations=0, method="brent", converged=False
        )

    sigma, result = brentq(objective, sigma_low, sigma_high,
                           xtol=tol, full_output=True)
    return ImpliedVolResult(
        sigma=float(sigma),
        price_error=abs(objective(sigma)),
        iterations=result.iterations,
        method="brent",
        converged=result.converged,
    )


def implied_vol(
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: str = "call",
    tol: float = 1e-8,
) -> ImpliedVolResult:
    """Find implied volatility, trying Newton first, falling back to Brent.

    This is the recommended entry point. It's fast (Newton usually converges
    in 4-6 iterations) and robust (Brent catches the edge cases).
    """
    result = implied_vol_newton(market_price, S, K, T, r, option_type, tol=tol)
    if result.converged:
        return result

    # Newton failed; fall back to Brent.
    return implied_vol_brent(market_price, S, K, T, r, option_type, tol=tol)


def _demo() -> None:
    """Demonstrate the solver on a few examples."""
    S, K, T, r = 100, 100, 1.0, 0.05

    print("Implied volatility solver demo")
    print("=" * 60)
    print(f"Parameters: S={S}, K={K}, T={T}, r={r}")
    print()

    # Generate "market prices" from known vols, then recover the vols.
    for true_sigma in [0.10, 0.20, 0.30, 0.50, 0.80]:
        mkt_price = bs_price(S, K, T, r, true_sigma, "call")
        result = implied_vol(mkt_price, S, K, T, r, "call")
        error = abs(result.sigma - true_sigma)
        print(f"  True σ={true_sigma:.2f}  →  price={mkt_price:.4f}  →  "
              f"recovered σ={result.sigma:.6f}  "
              f"(error={error:.2e}, {result.iterations} iters, {result.method})")

    print()
    # Edge case: deep OTM call.
    deep_otm_price = bs_price(100, 150, 0.1, 0.05, 0.2, "call")
    result = implied_vol(deep_otm_price, 100, 150, 0.1, 0.05, "call")
    print(f"  Deep OTM (K=150, T=0.1, σ=0.2): price={deep_otm_price:.6f}  →  "
          f"recovered σ={result.sigma:.6f} ({result.method})")


if __name__ == "__main__":
    _demo()