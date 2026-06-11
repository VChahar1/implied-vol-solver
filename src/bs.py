"""Black-Scholes pricing and Greeks needed for the implied-vol solver.

We need the price (to evaluate the objective) and vega (to drive Newton-
Raphson). Both are standard closed-form results.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes price for a European call or put."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S - K * np.exp(-r * T), 0.0)
        return max(K * np.exp(-r * T) - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega: the derivative of the Black-Scholes price with respect to sigma.

    Vega is the same for calls and puts (follows from put-call parity, since
    the parity relation S - K*exp(-rT) doesn't depend on sigma).

    This is the Jacobian that Newton-Raphson uses to find implied vol.
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(S * np.sqrt(T) * norm.pdf(d1))