"""Tests for the implied volatility solver."""
from __future__ import annotations

import numpy as np
import pytest

from src.bs import bs_price, bs_vega
from src.solver import implied_vol, implied_vol_newton, implied_vol_brent


# ---- Black-Scholes sanity checks ----

def test_atm_call_positive():
    assert bs_price(100, 100, 1.0, 0.05, 0.2, "call") > 0


def test_put_call_parity():
    S, K, T, r, sigma = 100, 105, 0.5, 0.03, 0.25
    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")
    parity = call - put - (S - K * np.exp(-r * T))
    assert abs(parity) < 1e-10


def test_vega_positive():
    assert bs_vega(100, 100, 1.0, 0.05, 0.2) > 0


def test_vega_near_zero_for_deep_otm():
    assert bs_vega(100, 200, 0.01, 0.05, 0.1) < 1e-6


# ---- Newton-Raphson ----

@pytest.mark.parametrize("true_sigma", [0.05, 0.10, 0.20, 0.40, 0.80])
def test_newton_recovers_known_vol(true_sigma):
    S, K, T, r = 100, 100, 1.0, 0.05
    mkt = bs_price(S, K, T, r, true_sigma, "call")
    result = implied_vol_newton(mkt, S, K, T, r, "call")
    assert result.converged
    assert abs(result.sigma - true_sigma) < 1e-6


def test_newton_works_for_puts():
    true_sigma = 0.25
    S, K, T, r = 100, 110, 0.5, 0.03
    mkt = bs_price(S, K, T, r, true_sigma, "put")
    result = implied_vol_newton(mkt, S, K, T, r, "put")
    assert result.converged
    assert abs(result.sigma - true_sigma) < 1e-6


def test_newton_few_iterations():
    mkt = bs_price(100, 100, 1.0, 0.05, 0.3, "call")
    result = implied_vol_newton(mkt, 100, 100, 1.0, 0.05, "call")
    assert result.iterations <= 10


# ---- Brent fallback ----

@pytest.mark.parametrize("true_sigma", [0.05, 0.50, 1.00])
def test_brent_recovers_known_vol(true_sigma):
    S, K, T, r = 100, 100, 1.0, 0.05
    mkt = bs_price(S, K, T, r, true_sigma, "call")
    result = implied_vol_brent(mkt, S, K, T, r, "call")
    assert result.converged
    assert abs(result.sigma - true_sigma) < 1e-6


def test_brent_handles_extreme_otm():
    """Deep OTM: vega is tiny, Newton would struggle, Brent should work."""
    mkt = bs_price(100, 160, 0.1, 0.05, 0.3, "call")
    result = implied_vol_brent(mkt, 100, 160, 0.1, 0.05, "call")
    assert result.converged
    assert abs(result.sigma - 0.3) < 1e-4


# ---- Combined solver ----

def test_combined_atm_call():
    mkt = bs_price(100, 100, 1.0, 0.05, 0.25, "call")
    result = implied_vol(mkt, 100, 100, 1.0, 0.05, "call")
    assert result.converged
    assert abs(result.sigma - 0.25) < 1e-6


def test_combined_invalid_price_no_crash():
    """A near-zero price for a deep OTM call should not crash."""
    result = implied_vol(0.0001, 100, 200, 1.0, 0.05, "call")
    assert isinstance(result.sigma, float)


def test_smile_roundtrip():
    """Generate prices from a smile, recover the vols."""
    S, T, r = 100, 0.5, 0.05
    strikes = [80, 90, 100, 110, 120]
    true_vols = [0.30, 0.24, 0.20, 0.24, 0.30]
    for K, tv in zip(strikes, true_vols):
        mkt = bs_price(S, K, T, r, tv, "call")
        result = implied_vol(mkt, S, K, T, r, "call")
        assert result.converged
        assert abs(result.sigma - tv) < 1e-5