A synthetic quadratic volatility smile (higher implied vol for away-from-ATM strikes) is recovered exactly by the solver across the full strike range.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.solver         # demo: recover known vols from prices
python -m src.make_figure    # regenerate the figures above
pytest -v                    # ~15 tests
```

## Verification

The solver is tested against:
- Known volatilities recovered to 1e-6 accuracy across σ ∈ [0.05, 1.0].
- Put-call parity (BS call − put = S − K·e^{−rT}).
- Vega positivity (a basic sanity check).
- Deep OTM edge cases where Newton fails and Brent catches.
- A full smile round-trip: generate prices from a quadratic smile, recover all vols.

## Notes and extensions

This implementation handles vanilla European options under Black-Scholes. Natural extensions include: SABR or SVI parameterization of the smile surface (fitting σ(K,T) rather than solving pointwise), dividend-adjusted BS for equity options, and the Jaeckel (2015) "Let's Be Rational" method — which replaces Newton-Raphson with an initial approximation that's accurate to machine precision in ~2 iterations, eliminating the need for a fallback entirely. The Jaeckel method is what Bloomberg and most production systems use.

The vega-based Newton-Raphson approach implemented here is the standard teaching version and is the method most likely to appear in quant interviews.
