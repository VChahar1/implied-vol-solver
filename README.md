A synthetic quadratic volatility smile (higher implied vol for away-from-ATM strikes) is recovered numerically by the solver for the tested strikes.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.solver         # demo: recover known vols from prices
python -m src.make_figure    # generate local comparison figures
pytest -v                    # 18 tests in the cleanup run
```

## Verification

The solver is tested against:
- Known-volatility test cases between σ = 0.05 and 1.0 recovered within 1e-6 absolute error.
- Put-call parity (BS call − put = S − K·e^{−rT}).
- Vega positivity (a basic sanity check).
- Deep OTM edge cases where Newton fails and Brent catches.
- A smile round-trip: generate prices from specified smile points and recover their vols within 1e-5 absolute error. The deep-OTM Brent case uses a 1e-4 tolerance.

## Notes and extensions

This implementation handles vanilla European options under Black-Scholes. Possible extensions include fitting a parametric volatility surface and adding dividend-aware pricing.

TODO: benchmark alternative solvers before making numerical iteration-count or comparative performance claims. No evidence for the previous alternative-solver speed or production-adoption claims is included in this repository.
