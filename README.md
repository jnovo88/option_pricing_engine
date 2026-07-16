# Option Pricing Engine

A small Python project for pricing European options with Black-Scholes, Monte Carlo simulation, and a vectorized Cox-Ross-Rubinstein binomial tree.

## Features

- Black-Scholes pricing
- Monte Carlo valuation
- Vectorized binomial tree pricing
- Analytical and finite-difference Greeks
- Notebook-based demo with payoff and convergence plots

## Requirements

- Python 3.10+
- NumPy
- SciPy
- Matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
python -m unittest discover -s tests -p "test*.py" -q
```

## Usage

See [notebooks/pricing_demo.ipynb](notebooks/pricing_demo.ipynb) for an end-to-end example.
