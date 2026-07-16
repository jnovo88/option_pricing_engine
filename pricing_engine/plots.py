"""
Plotting utilities for visualizing payoffs, net profits, time decay curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_payoff_profit_decay(option, S_range=None, times_before_expiry=None, ax=None):
    """
    Plot option payoff, undiscounted net P&L, and Black-Scholes curve progressions.
    """
    K, r, sigma, T, otype = option.K, option.r, option.sigma, option.T, option.option_type

    if S_range is None:
        S_range = np.linspace(0.5 * K, 1.5 * K, 200)
    S_range = np.asarray(S_range, dtype=float)

    if times_before_expiry is None:
        times_before_expiry = [T, T / 2, T / 4, T / 12]

    payoff = option.payoff(S_range) # The payoffs for all considered asset values at expiry
    premium, _ = option.price_black_scholes() # The premium is the value of the option at t=0
    profit = payoff - premium

    # Black-Scholes value
    def bs_value(S, tau):
        if tau <= 0:
            return option.payoff(S) # zero or negative time to expiry, yield the value at expiry
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        if otype == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        else:
            return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(S_range, payoff, color='black', lw=2.5,
            label=f'Payoff at expiry  max({"S−K" if otype=="call" else "K−S"}, 0)')

    ax.plot(S_range, profit, color='crimson', lw=2, linestyle='--',
            label=f'Net P&L (premium = {premium:.2f})')
    ax.axhline(0, color='grey', lw=0.8, alpha=0.6)

    taus = sorted(times_before_expiry, reverse=True)
    cmap = plt.cm.Blues
    for i, tau in enumerate(taus):
        shade = 0.35 + 0.6 * (i + 1) / len(taus)
        ax.plot(S_range, bs_value(S_range, tau), color=cmap(shade), lw=1.8,
                label=f'Value at t = T − {tau:.3g}  (τ = {tau:.3g})')

    label_y = ax.get_ylim()[1] * 0.02
    ax.axvline(K, color='grey', lw=0.8, linestyle=':', alpha=0.7)
    ax.text(K, label_y, f'  K = {K:g}', color='grey', fontsize=9, va='bottom')

    break_even = K + premium if otype == 'call' else K - premium
    ax.axvline(break_even, color='crimson', lw=0.8, linestyle=':', alpha=0.6)
    ax.text(break_even, ax.get_ylim()[1] * 0.08, f'  break-even = {break_even:.2f}', color='crimson', fontsize=9, va='bottom')

    ax.set_xlabel('Underlying price  S')
    ax.set_ylabel('Value  /  P&L')
    ax.set_title(f'European {otype.capitalize()}  —  Payoff, P&L, and Time Decay')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    return ax

