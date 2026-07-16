"""
European Option Pricing Engine.
Includes analytical Black-Scholes, Monte Carlo Simulation, and Vectorized Binomial Tree pricing frameworks,
along with analytical and numerical Finite Difference Greeks.
"""

from copy import copy
import numpy as np
from scipy.stats import norm
import time


class EuropeanOption:
    """
    Class to model and price a European Option (both call and put) using three pricing methods,
    and compute standard analytical and numerical risk sensitivities (Greeks).
    params:
    S: underlying asset initial value 
    K: strike price
    T: time to expiry
    r: risk-free interet rate
    sigma: annualized volatility of underlying asset's returns

    For comparison the clock time of each method is also measured
    """
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.option_type = option_type.lower()
        if self.option_type not in ('call', 'put'):
            raise ValueError("option_type must be either 'call' or 'put'")
        if self.T<0:
            raise ValueError("Time to expiry must be non-negative")
        if self.sigma<0:
            raise ValueError("Volatility must be non-negative")
    
    # Auxiliary functions
    def payoff(self, ST):
        """
        Calculates option payoff at maturity given a vector of underlying prices, ST.
        """
        ST = np.asarray(ST)
        if self.option_type == 'call':
            return np.maximum(0.0, ST - self.K)
        else:
            return np.maximum(0.0, self.K - ST)
        
    
    def d1_d2(self):
        """Compute d1 and d2. Assumes T > 0."""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2


    def reprice(self, **overrides):
        """
        Return an analytical price with one or more inputs modified.
        Useful for the finite differences.
        """
        o = copy(self)
        for k, v in overrides.items():
            setattr(o, k, v)
        price, _ = o.price_black_scholes()
        return price
    

    # Pricing fucntions
    def price_black_scholes(self):
        """Analytical Black-Scholes valuation."""
        start = time.perf_counter()
        if self.T == 0:
            return self.payoff(self.S), time.perf_counter() - start

        #d1 and d2 are computed by an auxilliary function    
        d1, d2 = self.d1_d2()
        
        if self.option_type == 'call':
            price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
            
        return price, time.perf_counter() - start

    def price_monte_carlo(self, simulations=100000, seed=None):
        """
        Valuation via Monte Carlo path simulation of Geometric Brownian Motion.
        There is an option to specify a seed in order to get reproducibility in code testing.
        """
        start = time.perf_counter()
        if self.T == 0:
            return self.payoff(self.S), time.perf_counter() - start
        

        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(simulations) #Vector of random variables extracted from N(0,1)

        ST = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)
        payoffs = self.payoff(ST)
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return price, time.perf_counter() - start

    def price_binomial_tree(self, steps=100):
        """Vectorized Binomial Tree pricing model (Cox-Ross-Rubinstein)."""
        start = time.perf_counter()
        if self.T <= 0:
            return self.payoff(self.S), time.perf_counter() - start

        delta_t = self.T / steps
        u = np.exp(self.sigma * np.sqrt(delta_t)) # up step
        d = 1.0 / u # down step
        p = (np.exp(self.r * delta_t) - d) / (u - d)
        discount = np.exp(-self.r * delta_t)

        j = np.arange(steps + 1) # number of possibilities at the T
        ST = self.S * (u ** j) * (d ** (steps - j)) # possible asset values at T
        V = self.payoff(ST) # option values at T

        for i in range(steps - 1, -1, -1):
            V = discount * (p * V[1:] + (1.0 - p) * V[:-1]) # going backwards in the tree until t=0 is reached

        price = V[0]
        return price, time.perf_counter() - start


    # --- ANALYTICAL GREEKS (for Black-Scholes model) ---
    def delta(self):
        """Sensitivity of option value to underlying spot: dV/dS."""
        if self.T == 0:
            if self.option_type == 'call':
                return 1.0 if self.S > self.K else 0.0
            else:
                return -1.0 if self.S < self.K else 0.0

        d1, _ = self.d1_d2()
        if self.option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0

    def gamma(self):
        """Sensitivity of delta to spot (curvature of price): d^2V/dS^2."""
        if self.T == 0:
            return 0.
        d1, _ = self.d1_d2()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        """Sensitivity of option price to underlying volatility: dV/dsigma."""
        if self.T == 0:
            return 0.
        d1, _ = self.d1_d2()
        return self.S * norm.pdf(d1) * np.sqrt(self.T)

    def theta(self):
        """Time-decay of option price (per year): dV/dt."""
        if self.T == 0:
            return 0.
        d1, d2 = self.d1_d2()
        first = -(self.S * norm.pdf(d1) * self.sigma) / (2.0 * np.sqrt(self.T))
        if self.option_type == 'call':
            return first - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return first + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)

    def rho(self):
        """Sensitivity of option price to interest rates: dV/dr."""
        if self.T == 0:
            return 0.
        _, d2 = self.d1_d2()
        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)

    # --- NUMERICAL (FINITE DIFFERENCE) GREEKS (for Black-Scholes model)---
    def delta_fd(self, h=None):
        """Central-difference numerical delta."""
        if h is None:
            h = 1e-4 * self.S
        up = self.reprice(S=self.S + h)
        down = self.reprice(S=self.S - h)
        return (up - down) / (2.0 * h)

    def gamma_fd(self, h=None):
        """Three-point second-difference numerical gamma."""
        if h is None:
            h = 1e-3 * self.S
        up = self.reprice(S=self.S + h)
        mid, _ = self.price_black_scholes()
        down = self.reprice(S=self.S - h)
        return (up - 2.0 * mid + down) / (h * h)

    def vega_fd(self, h=1e-4):
        """Central-difference numerical vega."""
        up = self.reprice(sigma=self.sigma + h)
        down = self.reprice(sigma=self.sigma - h)
        return (up - down) / (2.0 * h)

    def theta_fd(self, h=None):
        """Central-difference numerical theta with boundary handling at expiry."""
        if h is None:
            h = 1e-4 * max(self.T, 1e-6)
        if self.T - h <= 0:
            up = self.reprice(T=self.T + h)
            mid, _ = self.price_black_scholes()
            return -(up - mid) / h
        up = self.reprice(T=self.T + h)
        down = self.reprice(T=self.T - h)
        return -(up - down) / (2.0 * h)

    def rho_fd(self, h=1e-5):
        """Central-difference numerical rho."""
        up = self.reprice(r=self.r + h)
        down = self.reprice(r=self.r - h)
        return (up - down) / (2.0 * h)
