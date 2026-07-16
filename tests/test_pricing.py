import unittest
from pricing_engine.options import EuropeanOption

class TestOptionPricingEngine(unittest.TestCase):
    def setUp(self):
        self.call = EuropeanOption(S=100, K=105, T=1.0, r=0.05, sigma=0.20, option_type='call')
        self.put = EuropeanOption(S=100, K=105, T=1.0, r=0.05, sigma=0.20, option_type='put')

    def test_analytical_greek_tolerances(self):
        # Numerical finite difference Greeks should align tightly with Analytical formulas
        self.assertAlmostEqual(self.call.delta(), self.call.delta_fd(), places=4)
        self.assertAlmostEqual(self.call.gamma(), self.call.gamma_fd(), places=4)
        self.assertAlmostEqual(self.call.vega(), self.call.vega_fd(), places=4)
        self.assertAlmostEqual(self.call.theta(), self.call.theta_fd(), places=4)

    def test_binomial_tree_convergence(self):
        # With enough steps (e.g., 2000), the CRR model must converge tightly with Black-Scholes
        bs_price, _ = self.call.price_black_scholes()
        bt_price, _ = self.call.price_binomial_tree(steps=2000)
        self.assertAlmostEqual(bs_price, bt_price, places=2)

if __name__ == '__main__':
    unittest.main()
