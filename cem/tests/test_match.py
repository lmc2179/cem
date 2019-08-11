import unittest
import pandas as pd
import numpy as np
import random
from cem.match import ExactMatching, Stratum


class StratumTest(unittest.TestCase):
    def test_difference_calculation(self):
        s = Stratum(0, [0,1,2], [1,2,3])
        delta = s.get_difference()
        self.assertEqual(delta, 1)

    def test_ci(self):        
        correct, total = 0., 0.
        for i in range(5000):
            mu_1, mu_0 = np.random.normal(0, 10, 2)
            s_1, s_0 = np.abs(np.random.normal(0, 10, 2))
            N = 1000
            x_1 = np.random.normal(mu_1, s_1, N)
            x_0 = np.random.normal(mu_0, s_0, N)
            s = Stratum(0, x_0, x_1)
            l, u = s.get_difference_confint(.05)
            if l <= (mu_1 - mu_0) <= u:
                correct += 1
            total += 1
        self.assertEqual(round(correct / total, 2), 1. - .05)

# Simulate a difference in mean: mu_1, mu_0, sigma_1, sigma_0; confirm that CI contains true mean alpha percent of the time

class MatchTest(unittest.TestCase):
    def test_simple_match_one_stratum(self):
        X = pd.DataFrame({'match_var': ['A']*6})
        t = [0, 0, 0, 1, 1, 1]
        y = [0, 0, 0, 1, 1, 1]
        m = ExactMatching()
        m.match(X, t, y)
        strata = m.get_strata() 
        self.assertEqual(len(strata), 1)
        self.assertEqual(strata[('A',)].get_difference(), 1)
        self.assertEqual(m.get_att(), 1)
        self.assertEqual(m.get_atc(), 1)
        self.assertEqual(m.get_ate(), 1)

    def test_simple_match_two_stratum(self):
        pass

    @unittest.skip('make this faster')
    def test_confounding(self):
        def _sim_t(x):
            p = .25 if x == 1 else 0.75
            return np.random.binomial(1, p, 1)
        def _sim_y(t, x):
            p = .4 if x == 1 else .6
            if t == 1:
                p += 0.05
            return np.random.binomial(1, p, 1)
        N = 100000
        X = pd.DataFrame({'match_var': np.random.binomial(1, 0.45, N)})
        t = [_sim_t(x) for x in X['match_var']]
        y = [_sim_y(t, x) for t, x in zip(t, X['match_var'])]
 
        m = ExactMatching()
        m.match(X, t, y)
        strata = m.get_strata()
        self.assertEqual(len(strata), 2)
        self.assertEqual(round(strata[(1,)].get_difference(), 2), .05)
        self.assertEqual(round(strata[(0,)].get_difference(), 2), .05)
