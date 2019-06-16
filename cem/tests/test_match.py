import unittest
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
        for i in range(10000):
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
    def test_match(self):
        m = ExactMatching()
