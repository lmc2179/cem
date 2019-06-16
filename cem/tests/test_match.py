import unittest
import numpy as np
import random
from cem.match import ExactMatching, Stratum

class StratumTest(unittest.TestCase):
    def test_stratum(self):
        s = Stratum(0, [0,1,2], [1,2,3])
        delta = s.get_difference()
        self.assertEqual(delta, 1)

# Simulate a difference in mean: mu_1, mu_0, sigma_1, sigma_0; confirm that CI contains true mean alpha percent of the time

class MatchTest(unittest.TestCase):
    def test_match(self):
        m = ExactMatching()
