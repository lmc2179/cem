import unittest
import numpy as np
import random
from cem.match import ExactMatching, Stratum

class StratumTest(unittest.TestCase):
    def test_stratum(self):
        s = Stratum(0, [0,1,2], [1,2,3])
        delta = s.get_difference()
        self.assertEqual(delta, 1)

class MatchTest(unittest.TestCase):
    def test_match(self):
        m = ExactMatching()
