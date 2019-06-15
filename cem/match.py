import numpy as np
import pandas as pd

class Stratum(object):
    def __init__(self, signature, y_control, y_treatment):
        self.signature = signature
        self.y_control = y_control
        self.y_treatment = y_treatment
        self.difference = np.mean(y_treatment) - np.mean(y_control)
        self.sigma = np.sqrt((np.var(y_control)/len(y_control)) + (np.var(y_treatment)/len(y_treatment)))

    def get_difference(self):
        return self.difference

class ExactMatching(object):
    def match(self, X, t, y):
        """Perform exact matching using covariates X, treatment indicator t, and response variable y."""
        pass
