import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import norm

class Stratum(object):
    def __init__(self, signature, y_control, y_treatment):
        self.signature = signature
        self.y_control = y_control
        self.y_treatment = y_treatment
        self.difference = np.mean(y_treatment) - np.mean(y_control)
        self.sigma = np.sqrt((np.var(y_control)/len(y_control)) + (np.var(y_treatment)/len(y_treatment)))

    def get_signature(self):
        return self.signature

    def get_difference(self):
        return self.difference

    def get_sigma(self):
        return self.sigma

    def get_difference_confint(self, alpha):
        z = norm(0, 1).interval(1. - alpha)[1]
        w = z * self.sigma
        return self.difference - w, self.difference + w

    def get_control_count(self):
        return len(self.y_control)

    def get_treatment_count(self):
        return len(self.y_treatment)

class ExactMatching(object):
    def __init__(self):
        self.strata = None

    def match(self, X, t, y):
        control_obs, treatment_obs = defaultdict(list), defaultdict(list)
        """Perform exact matching using covariates X (pandas dataframe), treatment indicator t, and response variable y."""
        for i, tup in enumerate(X.iterrows()):
            _, x_row = tup
            if t[i] == 1:
                treatment_obs[tuple(x_row)].append(y[i])
            elif t[i] == 0:
                control_obs[tuple(x_row)].append(y[i])
            else:
                raise Exception('Treatment values must be 1 or 0, but got {0}'.format(t[i]))
        unique_signatures = list(set(treatment_obs) | set(control_obs))
        self.strata = {}
        for k in unique_signatures:
            self.strata[k] = Stratum(k, control_obs[k], treatment_obs[k])

    def get_att(self):
        strata_df = self.get_strata_dataframe()
        w = strata_df['treatment_count'] / strata_df['treatment_count'].sum()
        return np.dot(strata_df['diff'], w)

    def get_atc(self):
        strata_df = self.get_strata_dataframe()
        w = strata_df['control_count'] / strata_df['control_count'].sum()
        return np.dot(strata_df['diff'], w)

    def get_ate(self):
        strata_df = self.get_strata_dataframe()
        w = strata_df['total_count'] / strata_df['total_count'].sum()
        return np.dot(strata_df['diff'], w)

    def get_strata(self):
        return self.strata

    def get_strata_dataframe(self):
        rows = [[sig, s.get_difference(), s.get_sigma(), s.get_control_count(), s.get_treatment_count(), s.get_control_count() + s.get_treatment_count()] for sig, s in self.get_strata().items()]
        return pd.DataFrame(rows, columns=['signature', 'diff', 'sigma', 'control_count', 'treatment_count', 'total_count'])

    def get_pruned_strata(self):
        pass
