import numpy as np


def simulatePoissonCountsInTrials(n_neurons, n_trials, baseline_count=1.0,
                                  min_trials_weight=0.0, max_trials_weight=5.0,
                                  min_neurons_weight=0.0,
                                  max_neurons_weight=1.0):
    ''' simulate activity of N neurons for T trials.'''
    trials_weights = np.linspace(min_trials_weight, max_trials_weight,
                                 n_trials)
    neurons_weights = np.linspace(min_neurons_weight, max_neurons_weight,
                                  n_neurons)**2
    counts = np.random.poisson(
        baseline_count + trials_weights[:, None] * neurons_weights[None, :])
    return counts, trials_weights, neurons_weights

def computePoissonLoglike(intercept, coefs, y, X):
    pred_mean = np.exp(intercept + X @ coefs)
    y_factorial = np.array([np.math.factorial(an_y) for an_y in y])
    loglike = np.sum(y*np.log(pred_mean) - pred_mean - np.log(y_factorial))
    return loglike


def computeGaussianLoglike(intercept, coefs, y, X):
    pred_mean = intercept + X @ coefs
    residuals = y - pred_mean
    sigma2 = np.var(residuals)
    N = len(y)
    loglike = -0.5*(N*np.log(2*np.pi*sigma2) + np.sum(residuals**2)/sigma2)
    return loglike



