
import sys
import argparse
import numpy as np
import scipy.stats
import statsmodels.api as sm

import utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, help="number of trials",
                        default=100)
    parser.add_argument("--n_neurons", type=int, help="number of neurons",
                        default=30)
    parser.add_argument("--target_neuron_index", type=int,
                        help="target neuron index", default=29)
    parser.add_argument("--glm_family", type=str,
                        help="distribution family used in GLM",
                        default="Poisson")
    args = parser.parse_args()

    n_trials = args.n_trials
    n_neurons = args.n_neurons
    target_neuron_index = args.target_neuron_index
    glm_family = args.glm_family

    count_data, trials_weights, neurons_weigts = \
        utils.simulatePoissonCountsInTrials(
            n_trials=n_trials,
            n_neurons=n_neurons)

    neurons_indices = np.arange(n_neurons)
    X = count_data[:, neurons_indices != target_neuron_index]
    y = count_data[:, target_neuron_index]
    # X = np.random.poisson(size=(n_trials, n_neurons-1))
    # y = np.random.poisson(size=(n_trials, 1))

    exog = sm.add_constant(X)
    reduced_exog = np.ones(len(y))
    if glm_family == "Poisson":
        full_model = sm.GLM(y, exog, family=sm.families.Poisson())
        reduced_model = sm.GLM(y, reduced_exog, family=sm.families.Poisson())
    elif glm_family == "Gaussian":
        full_model = sm.GLM(y, exog, family=sm.families.Gaussian())
        reduced_model = sm.GLM(y, reduced_exog, family=sm.families.Gaussian())
    else:
        raise ValueError("Invalid glm_family: {glm_family}")
    full_result = full_model.fit()
    reduced_result = reduced_model.fit()
    p_value_intercept = 1 - scipy.stats.chi2.cdf(
        x=reduced_result.deviance-full_result.deviance,
        df=len(full_result.params)-len(reduced_result.params),
    )
    p_value_saturated = 1 - scipy.stats.chi2.cdf(
        x=reduced_result.deviance,
        df=n_trials-len(reduced_result.params),
    )
    print(f"p_value_intercept={p_value_intercept}")
    print(f"p_value_reduced={p_value_reduced}")
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
