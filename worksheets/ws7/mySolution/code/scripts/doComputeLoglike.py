
import sys
import argparse
import numpy as np
import sklearn.linear_model

import utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, help="Number of trials",
                        default=100)
    parser.add_argument("--n_neurons", type=int, help="Number of neurons",
                        default=30)
    parser.add_argument("--target_neuron_index", type=int,
                        help="target neuron index", default=29)
    parser.add_argument("--glm_family", type=str,
                        help="distribution family used in GLM",
                        default="Poisson")
    parser.add_argument("--data_filename_pattern", type=str,
                        help="data filename pattern",
                        default=("../../results/counts_nNeurons_{:d}_"
                                 "nTrials_{:d}.npz"))
    args = parser.parse_args()

    n_trials = args.n_trials
    n_neurons = args.n_neurons
    target_neuron_index = args.target_neuron_index
    glm_family = args.glm_family
    data_filename = args.data_filename_pattern.format(n_neurons, n_trials)

    load_res = np.load(data_filename)
    count_data = load_res["count_data"]

    neurons_indices = np.arange(n_neurons)
    if glm_family == "Poisson":
        model = sklearn.linear_model.PoissonRegressor(alpha=0.0)
    elif glm_family == "Gaussian":
        model = sklearn.linear_model.LinearRegression()
    else:
        raise ValueError(f"Invalid glm_family: {glm_family}")
    X = count_data[:, neurons_indices != target_neuron_index]
    y = count_data[:, target_neuron_index]
    model.fit(X=X, y=y)
    if glm_family == "Poisson":
        loglike = utils.computePoissonLoglike(intercept=model.intercept_,
                                              coefs=model.coef_,
                                              y=y, X=X)
    elif glm_family == "Gaussian":
        loglike = utils.computeGaussianLoglike(intercept=model.intercept_,
                                               coefs=model.coef_,
                                               y=y, X=X)
    else:
        raise ValueError(f"Invalid glm_family: {glm_family}")
    print(f"loglike: {loglike}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
