
import sys
import pickle
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
    parser.add_argument("--data_filename_pattern", type=str,
                        help="data filename pattern",
                        default=("../../results/counts_nNeurons_{:d}_"
                                 "nTrials_{:d}.npz"))
    parser.add_argument("--model_filename_pattern", type=str,
                        help="model filename pattern",
                        default=("../../results/model{:s}_target_{:d}_"
                                 "nNeurons_{:d}_nTrials_{:d}.pickle"))
    args = parser.parse_args()

    n_trials = args.n_trials
    n_neurons = args.n_neurons
    target_neuron_index = args.target_neuron_index
    data_filename = args.data_filename_pattern.format(n_neurons, n_trials)
    model_filename_pattern = args.model_filename_pattern 

    load_res = np.load(data_filename)
    count_data = load_res["count_data"]
    neurons_indices = np.arange(n_neurons)
    X = count_data[:, neurons_indices != target_neuron_index]
    y = count_data[:, target_neuron_index]

    poisson_model = sklearn.linear_model.PoissonRegressor(alpha=0.0)
    gaussian_model = sklearn.linear_model.LinearRegression()
    poisson_model.fit(X=X, y=y)
    gaussian_model.fit(X=X, y=y)

    poisson_model_filename = model_filename_pattern.format("Poisson",
                                                           target_neuron_index,
                                                           n_neurons, n_trials)
    with open(poisson_model_filename, "wb") as f:
        pickle.dump(poisson_model, f) 

    gaussian_model_filename = model_filename_pattern.format("Gaussian",
                                                            target_neuron_index,
                                                            n_neurons, n_trials)
    with open(gaussian_model_filename, "wb") as f:
        pickle.dump(gaussian_model, f) 

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
