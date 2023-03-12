
import sys
import argparse
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go


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
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default=("../../figures/predicted_counts_"
                                 "nNeurons_{:d}_nTrials_{:d}.{:s}"))
    args = parser.parse_args()

    n_trials = args.n_trials
    n_neurons = args.n_neurons
    target_neuron_index = args.target_neuron_index
    glm_family = args.glm_family
    data_filename = args.data_filename_pattern.format(n_neurons, n_trials)
    fig_filename_pattern = args.fig_filename_pattern

    load_res = np.load(data_filename)
    count_data = load_res["count_data"]

    neurons_indices = np.arange(n_neurons)
    X = count_data[:, neurons_indices != target_neuron_index]
    y = count_data[:, target_neuron_index]

    exog = sm.add_constant(X)
    if glm_family == "Poisson":
        model = sm.GLM(y, exog, family=sm.families.Poisson())
    elif glm_family == "Gaussian":
        model = sm.GLM(y, exog, family=sm.families.Gaussian())
    else:
        raise ValueError("Invalid glm_family: {glm_family}")
    result = model.fit()
    print(result.summary())

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
