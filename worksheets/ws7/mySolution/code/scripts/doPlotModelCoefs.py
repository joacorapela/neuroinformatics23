
import sys
import argparse
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

import utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, help="number of trials",
                        default=1000)
    parser.add_argument("--n_neurons", type=int, help="number of neurons",
                        default=30)
    parser.add_argument("--target_neuron_index", type=int,
                        help="target neuron index", default=29)
    parser.add_argument("--glm_family", type=str,
                        help="distribution family used in GLM",
                        default="Poisson")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="results filename pattern",
                        default=("../../figures/coefs_{:s}_"
                                 "nNeurons_{:d}_nTrials_{:d}.{:s}"))
    args = parser.parse_args()

    n_trials = args.n_trials
    n_neurons = args.n_neurons
    target_neuron_index = args.target_neuron_index
    glm_family = args.glm_family
    fig_filename_pattern = args.fig_filename_pattern

    count_data, trials_weights, neurons_weigts = \
        utils.simulatePoissonCountsInTrials(
            n_trials=n_trials,
            n_neurons=n_neurons)

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
    coefs = result.params
    conf_int = result.conf_int()
    coefs_indices = np.arange(len(coefs))

    fig = go.Figure()
    trace = go.Scatter(x=coefs_indices,
                       y=coefs,
                       error_y=dict(
                           type="data",
                           symmetric=False,
                           array=conf_int[:, 1]-coefs,
                           arrayminus=coefs-conf_int[:, 0],
                       ),
                      )
    fig.add_trace(trace)
    fig.add_hline(y=0.0)
    fig.update_layout(
        xaxis_title="Coefficient Index",
        yaxis_title="Coefficient Value",
    )
    fig.write_image(fig_filename_pattern.format(glm_family, n_neurons,
                                                n_trials, "png"))
    fig.write_html(fig_filename_pattern.format(glm_family, n_neurons,
                                               n_trials, "html"))
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
