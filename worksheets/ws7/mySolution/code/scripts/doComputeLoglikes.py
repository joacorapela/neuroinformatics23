
import sys
import argparse
import numpy as np
import scipy.stats
import sklearn.linear_model
import plotly.graph_objects as go

import utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, help="Number of trials",
                        default=1000)
    parser.add_argument("--n_neurons", type=int, help="Number of neurons",
                        default=30)
    parser.add_argument("--n_repeats", type=int, help="Number of repeats",
                        default=1000)
    parser.add_argument("--cross_validated", action="store_true",
                        help=("Use this option to calculated cross-validated "
                              "log likelihoods"))
    parser.add_argument("--target_neuron_index", type=int,
                        help="target neuron index", default=29)
    parser.add_argument("--data_filename_pattern", type=str,
                        help="data filename pattern",
                        default=("../../results/counts_nNeurons_{:d}_"
                                 "nTrials_{:d}.npz"))
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default=("../../figures/logLikes{:s}_"
                                 "crossValidated_{:d}_"
                                 "target_{:d}_"
                                 "nNeurons_{:d}_nTrials_{:d}_"
                                 "nRepeats_{:d}.{:s}"))
    args = parser.parse_args()

    n_trials = args.n_trials
    n_neurons = args.n_neurons
    n_repeats = args.n_repeats
    cross_validated = args.cross_validated
    target_neuron_index = args.target_neuron_index
    data_filename = args.data_filename_pattern.format(n_neurons, n_trials)
    fig_filename_pattern = args.fig_filename_pattern 

    load_res = np.load(data_filename)
    count_data = load_res["count_data"]
    neurons_indices = np.arange(n_neurons)
    X = count_data[:, neurons_indices != target_neuron_index]
    y = count_data[:, target_neuron_index]

    poisson_model = sklearn.linear_model.PoissonRegressor(alpha=0.0)
    gaussian_model = sklearn.linear_model.LinearRegression()
    poisson_model.fit(X=X, y=y)
    gaussian_model.fit(X=X, y=y)

    poisson_loglikes = [None] * n_repeats
    gaussian_loglikes = [None] * n_repeats
    for i in range(n_repeats):
        print(f"Processing repeat {i} ({n_repeats})")
        count_data, _, _ = \
            utils.simulatePoissonCountsInTrials(n_trials=n_trials,
                                                n_neurons=n_neurons)
        X = count_data[:, neurons_indices != target_neuron_index]
        y = count_data[:, target_neuron_index]
        if not cross_validated:
            poisson_model.fit(X=X, y=y)
            gaussian_model.fit(X=X, y=y)
        poisson_loglikes[i] = utils.computePoissonLoglike(
            intercept=poisson_model.intercept_, coefs=poisson_model.coef_,
            y=y, X=X)
        gaussian_loglikes[i] = utils.computeGaussianLoglike(
            intercept=gaussian_model.intercept_, coefs=gaussian_model.coef_,
            y=y, X=X)
    result = scipy.stats.ttest_rel(gaussian_loglikes, poisson_loglikes)

    title = f"Poisson mean loglike: {np.mean(poisson_loglikes):.6f}. Gaussian mean loglike: {np.mean(gaussian_loglikes):.6f} <br> t-test: statistic {result.statistic:.6f}, p-value {result.pvalue:.6f}"
    fig = go.Figure()
    trace = go.Scatter(x=poisson_loglikes, y=gaussian_loglikes, mode="markers",
                      showlegend=False)
    fig.add_trace(trace)
    fig.update_layout(
        title=title,
        xaxis_title="Poisson Log Likelihood",
        yaxis_title="Gaussian Log Likelihood",
    )
    trace = go.Scatter(x=poisson_loglikes, y=poisson_loglikes, mode="lines",
                       line=dict(color="gray", dash="dash"),
                       showlegend=False)
    fig.add_trace(trace)
    fig.write_image(fig_filename_pattern.format("Scatter", cross_validated,
                                                target_neuron_index, n_neurons,
                                                n_trials, n_repeats, "png"))
    fig.write_html(fig_filename_pattern.format("Scatter", cross_validated,
                                               target_neuron_index, n_neurons,
                                               n_trials, n_repeats, "html"))
    fig = go.Figure()
    trace = go.Histogram(x=poisson_loglikes, name="Poisson")
    fig.add_trace(trace)
    trace = go.Histogram(x=gaussian_loglikes, name="Gaussian")
    fig.add_trace(trace)
    fig.update_layout(
        title=title,
        xaxis_title="Log Likelihoods",
        yaxis_title="Counts",
        bargap=0.2, # gap between bars of adjacent location coordinates
        bargroupgap=0.1 # gap between bars of the same location coordinates
    )
    fig.write_image(fig_filename_pattern.format("Hist", cross_validated,
                                                target_neuron_index, n_neurons,
                                                n_trials, n_repeats, "png"))
    fig.write_html(fig_filename_pattern.format("Hist", cross_validated,
                                               target_neuron_index, n_neurons,
                                               n_trials, n_repeats, "html"))
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
