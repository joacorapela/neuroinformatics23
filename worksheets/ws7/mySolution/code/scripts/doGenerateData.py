
import sys
import argparse
import numpy as np
import plotly.graph_objects as go

import utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, help="Number of trials",
                        default=100)
    parser.add_argument("--n_neurons", type=int, help="Number of neurons",
                        default=30)
    parser.add_argument("--random_seed", type=int, help="Random seed",
                        default=1)
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default=("../../figures/counts_nNeurons_{:d}_"
                                 "nTrials_{:d}.{:s}"))
    parser.add_argument("--data_filename_pattern", type=str,
                        help="data filename pattern",
                        default=("../../results/counts_nNeurons_{:d}_"
                                 "nTrials_{:d}.npz"))
    args = parser.parse_args()

    n_trials = args.n_trials
    n_neurons = args.n_neurons
    random_seed = args.random_seed
    fig_filename_pattern = args.fig_filename_pattern
    data_filename = args.data_filename_pattern.format(n_neurons, n_trials)
    if random_seed > 0:
        np.random.seed(random_seed)
    count_data, trials_weights, neurons_weigts = \
        utils.simulatePoissonCountsInTrials(n_trials=n_trials,
                                            n_neurons=n_neurons)
    np.savez(data_filename, count_data=count_data,
             trials_weights=trials_weights, neurons_weigts=neurons_weigts)

    fig = go.Figure()
    trace = go.Heatmap(x=np.arange(n_neurons), y=np.arange(n_trials),
                       z=count_data)
    fig.add_trace(trace)
    fig.update_xaxes(title_text="Neuron")
    fig.update_yaxes(title_text="Trial")
    fig.write_image(fig_filename_pattern.format(n_neurons, n_trials, "png"))
    fig.write_html(fig_filename_pattern.format(n_neurons, n_trials, "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
