
import sys
import pickle
import argparse
import numpy as np
import sklearn.linear_model
import plotly.graph_objects as go

import utils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, help="Number of trials",
                        default=100)
    parser.add_argument("--n_neurons", type=int, help="Number of neurons",
                        default=30)
    parser.add_argument("--target_neuron_index", type=int,
                        help="target neuron index", default=29)
    parser.add_argument("--predict_on_test_data", action="store_true",
                        help=("use this flag to run the predictions on test "
                              "data"))
    parser.add_argument("--glm_family", type=str,
                        help="distribution family used in GLM",
                        default="Poisson")
    parser.add_argument("--data_filename_pattern", type=str,
                        help="data filename pattern",
                        default=("../../results/counts_nNeurons_{:d}_"
                                 "nTrials_{:d}.npz"))
    parser.add_argument("--model_filename_pattern", type=str,
                        help="model filename pattern",
                        default=("../../results/model{:s}_target_{:d}_"
                                 "nNeurons_{:d}_nTrials_{:d}.pickle"))
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default=("../../figures/predicted_counts_"
                                 "{:s}_predictOnTest_{:d}_nNeurons_{:d}_nTrials_{:d}.{:s}"))
    args = parser.parse_args()

    n_trials = args.n_trials
    n_neurons = args.n_neurons
    target_neuron_index = args.target_neuron_index
    predict_on_test_data = args.predict_on_test_data 
    glm_family = args.glm_family
    data_filename = args.data_filename_pattern.format(n_neurons, n_trials)
    model_filename = args.model_filename_pattern.format(glm_family,
                                                        target_neuron_index,
                                                        n_neurons, n_trials)
    fig_filename_pattern = args.fig_filename_pattern

    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    neurons_indices = np.arange(n_neurons)
    if predict_on_test_data:
        count_data, trials_weights, neurons_weigts = \
            utils.simulatePoissonCountsInTrials(n_trials=n_trials,
                                                n_neurons=n_neurons)
        X = count_data[:, neurons_indices != target_neuron_index]
        y = count_data[:, target_neuron_index]
    else:
        load_res = np.load(data_filename)
        count_data = load_res["count_data"]
        X = count_data[:, neurons_indices != target_neuron_index]
        y = count_data[:, target_neuron_index]

    y_pred = model.predict(X)
    mse = np.mean((y-y_pred)**2)
    mean_pred_data = {i: y_pred[y == i].mean() for i in np.unique(y)}

    fig = go.Figure()
    trace = go.Scatter(x=y, y=y_pred, mode="markers", name="predictions")
    fig.add_trace(trace)
    trace = go.Scatter(x=list(mean_pred_data.keys()),
                       y=list(mean_pred_data.values()),
                       mode="lines+markers", name="means")
    fig.add_trace(trace)
    trace = go.Scatter(x=y, y=y, mode="lines",
                       line=dict(color="gray", dash="dash"),
                       showlegend=False)
    fig.add_trace(trace)

    fig.update_xaxes(title_text="Observed Spike Count")
    fig.update_yaxes(title_text="Predicted Spike Count")
    fig.update_layout(title=f"Mean squared error: {mse:.02f}")
    fig.write_image(fig_filename_pattern.format(glm_family,
                                                predict_on_test_data,
                                                n_neurons, n_trials, "png"))
    fig.write_html(fig_filename_pattern.format(glm_family,
                                               predict_on_test_data,
                                               n_neurons, n_trials, "html"))


if __name__ == "__main__":
    main(sys.argv)
