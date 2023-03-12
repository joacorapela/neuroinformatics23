
import sys
import argparse
import numpy as np
import sklearn.linear_model
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
                                 "{:s}_nNeurons_{:d}_nTrials_{:d}.{:s}"))
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
    if glm_family == "Poisson":
        model = sklearn.linear_model.PoissonRegressor(alpha=0.0)
    elif glm_family == "Gaussian":
        model = sklearn.linear_model.LinearRegression()
    X = count_data[:, neurons_indices != target_neuron_index]
    y = count_data[:, target_neuron_index]
    model.fit(X=X, y=y)
    y_pred = model.predict(X)
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
    fig.write_image(fig_filename_pattern.format(glm_family, n_neurons,
                                                n_trials, "png"))
    fig.write_html(fig_filename_pattern.format(glm_family, n_neurons,
                                               n_trials, "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
