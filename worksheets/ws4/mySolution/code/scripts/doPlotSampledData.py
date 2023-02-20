
import sys
import argparse
import numpy as np
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_samples", type=int, help="number of samples",
                        default=1000)
    parser.add_argument("--kappa", type=float, help="kappa", default=1.0)
    parser.add_argument("--loc_intercept", type=float,
                        help="location intercept coefficient", default=0.0)
    parser.add_argument("--loc_slope", type=float,
                        help="location slope coefficient", default=2*np.pi)
    parser.add_argument("--x_label", type=str, help="x_label",
                        default="x")
    parser.add_argument("--y_label", type=str, help="y_label",
                        default="theta")
    parser.add_argument("--title_pattern", type=str, help="title",
                        default="n_samples {:d}")
    parser.add_argument("--results_filename_pattern", type=str,
                        help="results filename pattern",
                        default="../../results/data_nSamples_{:d}_kappa_{:.02f}_locIntercept_{:.02f}_locSlope_{:02f}.csv")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/data_nSamples_{:d}_kappa_{:.02f}_locIntercept_{:.02f}_locSlope_{:02f}.{:s}")
    args = parser.parse_args()

    n_samples = args.n_samples
    kappa = args.kappa
    loc_intercept = args.loc_intercept
    loc_slope = args.loc_slope
    x_label = args.x_label
    y_label = args.y_label
    title_pattern = args.title_pattern
    results_filename_pattern = args.results_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    results_filename = results_filename_pattern.format(
        n_samples, kappa, loc_intercept, loc_slope)
    results = np.genfromtxt(results_filename)
    x = results[:, 0]
    theta = results[:, 1]

    title = title_pattern.format(n_samples)
    fig = go.Figure()
    trace = go.Scatter(x=np.concatenate([x, x]),
                       y=np.concatenate([theta, theta+2*np.pi]),
                       mode="markers")
    fig.add_trace(trace)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(title=title)

    fig.write_html(fig_filename_pattern.format(
        n_samples, kappa, loc_intercept, loc_slope, "html"))
    fig.write_image(fig_filename_pattern.format(
        n_samples, kappa, loc_intercept, loc_slope, "png"))

    # fig.show()

    # breakpoint()


if __name__ == "__main__":
    main(sys.argv)
