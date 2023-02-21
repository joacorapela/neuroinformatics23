
import sys
import argparse
import pickle
import numpy as np
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_resamples", type=int,
                        help="number of resamples for permutation test",
                        default=100)
    parser.add_argument("--n_samples", type=int, help="number of samples",
                        default=1000)
    parser.add_argument("--kappa", type=float, help="kappa", default=1.0)
    parser.add_argument("--loc_intercept", type=float,
                        help="location intercept coefficient", default=0.0)
    parser.add_argument("--loc_slope", type=float,
                        help="location slope coefficient", default=2*np.pi)
    parser.add_argument("--percentile_percent", type=float,
                        help=("percent of the percentile to be calculated "
                              "(e.g., 95.0)"), default=95.0)
    parser.add_argument("--n_bins", type=int, help="number of bins",
                        default=20)
    parser.add_argument("--results_filename_pattern", type=str,
                        help="results filename pattern",
                        default=("../../results/lwllPermutationTest_"
                                 "nSamples_{:d}_kappa_{:.02f}_"
                                 "locIntercept_{:.02f}_"
                                 "locSlope_{:02f}_nResamples_{:d}.pickle"))
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default=("../../figures/lwllPermutationTest_"
                                 "nSamples_{:d}_kappa_{:.02f}_"
                                 "locIntercept_{:.02f}_"
                                 "locSlope_{:02f}_nResamples_{:d}_"
                                 "nBins_{:d}.{:s}"))
    args = parser.parse_args()

    n_resamples = args.n_resamples
    n_samples = args.n_samples
    kappa = args.kappa
    loc_intercept = args.loc_intercept
    loc_slope = args.loc_slope
    percentile_percent = args.percentile_percent
    n_bins = args.n_bins
    results_filename_pattern = args.results_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    results_filename = results_filename_pattern.format(n_resamples, kappa,
                                                       loc_intercept,
                                                       loc_slope, n_resamples)
    with open(results_filename, "rb") as f:
        load_res = pickle.load(f)

    shuffled_stat_values = load_res["shuffled_stat_values"]
    obs_stat_value = load_res["obs_stat_value"]
    p_value = load_res["p_value"]
    percentile_value = np.percentile(shuffled_stat_values, percentile_percent)

    percentile_1 = np.percentile(shuffled_stat_values, 1)
    percentile_99 = np.percentile(shuffled_stat_values, 99)
    bin_size = (percentile_99 - percentile_1)/n_bins

    title = f"p_value: {p_value}"
    x_label = "locally-weighted log likelihood"
    y_label = "count"

    fig = go.Figure()
    trace = go.Histogram(x=shuffled_stat_values,
                         xbins=dict(start=percentile_1, end=percentile_99,
                                    size=bin_size))
    fig.add_trace(trace)
    fig.add_vline(x=obs_stat_value, annotation_text="observed")
    fig.add_vline(x=percentile_value,
                  annotation_text=f"{percentile_percent}% percentile")
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(title=title)

    fig.write_html(fig_filename_pattern.format(
        n_samples, kappa, loc_intercept, loc_slope, n_resamples, n_bins,
        "html"))
    fig.write_image(fig_filename_pattern.format(
        n_samples, kappa, loc_intercept, loc_slope, n_resamples, n_bins,
        "png"))
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
