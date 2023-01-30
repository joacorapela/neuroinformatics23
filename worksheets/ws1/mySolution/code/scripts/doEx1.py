import sys
import argparse
import numpy as np

import stats
import plots


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", type=str, default="Normal",
                        help="distribution from which to generate samples")
    parser.add_argument("--popmean", type=float, default=0.0,
                        help="population mean")
    parser.add_argument("--normal_mean", type=float, default=0.0,
                        help="normal_mean of Normal distribution")
    parser.add_argument("--normal_std", type=float, default=1.0,
                        help="standard deviation of Normal distribution")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="number of samples of Noral distribution "
                             "to generate per repeat")
    parser.add_argument("--n_repeats", type=int, default=1000,
                        help="number of repeats")
    parser.add_argument("--n_bins", type=int, default=20,
                        help="number of bins for the p-values histogram")
    parser.add_argument("--fig_filename_pattern", type=str,
                        default="../../figures/ex1_distribution{:s}_popmean{:.4f}_mean{:.4f}_nSamples{:d}.{:s}",
                        help="figure filename pattern")
    args = parser.parse_args()

    distribution = args.distribution
    popmean = args.popmean
    normal_mean = args.normal_mean
    normal_std = args.normal_std
    n_samples = args.n_samples
    n_repeats = args.n_repeats
    n_bins = args.n_bins
    fig_filename_pattern = args.fig_filename_pattern

    all_std_errors = []
    p_values_hist, bins_centers, count_p_values_less0_05, std_errors = \
        stats.get_pvalues_hist(distribution=distribution,
                               normal_mean=normal_mean,
                               normal_std=normal_std,
                               n_repeats=n_repeats,
                               popmean=popmean,
                               n_bins=n_bins)
    all_std_errors.append(std_errors)

    mean_all_std_errors = np.mean(all_std_errors)
    std_all_std_errors = np.std(all_std_errors)
    title = f"{count_p_values_less0_05} out of {n_repeats} tests with p<0.05, std_error={mean_all_std_errors}&#177;{std_all_std_errors}"
    fig = plots.getPlotHistPValues(bins_centers=bins_centers, p_values_hist=p_values_hist, title=title)
    fig.write_image(fig_filename_pattern.format(distribution, popmean, normal_mean, n_samples, "png"))
    fig.write_html(fig_filename_pattern.format(distribution, popmean, normal_mean, n_samples, "html"))
    # breakpoint()

if __name__ == "__main__":
    main(sys.argv)
