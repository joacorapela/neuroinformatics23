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
    parser.add_argument("--mean", type=float, default=0.0,
                        help="mean of Normal distribution")
    parser.add_argument("--std", type=float, default=1.0,
                        help="standard deviation of Normal distribution")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="number of samples of Noral distribution "
                             "to generate per repeat")
    parser.add_argument("--n_repeats", type=int, default=1000,
                        help="number of repeats")
    parser.add_argument("--n_resamples", type=int, default=100,
                        help="number of resamples used to estimate histograms "
                             "means and stds")
    parser.add_argument("--n_bins", type=int, default=20,
                        help="number of bins for the p-values histogram")
    parser.add_argument("--fig_filename_pattern", type=str,
                        default="../../figures/ex1_distribution{:s}_popmean{:.4f}_mean{:.4f}_nSamples{:d}_withErrorBars.{:s}",
                        help="figure filename pattern")
    args = parser.parse_args()

    distribution = args.distribution
    popmean = args.popmean
    mean = args.mean
    std = args.std
    n_samples = args.n_samples
    n_repeats = args.n_repeats
    n_resamples = args.n_resamples
    n_bins = args.n_bins
    fig_filename_pattern = args.fig_filename_pattern

    p_values_hist_resamples = np.empty((n_resamples, n_bins), dtype=np.double)
    count_p_values_less0_05_resamples = np.empty(n_resamples, dtype=np.int)
    for n in range(n_resamples):
        print("Processed resample {:03d} ({:d})".format(n, n_resamples))
        p_values_hist_resamples[n, :], bins_centers, count_p_values_less0_05_resamples[n] = \
            stats.get_pvalues_hist(distribution=distribution,
                                   mean=mean, std=std,
                                   n_samples=n_samples,
                                   n_repeats=n_repeats,
                                   popmean=popmean,
                                   n_bins=n_bins)
    p_values_hist_means = np.mean(p_values_hist_resamples, axis=0)
    p_values_hist_stds = np.std(p_values_hist_resamples, axis=0)
    count_p_values_less0_05_mean = np.mean(count_p_values_less0_05_resamples)
    count_p_values_less0_05_std = np.std(count_p_values_less0_05_resamples)

    title="{:.02f}&#177;{:.02f} out of {:d} tests with p<0.05, n_samples={:d}".format(count_p_values_less0_05_mean, 1.96*count_p_values_less0_05_std, n_repeats, n_samples)
    fig = plots.getPlotHistPValues(bins_centers=bins_centers, p_values_hist=p_values_hist_means, title=title, errors=1.96*p_values_hist_stds)
    fig.write_image(fig_filename_pattern.format(distribution, popmean, mean, n_samples, "png"))
    fig.write_html(fig_filename_pattern.format(distribution, popmean, mean, n_samples, "html"))
    # breakpoint()


if __name__ == "__main__":
    main(sys.argv)
