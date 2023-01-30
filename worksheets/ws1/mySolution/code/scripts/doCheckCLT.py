import sys
import argparse
import numpy as np
import scipy.stats
import plotly.graph_objects as go

import stats


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", type=str, default="Rademacher",
                        help="distribution from which to generate samples")
    parser.add_argument("--normal_mean", type=float, default=0.0,
                        help="normal_mean of Normal distribution")
    parser.add_argument("--normal_std", type=float, default=1.0,
                        help="standard deviation of Normal distribution")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="number of samples of Noral distribution "
                             "to generate per repeat")
    parser.add_argument("--n_repeats", type=int, default=10000,
                        help="number of repeats")
    parser.add_argument("--n_bins", type=int, default=100,
                        help="number of bins for the p-values histogram")
    parser.add_argument("--n_samples_pdf", type=int, default=1000,
                        help="number of bins for the p-values histogram")
    parser.add_argument("--n_stds_xaxis", type=int, default=10,
                        help="number of standard deviation away from the mean to include in the x-axis")
    parser.add_argument("--fig_filename_pattern", type=str,
                        default="../../figures/checkCLT_distribution{:s}_nSamples{:d}.{:s}",
                        help="figure filename pattern")
    args = parser.parse_args()

    distribution = args.distribution
    normal_mean = args.normal_mean
    normal_std = args.normal_std
    n_samples = args.n_samples
    n_repeats = args.n_repeats
    n_bins = args.n_bins
    n_samples_pdf = args.n_samples_pdf
    n_stds_xaxis = args.n_stds_xaxis
    fig_filename_pattern = args.fig_filename_pattern

    means = [None] * n_repeats
    for i in range(n_repeats):
        if i%1000 == 0:
            print(f"Processing repeat {i}")
        if distribution == "Normal":
            sample = stats.sample(distribution=distribution,
                                  normal_mean=normal_mean,
                                  normal_std=normal_std,
                                  n_samples=n_samples)
        elif distribution == "Rademacher":
            sample = stats.sample(distribution=distribution,
                                  n_samples=n_samples)
            pop_mean = 0.0
            pop_std = 1.0
        elif distribution == "StdCauchy":
            sample = stats.sample(distribution=distribution,
                                  n_samples=n_samples)
        else:
            raise ValueError(f"Invalid distribution {distribution}")
        means[i] = np.mean(sample)

    if distribution == "Normal":
        pop_mean = normal_mean
        pop_std = normal_std
    elif distribution == "Rademacher":
        pop_mean = 0.0
        pop_std = 1.0
    elif distribution == "StdCauchy":
        pop_mean = 0.0
        pop_std = np.std(means)
        # pop_std = scipy.stats.median_abs_deviation(means)
    else:
        raise ValueError(f"Invalid distribution {distribution}")

    clt_x = np.linspace(pop_mean-n_stds_xaxis*pop_std/np.sqrt(n_samples), pop_mean+n_stds_xaxis*pop_std/np.sqrt(n_samples), n_samples_pdf)
    clt_y = scipy.stats.norm.pdf(x=clt_x, loc=pop_mean, scale=pop_std/np.sqrt(n_samples))
    if distribution == "StdCauchy":
        stdCauchy_y = scipy.stats.cauchy.pdf(x=clt_x)

    empirical_bins = np.linspace(pop_mean-n_stds_xaxis*pop_std/np.sqrt(n_samples), pop_mean+n_stds_xaxis*pop_std/np.sqrt(n_samples), n_bins)
    empirical_y, _ = np.histogram(means, bins=empirical_bins, density=True)

    empirical_bins_min = np.min(empirical_bins)
    empirical_bins_max = np.max(empirical_bins)
    empirical_x = np.linspace(empirical_bins_min+(empirical_bins[1]-empirical_bins[0])/2,
                              empirical_bins_max, len(empirical_bins)-1)

    fig = go.Figure()

    trace = go.Bar(x=empirical_x, y=empirical_y, name="Empirical PDF")
    fig.add_trace(trace)

    trace = go.Scatter(x=clt_x, y=clt_y, name="CLT PDF")
    fig.add_trace(trace)

    if distribution == "StdCauchy":
        trace = go.Scatter(x=clt_x, y=stdCauchy_y, name="Std Cauchy PDF")
        fig.add_trace(trace)

    title = f"Distribution: {distribution}"
    fig.update_layout(xaxis_title="mean", yaxis_title="pdf(mean)", title=title)

    fig.write_image(fig_filename_pattern.format(distribution, n_samples, "png"))
    fig.write_html(fig_filename_pattern.format(distribution, n_samples, "html"))

    fig.show()

    # breakpoint()

if __name__ == "__main__":
    main(sys.argv)
