import sys
import argparse
import numpy as np
import scipy.stats
import plotly.graph_objects as go

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
                        default="../../figures/ex1_distribution{:s}_popmean{:.4f}_mean{:.4f}_nSamples{:d}.{:s}",
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

    bins = np.arange(0, 1+1.0/n_bins, 1.0/n_bins)
    p_values_hist_resamples = np.empty((n_resamples, n_bins), dtype=np.double)

    for n in range(n_resamples):
        print("Processed resample {:03d} ({:d})".format(n, n_resamples))
        p_values = [None] * n_repeats

        for i in range(n_repeats):
            if distribution == "Normal":
                sample = np.random.normal(loc=mean, scale=std, size=n_samples)
            elif distribution == "StdCauchy":
                sample = np.random.normal(size=n_samples)
            elif distribution == "Rademacher":
                uniforms = np.random.uniform(size=n_samples)
                sample = np.where(uniforms<0.5, 1, -1)
            elif distribution == "VerySkewed":
                uniforms = np.random.uniform(size=n_samples)
                sample = np.where(uniforms<1e-3, 1, 0)
            else:
                raise ValueError(f"Invalid distribution: {distribution}")
            _, p_values[i] = scipy.stats.ttest_1samp(sample, popmean=popmean)
        p_values_hist_resamples[n, :], _ = np.histogram(p_values, bins=bins)
    p_values_hist_means = np.mean(p_values_hist_resamples, axis=0)
    p_values_hist_stds = np.std(p_values_hist_resamples, axis=0)
    bin_centers = np.arange(1/n_bins, 1.05, 1.0/n_bins)

    fig = go.Figure()
    trace = go.Bar(x=bin_centers, y=p_values_hist_means, error_y=dict(type="data", array=1.96*p_values_hist_stds))
    fig.add_trace(trace)
    fig.update_layout(xaxis_title="p value", yaxis_title="count")
    fig.write_image(fig_filename_pattern.format(distribution, popmean, mean, n_samples, "png"))
    fig.write_html(fig_filename_pattern.format(distribution, popmean, mean, n_samples, "html"))

    # breakpoint()

if __name__ == "__main__":
    main(sys.argv)
