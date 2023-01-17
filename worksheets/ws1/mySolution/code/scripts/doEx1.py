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
    fig_filename_pattern = args.fig_filename_pattern

    p_values = [None] * n_repeats

    for i in range(n_repeats):
        if i%100 == 0:
            print(f"Processed {i} ({n_repeats})")
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
    count_p_values_less0_05 = np.sum(np.array(p_values)<0.05)

    fig = go.Figure()
    trace = go.Histogram(x=p_values)
    fig.add_trace(trace)
    fig.update_layout(xaxis_title="p value",
                      yaxis_title="count",
                      title=f"{count_p_values_less0_05} out of {n_repeats} "
                            "tests with p<0.05")
    fig.write_image(fig_filename_pattern.format(distribution, popmean, mean, n_samples, "png"))
    fig.write_html(fig_filename_pattern.format(distribution, popmean, mean, n_samples, "html"))

    # breakpoint()

if __name__ == "__main__":
    main(sys.argv)
