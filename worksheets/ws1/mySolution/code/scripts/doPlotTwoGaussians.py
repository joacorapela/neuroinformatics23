import sys
import argparse
import math
import numpy as np
import scipy.stats
import plotly.graph_objs as go

import stats
import plots


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean0", type=float, default="0.0",
                        help="mean of the first Gaussian")
    parser.add_argument("--mean1", type=float, default="2.5",
                        help="mean of the second Gaussian")
    parser.add_argument("--std", type=float, default=10.0,
                        help="standard deviation of both Gaussians")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="critical value")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="number of samples")
    parser.add_argument("--xmin", type=float, default=-3.00,
                        help="minimum x value")
    parser.add_argument("--xmax", type=float, default=5.50,
                        help="maximum x value")
    parser.add_argument("--xdt", type=float, default=0.01,
                        help="dt for x")
    parser.add_argument("--fig_filename_pattern", type=str,
                        default="../../figures/two_gaussians_mean0{:.04f}_mean1{:.04f}_std{:.04f}_alpha{:.04f}_nSamples{:d}.{:s}",
                        help="figure filename pattern")
    args = parser.parse_args()

    mean0 = args.mean0
    mean1 = args.mean1
    std = args.std
    alpha = args.alpha
    n_samples = args.n_samples
    xmin = args.xmin
    xmax = args.xmax
    xdt = args.xdt
    fig_filename_pattern = args.fig_filename_pattern

    x = np.arange(xmin, xmax, xdt)
    gauss0 = scipy.stats.norm.pdf(x, loc=mean0, scale=std/math.sqrt(n_samples))
    gauss1 = scipy.stats.norm.pdf(x, loc=mean1, scale=std/math.sqrt(n_samples))
    z = scipy.stats.norm.ppf(1-alpha)

    fig = go.Figure()
    trace0 = go.Scatter(x=x, y=gauss0, mode="lines", name=f"Gaussian mean={mean0}")
    trace1 = go.Scatter(x=x, y=gauss1, mode="lines", name=f"Gaussian mean={mean1}")
    fig.add_trace(trace0)
    fig.add_trace(trace1)
    fig.add_vline(x=std/math.sqrt(n_samples)*z, line={"dash": "dash"})
    fig.update_xaxes(title_text="x&#772;", range=[xmin, xmax])
    fig.update_yaxes(title_text="pdf(x&#772;)")
    fig.update_layout(title=f"pdf of x&#772; with n_samples={n_samples}", font={"size": 12})

    png_filename = fig_filename_pattern.format(mean0, mean1, std, alpha, n_samples, "png")
    html_filename = fig_filename_pattern.format(mean0, mean1, std, alpha, n_samples, "html")
    fig.write_image(png_filename)
    fig.write_html(html_filename)

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
