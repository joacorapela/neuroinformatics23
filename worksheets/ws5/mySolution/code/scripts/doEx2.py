import sys
import argparse
import numpy as np
import scipy.stats
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.colors

import one.api


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_size", type=float, help="bin_size", default=1.0)
    parser.add_argument("--experiment_id", type=str,
                        help="experiment to analyze",
                        default="ebe2efe3-e8a1-451a-8947-76ef42427cc9")
    parser.add_argument("--probe_id", type=str,
                        help="id of the probe to analyze",
                        default="probe00")
    parser.add_argument("--colorscale", type=str, help="colorscale",
                        default="viridis")
    parser.add_argument("--x_label", type=str, help="x_label",
                        default="Time (sec)")
    parser.add_argument("--y_label", type=str, help="x_label",
                        default="Cluster ID")
    parser.add_argument("--fig_width", type=int, help="figure width",
                        default=16)
    parser.add_argument("--fig_height", type=int, help="figure height",
                        default=5)
#     parser.add_argument("--zmin", type=float, help="min value to plot on z axis",
#                         default=0.0)
#     parser.add_argument("--zmax", type=int, help="max value to plot on z axis",
#                         default=5.0)
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/spikes_counts_svd_binSize_{:.02f}_{:s}.{:s}")
    args = parser.parse_args()

    bin_size = args.bin_size
    eID = args.experiment_id
    probe_id = args.probe_id
    colorscale = args.colorscale
    x_label = args.x_label
    y_label = args.y_label
    x_label = args.x_label
    y_label = args.y_label
    fig_width = args.fig_width
    fig_height = args.fig_height
#     zmin = args.zmin
#     zmax = args.zmax
    fig_filename_pattern = args.fig_filename_pattern

    aOne = one.api.ONE(base_url='https://openalyx.internationalbrainlab.org',
                       password='international', silent=True)
    spikes = aOne.load_object(eID, 'spikes', f'alf/{probe_id}/pykilosort')
    trials = aOne.load_object(eID, 'trials')

    spike_time_binary = np.floor(spikes.times/bin_size).astype(int)
    activity_array, edges = np.histogramdd(
        (spike_time_binary, spikes.clusters),
        bins=(spike_time_binary.max(), spikes.clusters.max()))
    activity_arrayZ = scipy.stats.zscore(activity_array)

    times = (edges[0][1:] + edges[0][:-1])/2.0*bin_size
    clusters_ids = edges[1][:-1]

    zmin, zmax = np.percentile(activity_arrayZ, q=(1.0, 99.0))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=zmin, vcenter=0, vmax=zmax)

    # a good start: blue to white to red colormap
    plt.set_cmap('bwr')

    [X, Y] = np.meshgrid(times, clusters_ids)

    u, s, vh = np.linalg.svd(a=activity_arrayZ, full_matrices=False)

    fig = go.Figure()
    trace = go.Scatter(x=times, y=u[:, 0])
    fig.add_trace(trace)
    for i, response_time in enumerate(trials.response_times):
        fig.add_vline(x=response_time)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text="z-score")
    fig.write_image(fig_filename_pattern.format(bin_size, "uFirstCol",  "png"))
    fig.write_html(fig_filename_pattern.format(bin_size, "uFirstCol", "html"))
    # fig.show()

    argsort_res = np.argsort(vh[0, :])

    plt.figure(figsize=(fig_width, fig_height))
    shw = plt.pcolor(X, Y, activity_arrayZ[:, argsort_res].T, norm=norm)
    bar = plt.colorbar(shw)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    bar.set_label("z score")
    plt.savefig(fig_filename_pattern.format(bin_size, "vh0Sorted",  "png"))
    # plt.show()

    fig = go.Figure()
    trace = go.Histogram(x=vh[0, :])
    fig.add_trace(trace)
    fig.update_xaxes(title_text="weight")
    fig.update_yaxes(title_text="counts")
    fig.write_image(fig_filename_pattern.format(bin_size, "uFirstColWeigth",  "png"))
    fig.write_html(fig_filename_pattern.format(bin_size, "uFirstColWeight", "html"))
    # fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
