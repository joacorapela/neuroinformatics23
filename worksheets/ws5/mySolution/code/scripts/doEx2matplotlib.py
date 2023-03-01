import sys
import argparse
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

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

    zmin = 0.0
    zmax = 5.0

    u, s, vh = np.linalg.svd(a=activity_arrayZ, full_matrices=False)

    plt.figure(1, figsize=(16, 5))
    plt.plot(times, u[:, 0])
    for i, response_time in enumerate(trials.response_times):
        plt.axvline(x=response_time)
    plt.xlabel(x_label)
    plt.ylabel("z-score")
    # fig.write_image(fig_filename_pattern.format(bin_size, "uFirstCol",  "png"))
    plt.savefig(fig_filename_pattern.format(bin_size, "uFirstCol", "png"))
    # fig.show()
    plt.show()

    argsort_res = np.argsort(vh[0, :])
    plt.figure(2, figsize=(16, 5))
    plt.imshow(activity_arrayZ[:, argsort_res].T, aspect='auto', vmin=zmin,
               vmax=zmax)
#     plt.xticks(np.range(activity_arrayZ.shape[0]), times)
#     plt.yticks(np.range(activity_arrayZ.shape[1]), clusters_ids)
#     trace = go.Heatmap(x=times, y=clusters_ids,
#                        z=activity_arrayZ[:, argsort_res].T,
#                        colorscale=colorscale, zmin=zmin, zmax=zmax)
#     fig.add_trace(trace)
    # fig.write_image(fig_filename_pattern.format(bin_size, "noZmax",  "png"))
#     fig.write_html(fig_filename_pattern.format(bin_size, "noZmax", "html"))
    plt.savefig(fig_filename_pattern.format(bin_size, "noZmax", "png"))
    # fig.show()

    plt.figure(3, figsize=(16, 5))
    plt.hist(x=vh[0, :])
    plt.xlabel("weight")
    plt.ylabel("counts")
    # fig.write_image(fig_filename_pattern.format(bin_size, "uFirstColWeigth",  "png"))
    plt.savefig(fig_filename_pattern.format(bin_size, "uFirstColWeight", "png"))
    # fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
