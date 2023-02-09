
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.signal
import plotly.graph_objects as go

import one.api
import brainbox.io.one
import brainbox.io.spikeglx


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--pid", type=str, help="probed ID",
                        default="38124fca-a0ac-4b58-8e8b-84a2357850e6")
    parser.add_argument("--start_time", type=float,
                        help="plotting start time (sec)", default=0.0)
    parser.add_argument("--duration", type=float,
                        help="plotting duration (sec)", default=600.0)
    parser.add_argument("--segment_length", type=int,
                        help="Welch segement length", default=2048)
    parser.add_argument("--n_ticks", type=int,
                        help="nunmber of ticks in colorbar", default=6)
    parser.add_argument("--colorscale", type=str, help="colorscale",
                        default="viridis")
    parser.add_argument("--x_label", type=str, help="x_label",
                        default="Frequency")
    parser.add_argument("--y_label", type=str, help="x_label",
                        default="Channel")
    parser.add_argument("--xlim", type=str, help="limits of x-axis (Hz)",
                        default="[0,200]")
    parser.add_argument("--title_pattern", type=str, help="title",
                        default="probe ID: {:s}, segment length: {:d}")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/welchPxx_segmentLength_{:d}_pid_{:s}.{:s}")
    args = parser.parse_args()

    pid = args.pid
    start_time = args.start_time
    duration = args.duration
    segment_length = args.segment_length
    n_ticks = args.n_ticks
    colorscale = args.colorscale
    x_label = args.x_label
    y_label = args.y_label
    xlim = [float(str) for str in args.xlim[1:-1].split(',')]
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    title = title_pattern.format(pid, segment_length)
    band = "lf"

    aOne = one.api.ONE(base_url="https://openalyx.internationalbrainlab.org",
                       password="international", silent=True)
    sr = brainbox.io.spikeglx.Streamer(pid=pid, one=aOne, remove_cached=False,
                                       typ=band)

    # extract channel location acronyms for hover
    eID, probe_label = aOne.pid2eid(pid=pid)
    els = brainbox.io.one.load_channel_locations(eID, one=aOne)
    channel_locs_acronyms = els[probe_label]["acronym"]

    s0 = start_time * sr.fs
    tsel = slice(int(s0), int(s0) + int(duration * sr.fs))

    # Important: remove sync channel from raw data, and transpose
    lfp = sr[tsel, :-sr.nsync].T
    n_channels, n_samples = lfp.shape
    print(f"Data has {n_channels} channels and {n_samples} samples")

    channels = np.arange(0, n_channels, 1)
    pxx_list = [None] * len(channels)
    for i, channel in enumerate(channels):
        print(f"Processing {i} ({len(channels)})")
        f, pxx_list[i] = scipy.signal.welch(x=lfp[channel, :]*1000, fs=sr.fs,
                                            nperseg=segment_length,
                                            scaling="spectrum")
    pxx_matrix = np.stack(pxx_list)
    zmin = np.min(pxx_matrix)
    zmax = np.max(pxx_matrix)

    # let's plot now
    def colorbar(zmin, zmax, n_ticks, title):
        ticktext = ["{:.2E}".format(val) for val in 10**np.linspace(np.log10(zmin), np.log10(zmax), n_ticks)]
        answer = dict(
            title = title,
            tickmode = "array",
            tickvals = np.linspace(np.log10(zmin), np.log10(zmax), n_ticks),
            ticktext = ticktext
        )
        return answer

    hovertext = []
    for yi, yy in enumerate(channels):
        hovertext.append([])
        for xi, xx in enumerate(f):
            hovertext[-1].append(f"frequency: {xx}<br />channel: {yy}<br />Pxx: {pxx_matrix[yi][xi]}<br />loc: {channel_locs_acronyms[yi]}")

    fig = go.Figure()
    trace = go.Heatmap(x=f, y=channels, z=np.log10(pxx_matrix), colorscale=colorscale, zmin=np.log10(zmin), zmax=np.log10(zmax), colorbar=colorbar(zmin=zmin, zmax=zmax, n_ticks=n_ticks, title="Power (mv^2)"), hoverinfo="text", text=hovertext)
    fig.add_trace(trace)
    fig.update_xaxes(title_text=x_label, range=xlim)
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(title=title)

    fig.write_image(fig_filename_pattern.format(segment_length, pid, "png"))
    fig.write_html(fig_filename_pattern.format(segment_length, pid, "html"))

    fig.show()

    breakpoint()


if __name__ == "__main__":
   main(sys.argv)
