
import sys
import argparse
import numpy as np
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
                        help="plotting duration (sec)", default=5.0)
    parser.add_argument("--plot_channel_spacing", type=int,
                        help="spacing between channels in plot", default=16)
    parser.add_argument("--x_label", type=str, help="x_label",
                        default="Time (sec)")
    parser.add_argument("--y_label", type=str, help="x_label",
                        default="LFP (mV)")
    parser.add_argument("--title_pattern", type=str, help="title",
                        default="probe ID: {:s}")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/lfp_selected_channels_pid_{:s}.{:s}")
    args = parser.parse_args()

    from_time_lfp = 0.0
    to_time_lfp = 5.0
    channel_selection_spacing = 16

    pid = args.pid
    start_time = args.start_time
    duration = args.duration
    plot_channel_spacing = args.plot_channel_spacing 
    x_label = args.x_label
    y_label = args.y_label
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    title = title_pattern.format(pid)
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

    samples = np.arange(start_time*sr.fs, (start_time+duration)*sr.fs,
                        dtype=int)
    times = samples/sr.fs
    selected_channels = np.arange(0, n_channels, plot_channel_spacing)
    fig = go.Figure()
    for i, selected_channel in enumerate(selected_channels):
        hovertext = [None] * len(times)
        for j, time in enumerate(times):
            hovertext[j] = f"channnel: {selected_channel}<br>lfp: {lfp[selected_channel, j]*1000} mv^2<br>loc: {channel_locs_acronyms[selected_channel]}<br>time: {time}"
        print(f"Processing channel {i} ({len(selected_channels)})")
        trace = go.Scatter(x=times, y=lfp[selected_channel, samples]*1000+i,
                           hoverinfo="text", text=hovertext,
                           name=f"{selected_channel} {channel_locs_acronyms[selected_channel]}")
        fig.add_trace(trace)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label, showticklabels=False)
    fig.update_layout(title=title)

    fig.write_image(fig_filename_pattern.format(pid, "png"))
    fig.write_html(fig_filename_pattern.format(pid, "html"))

    fig.show()

    breakpoint()


if __name__ == "__main__":
   main(sys.argv)
