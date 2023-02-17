
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
    parser.add_argument("--selected_channels", type=str,
                        help="selected channels to plot",
                        default="[0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368]")
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
                        default="../../figures/lfp_selected_channels_{:s}_pid_{:s}.{:s}")
    args = parser.parse_args()

    pid = args.pid
    selected_channels = args.selected_channels[1:-1].split(",")
    start_time = args.start_time
    duration = args.duration
    plot_channel_spacing = args.plot_channel_spacing 
    x_label = args.x_label
    y_label = args.y_label
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    selected_channels_str = "_".join(selected_channels)
    selected_channels = [int(value) for value in selected_channels]
    title = title_pattern.format(pid)

    aOne = one.api.ONE(base_url="https://openalyx.internationalbrainlab.org",
                       password="international", silent=True)
    sr = brainbox.io.spikeglx.Streamer(pid=pid, one=aOne, remove_cached=False,
                                       typ="lf")
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
    fig = go.Figure()
    for i, selected_channel in enumerate(selected_channels):
        hovertext = [None] * len(times)
        for j, time in enumerate(times):
            hovertext[j] = f"channnel: {selected_channel}<br>lfp: {lfp[selected_channel, j]*1000} mv^2<br>loc: {channel_locs_acronyms[selected_channel]}<br>time: {time}"
        print(f"Processing channel {i} ({len(selected_channels)})")
        trace = go.Scatter(x=times, y=lfp[selected_channel, samples]*1000+i,
                           hoverinfo="text", text=hovertext,
                           name=f"{selected_channel} ({channel_locs_acronyms[selected_channel]})",
                          )
        fig.add_trace(trace)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label, showticklabels=False)
    fig.update_layout(title=title)

    fig.write_image(fig_filename_pattern.format(selected_channels_str, pid, "png"))
    fig.write_html(fig_filename_pattern.format(selected_channels_str, pid, "html"))

    # fig.show()

    # breakpoint()


if __name__ == "__main__":
   main(sys.argv)
