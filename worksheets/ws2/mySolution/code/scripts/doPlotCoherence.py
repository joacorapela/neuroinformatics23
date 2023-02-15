
import sys
import argparse
import numpy as np
import scipy.signal
import plotly.graph_objects as go

import one.api
import brainbox.io.one
import brainbox.io.spikeglx

import plotUtils


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--pid", type=str, help="probed ID",
                        default="38124fca-a0ac-4b58-8e8b-84a2357850e6")
    parser.add_argument("--channel_nro", type=int,
                        help="channel number for coherence calculation",
                        default=250)
    parser.add_argument("--start_time", type=float,
                        help="plotting start time (sec)", default=0.0)
    parser.add_argument("--duration", type=float,
                        help="signal duration (sec)", default=10.0)
    parser.add_argument("--segment_len", type=int,
                        help="Segement length (samples)", default=-1)
    parser.add_argument("--n_ticks", type=int,
                        help="nunmber of ticks in colorbar", default=6)
    parser.add_argument("--colorscale", type=str, help="colorscale",
                        default="viridis")
    parser.add_argument("--x_label", type=str, help="x_label",
                        default="Frequency (Hz)")
    parser.add_argument("--y_label", type=str, help="y_label",
                        default="Channel Number")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default=("../../figures/coherence_startTime_{:.02f}_"
                                 "duration_{:.02f}_segmentLen_{:d}_"
                                 "pid_{:s}_channel_{:d}.{:s}"))
    args = parser.parse_args()

    pid = args.pid
    channel_nro = args.channel_nro
    start_time = args.start_time
    duration = args.duration
    segment_len = args.segment_len if args.segment_len > 0 else None
    n_ticks = args.n_ticks
    colorscale = args.colorscale
    x_label = args.x_label
    y_label = args.y_label
    fig_filename_pattern = args.fig_filename_pattern

    aOne = one.api.ONE(base_url="https://openalyx.internationalbrainlab.org",
                       password="international", silent=True)
    sr = brainbox.io.spikeglx.Streamer(pid=pid, one=aOne, remove_cached=False,
                                       typ="lf")

    s0 = start_time * sr.fs
    tsel = slice(int(s0), int(s0) + int(duration * sr.fs))

    lfp = sr[tsel, :]*1000
    n_samples, n_channels = lfp.shape
    channels = np.arange(n_channels)
    print(f"Data has {n_samples} samples and {n_channels} channels")

    f, coh = scipy.signal.coherence(x=lfp[tsel, channel_nro],
                                    y=lfp[tsel, channels], fs=sr.fs,
                                    nperseg=segment_len)

    # let's plot now
    hovertext = []
    for yi, yy in enumerate(channels):
        hovertext.append([])
        for xi, xx in enumerate(f):
            hovertext[-1].append(f"freq: {xx}<br />channel: {yy}<br />"
                                 f"coh: {coh[yi][xi]}<br />")

    zmin, zmax = 0.0, 1.0

    title = (f"Coherence: channel {channel_nro}, startTime: {start_time}, "
             f"duration: {duration}, segment_len: {segment_len}")
    fig = go.Figure()
    trace = go.Heatmap(x=f, y=channels, z=coh, colorscale=colorscale,
                       zmin=zmin, zmax=zmax,
                       colorbar=plotUtils.colorbar(zmin=zmin, zmax=zmax,
                                                   n_ticks=n_ticks,
                                                   title="coh"))
    fig.add_trace(trace)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(title=title)

    fig.write_html(fig_filename_pattern.format(start_time, duration,
                                               segment_len, pid, channel_nro,
                                               "html"))

    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
