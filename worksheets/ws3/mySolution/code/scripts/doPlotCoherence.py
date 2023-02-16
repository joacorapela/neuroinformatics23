
import sys
import time
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
                        # help="signal duration (sec)", default=3.0)
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
                        default=("../../figures/coh_startTime_{:.02f}_"
                                 "duration_{:.02f}_segmentLen_{:d}_"
                                 "pid_{:s}_channel_{:d}.{:s}"))
    args = parser.parse_args()

    pid = args.pid
    channel_nro = args.channel_nro
    start_time = args.start_time
    duration = args.duration
    segment_len = args.segment_len
    n_ticks = args.n_ticks
    colorscale = args.colorscale
    x_label = args.x_label
    y_label = args.y_label
    fig_filename_pattern = args.fig_filename_pattern

    print("Connecting to IBL ...")
    t0 = time.time()
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
    elapsed_time = time.time() - t0
    print(f"elapsed_time: {elapsed_time}")

    print("Getting LFP ...")
    t0 = time.time()

    lfp = sr[tsel, :-sr.nsync]*1000
    n_samples, n_channels = lfp.shape
    channels = np.arange(n_channels)
    print(f"Data has {n_samples} samples and {n_channels} channels")
    elapsed_time = time.time() - t0
    print(f"elapsed_time: {elapsed_time}")

    print("Computing coherence ...")
    t0 = time.time()
    nperseg = segment_len if segment_len>0 else None
    f, coh = scipy.signal.coherence(
        x=lfp[tsel, channel_nro], y=lfp[tsel, channels].T, fs=sr.fs,
        nperseg=nperseg)
    elapsed_time = time.time() - t0
    print(f"elapsed_time: {elapsed_time}")

    # let's plot now
    print("Plotting ...")
    t0 = time.time()
    hovertext = []
    for yi, yy in enumerate(channels):
        hovertext.append([])
        for xi, xx in enumerate(f):
            hovertext[-1].append(f"freq: {xx}<br />"
                                 f"channel: {yy}<br />"
                                 f"coh: {coh[yi][xi]}<br />"
                                 f"loc: {channel_locs_acronyms[yi]}")

    zmin, zmax = 0.0, 1.0

    title = (f"Coherence: channel {channel_nro}, startTime: {start_time}, "
             f"duration: {duration}, segment_len: {segment_len}")
    fig = go.Figure()
    trace = go.Heatmap(x=f, y=channels, z=coh, colorscale=colorscale,
                       zmin=zmin, zmax=zmax,
                       hoverinfo="text", text=hovertext,
                      )
    fig.add_trace(trace)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(title=title)
    elapsed_time = time.time() - t0
    print(f"elapsed_time: {elapsed_time}")

    fig.write_html(fig_filename_pattern.format(start_time, duration,
                                               segment_len, pid, channel_nro,
                                               "html"))

    # fig.show()

    # breakpoint()


if __name__ == "__main__":
    main(sys.argv)
