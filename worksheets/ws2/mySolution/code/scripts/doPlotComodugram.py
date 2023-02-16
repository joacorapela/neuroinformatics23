
import sys
import time
import argparse
import numpy as np
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
                        help="LFP extraction start time (sec)", default=4500.0)
    parser.add_argument("--duration", type=float,
                        help="LFP extraction duration (sec)", default=1000.0)
    parser.add_argument("--channel_nro", type=int,
                        help="channel number for spectrogram calculation",
                        default=250)
    parser.add_argument("--segment_len", type=int,
                        help="Segement length (samples)", default=2**13)
    parser.add_argument("--n_ticks", type=int,
                        help="nunmber of ticks in colorbar", default=6)
    parser.add_argument("--colorscale", type=str, help="colorscale",
                        default="RdBu")
    parser.add_argument("--x_label", type=str, help="x_label",
                        default="Frequency (Hz)")
    parser.add_argument("--y_label", type=str, help="y_label",
                        default="Frequency (Hz)")
    parser.add_argument("--xlim", type=str, help="limits of x-axis (sec)",
                        default="[4500.0,5500.0]")
    parser.add_argument("--ylim", type=str, help="limits of x-axis (Hz)",
                        default="[0,160]")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/comodugram_segmentLength_{:d}_pid_{:s}_channel_{:d}_startTime_{:.02f}_duration_{:.02f}.{:s}")
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
    xlim = [float(str) for str in args.xlim[1:-1].split(',')]
    ylim = [float(str) for str in args.ylim[1:-1].split(',')]
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
    # Important: remove sync channel from raw data, and transpose
    channel_lfp = sr[tsel, channel_nro]*1000
    n_samples = len(channel_lfp)
    print(f"Data has {n_samples} samples")
    elapsed_time = time.time() - t0
    print(f"elapsed_time: {elapsed_time}")

    print("Computing spectrogram ...")
    t0 = time.time()
    f, t, Sxx = scipy.signal.spectrogram(x=channel_lfp, fs=sr.fs,
                                         nperseg=segment_len,
                                         scaling="spectrum")
    elapsed_time = time.time() - t0
    print(f"elapsed_time: {elapsed_time}")

    print("Computing comodugram ...")
    t0 = time.time()
    Sxx = Sxx[f<120,:]
    comodugram = np.corrcoef(Sxx)
    elapsed_time = time.time() - t0
    print(f"elapsed_time: {elapsed_time}")

    # let's plot now
    print("Plotting ...")
    t0 = time.time()
#     hovertext = []
#     for yi, yy in enumerate(f):
#         hovertext.append([])
#         for xi, xx in enumerate(t):
#             hovertext[-1].append(f"time: {xx}<br />frequency: {yy}<br />Sxx: {Sxx[yi][xi]}<br />")

    title = f"Comodugram: channel {channel_nro}, region {channel_locs_acronyms[channel_nro]}, segment_len {segment_len}, start_time {start_time}, duration {duration}"
    fig = go.Figure()
    trace = go.Heatmap(x=f, y=f, z=comodugram, colorscale=colorscale,
                       zmin=-1.0, zmax=1.0)
    fig.add_trace(trace)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(title=title)
    elapsed_time = time.time() - t0
    print(f"elapsed_time: {elapsed_time}")

    # fig.write_image(fig_filename_pattern.format(segment_len, pid, channel_nro, "png"))
    fig.write_html(fig_filename_pattern.format(segment_len, pid, channel_nro,
                                               start_time, duration, "html"))

    fig.show()

    breakpoint()


if __name__ == "__main__":
   main(sys.argv)
