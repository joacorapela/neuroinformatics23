
import sys
import argparse
import numpy as np
import scipy.signal
import plotly.graph_objects as go
import nitime.algorithms.spectral as spectrum

import one.api
import brainbox.io.spikeglx


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--pid", type=str, help="probed ID",
                        default="38124fca-a0ac-4b58-8e8b-84a2357850e6")
    parser.add_argument("--start_time", type=float,
                        help="plotting start time (sec)", default=0.0)
    parser.add_argument("--duration", type=float,
                        help="plotting duration (sec)", default=600.0)
    parser.add_argument("--channel_to_plot", type=int,
                        help="channel to plot", default=0)
    parser.add_argument("--x_label", type=str, help="x_label",
                        default="Frequency (Hz)")
    parser.add_argument("--y_label", type=str, help="x_label",
                        default="Power (V^2)")
    parser.add_argument("--title_pattern", type=str, help="title",
                        default="probe ID: {:s}, Channel {:d}")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/pxx_multitaper_duration_{:f}_timeBandwidthProduct_{:f}_pid_{:s}_channel_{:d}.{:s}")
    args = parser.parse_args()

    pid = args.pid
    start_time = args.start_time
    duration = args.duration
    channel_to_plot = args.channel_to_plot
    x_label = args.x_label
    y_label = args.y_label
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    title = title_pattern.format(pid, channel_to_plot)
    band = "lf"

    aOne = one.api.ONE(base_url="https://openalyx.internationalbrainlab.org",
                       password="international", silent=True)
    sr = brainbox.io.spikeglx.Streamer(pid=pid, one=aOne, remove_cached=False,
                                       typ=band)
    s0 = start_time * sr.fs
    tsel = slice(int(s0), int(s0) + int(duration * sr.fs))

    # Important: remove sync channel from raw data, and transpose
    lfp = sr[tsel, :-sr.nsync].T
    n_channels, n_samples = lfp.shape
    print(f"Data has {n_channels} channels and {n_samples} samples")

    breakpoint()

    # f, Pxx = scipy.signal.welch(x=lfp[channel_to_plot, :]*1000, fs=sr.fs, nperseg=segment_length, scaling="spectrum")
    x = lfp[channel_to_plot, :]*1000
    f, Pxx, _ = spectrum.multi_taper_psd(x-x.mean(), Fs=sr.fs, NW=time_bandwidth_product)

    fig = go.Figure()
    trace = go.Scatter(x=f, y=Pxx)
    fig.add_trace(trace)

    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(title=title)
    fig.write_image(fig_filename_pattern.format(duration, time_bandwidth_product, pid, channel_to_plot, "png"))
    fig.write_html(fig_filename_pattern.format(duration, time_bandwidth_product, pid, channel_to_plot, "html"))

    # fig.show()

    # breakpoint()


if __name__ == "__main__":
   main(sys.argv)
