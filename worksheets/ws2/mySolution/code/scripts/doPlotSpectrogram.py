
import sys
import argparse
import numpy as np
import plotly.graph_objects as go

import plotUtils


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--pid", type=str, help="probed ID",
                        default="38124fca-a0ac-4b58-8e8b-84a2357850e6")
    parser.add_argument("--channel_nro", type=int,
                        help="channel number for spectogram calculation",
                        default=250)
    parser.add_argument("--start_time", type=float,
                        help="plotting start time (sec)", default=0.0)
    parser.add_argument("--duration", type=float,
                        help="signal duration (sec)", default=5800.0)
    parser.add_argument("--segment_len", type=int,
                        help="Segement length (samples)", default=16384)
    parser.add_argument("--n_ticks", type=int,
                        help="nunmber of ticks in colorbar", default=6)
    parser.add_argument("--colorscale", type=str, help="colorscale",
                        default="viridis")
    parser.add_argument("--x_label", type=str, help="x_label",
                        default="Time (sec)")
    parser.add_argument("--y_label", type=str, help="y_label",
                        default="Frequency (Hz)")
    parser.add_argument("--ylim", type=str, help="limits of x-axis (Hz)",
                        default="[0,160]")
    parser.add_argument("--spectrogram_filename_pattern", type=str,
                        help="results filename pattern",
                        default="../../results/spectogram_duration_{:.02f}_segmentLength_{:d}_pid_{:s}_channel_{:d}.npz")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/spectogram_segmentLength_{:d}_pid_{:s}_channel_{:d}.{:s}")
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
    ylim = [float(str) for str in args.ylim[1:-1].split(',')]
    spectrogram_filename_pattern = args.spectrogram_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    spectrogram_filename = spectrogram_filename_pattern.format(
        duration, segment_len, pid, channel_nro)
    load_res = np.load(spectrogram_filename)
    f = load_res["f"]
    t = load_res["t"]
    Sxx = load_res["Sxx"]

    zmin = np.min(Sxx)
    zmax = np.max(Sxx)

    # let's plot now
    hovertext = []
    for yi, yy in enumerate(f):
        hovertext.append([])
        for xi, xx in enumerate(t):
            hovertext[-1].append(f"time: {xx}<br />frequency: {yy}<br />Sxx: {Sxx[yi][xi]}<br />")

    title = f"Sxx (mv^2): channel {channel_nro}, duration: {duration}, segment_len: {segment_len}"
    fig = go.Figure()
    trace = go.Heatmap(x=t, y=f, z=np.log10(Sxx), colorscale=colorscale, zmin=np.log10(zmin), zmax=np.log10(zmax), colorbar=plotUtils.colorbar( zmin=zmin, zmax=zmax, n_ticks=n_ticks, title="Power (mv^2)"), hoverinfo="text", text=hovertext,)
    fig.add_trace(trace)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label, range=ylim)
    fig.update_layout(title=title)

    fig.write_image(fig_filename_pattern.format(segment_len, pid, channel_nro,
                                                "png"))
    fig.write_html(fig_filename_pattern.format(segment_len, pid, channel_nro,
                                               "html"))

    fig.show()

    breakpoint()


if __name__ == "__main__":
   main(sys.argv)
