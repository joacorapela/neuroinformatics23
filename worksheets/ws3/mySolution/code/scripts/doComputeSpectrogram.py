
import sys
import argparse
import numpy as np
import scipy.signal

import one.api
import brainbox.io.one
import brainbox.io.spikeglx


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
                        help="signal duration (sec)", default=1200.0)
    parser.add_argument("--segment_len", type=int,
                        help="Segement length (samples)", default=2**14)
    parser.add_argument("--results_filename_pattern", type=str,
                        help="results filename pattern",
                        default="../../results/spectogram_duration_{:.02f}_segmentLength_{:d}_pid_{:s}_channel_{:d}.npz")
    args = parser.parse_args()

    pid = args.pid
    channel_nro = args.channel_nro
    start_time = args.start_time
    duration = args.duration
    segment_len = args.segment_len
    results_filename_pattern = args.results_filename_pattern

    band = "lf"

    aOne = one.api.ONE(base_url="https://openalyx.internationalbrainlab.org",
                       password="international", silent=True)
    sr = brainbox.io.spikeglx.Streamer(pid=pid, one=aOne, remove_cached=False,
                                       typ=band)

    s0 = start_time * sr.fs
    tsel = slice(int(s0), int(s0) + int(duration * sr.fs))

    # Important: remove sync channel from raw data, and transpose
    channel_lfp = sr[tsel, channel_nro]*1000
    n_samples = len(channel_lfp)
    print(f"Data has {n_samples} samples")

    f, t, Sxx = scipy.signal.spectrogram(x=channel_lfp, fs=sr.fs,
                                         nperseg=segment_len,
                                         scaling="spectrum")
    results_filename = results_filename_pattern.format(duration, segment_len,
                                                       pid, channel_nro)
    np.savez(file=results_filename, f=f, t=t, Sxx=Sxx)


if __name__ == "__main__":
    main(sys.argv)
