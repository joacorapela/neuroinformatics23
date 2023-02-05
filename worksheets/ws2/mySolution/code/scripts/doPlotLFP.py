
import sys
import argparse
import numpy as np
import scipy.signal

import one.api
import brainbox.io.spikeglx


def main(argv):
    # pid = 'da8dfec1-d265-44e8-84ce-6ae9c109b8bd'
    pid = 'e31b4e39-e350-47a9-aca4-72496d99ff2a'
    time0 = 100 # timepoint in recording to stream
    time_win = 10 # number of seconds to stream
    band = 'lf' # either 'ap' or 'lf'

    aOne = one.api.ONE(base_url='https://openalyx.internationalbrainlab.org',
                       password='international', silent=True)
    sr = brainbox.io.spikeglx.Streamer(pid=pid, one=aOne, remove_cached=False,
                                       typ=band)
    s0 = time0 * sr.fs
    tsel = slice(int(s0), int(s0) + int(time_win * sr.fs))

    # Important: remove sync channel from raw data, and transpose
    lfp = sr[tsel, :-sr.nsync].T
    nChans, nSamps = lfp.shape
    print(f"Data has {nChans} channels and {nSamps} samples")

    breakpoint()


if __name__ == "__main__":
   main(sys.argv)
