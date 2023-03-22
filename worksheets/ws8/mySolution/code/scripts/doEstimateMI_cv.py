
import sys
import argparse
import numpy as np

import utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_repeats", type=int, help="number of repeats",
                        default=1000)
    parser.add_argument("--max_n_trials", type=int,
                        help="maximum number of trials",
                        default=100)
    parser.add_argument("--p", type=float, help="stimulus probability",
                        default=.5)
    parser.add_argument("--q", type=float, help="neural firing probability",
                        default=.5)
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/mi_nRepeats_{:d}_maxNTrials_{:d}_p_{:02f}_q_{:02f}.{:s}")
    args = parser.parse_args()

    n_repeats = args.n_repeats
    max_n_trials = args.max_n_trials
    p = args.p
    q = args.q
    fig_filename_pattern = args.fig_filename_pattern

    np.random.seed(1)
    C1 = utils.sim_data(T=10, p=.5, q=.5)
    C2 = utils.sim_data(T=10, p=.5, q=.5)
    print(C1)
    print(C2)
    print(utils.mi_cv(C1, C2))
    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
