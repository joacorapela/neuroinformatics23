
import sys
import argparse
import numpy as np
import plotly.graph_objects as go
# import warnings
# warnings.filterwarnings("error")

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
                        default="../../figures/mi_cv_nRepeats_{:d}_maxNTrials_{:d}_p_{:02f}_q_{:02f}.{:s}")
    args = parser.parse_args()

    n_repeats = args.n_repeats
    max_n_trials = args.max_n_trials
    p = args.p
    q = args.q
    fig_filename_pattern = args.fig_filename_pattern

    n_trials_col = np.arange(1, max_n_trials)

    mis_for_trials_lists = [[] for i in range(max_n_trials)]
    np.random.seed(1)
    for i, n_trials in enumerate(n_trials_col):
        for j in range(n_repeats):
            C1 = utils.sim_data(T=n_trials, p=p, q=q)
            C2 = utils.sim_data(T=n_trials, p=p, q=q)
            mis_for_trials_lists[i].append(utils.mi_cv(C1, C2))
    mis_for_trials = [np.median(np.array(mis_for_trials_lists[i]))
                      for i in range(len(mis_for_trials_lists))]

    fig = go.Figure()
    trace = go.Scatter(x=n_trials_col, y=mis_for_trials, mode="markers")
    fig.add_trace(trace)
    fig.update_layout(xaxis_title="Number of Trials",
                      yaxis_title="Cross-Validated Mutual Information")
    fig.write_image(fig_filename_pattern.format(n_repeats, max_n_trials, p, q,
                                                "png"))
    fig.write_html(fig_filename_pattern.format(n_repeats, max_n_trials, p, q,
                                               "html"))

if __name__ == "__main__":
    main(sys.argv)
