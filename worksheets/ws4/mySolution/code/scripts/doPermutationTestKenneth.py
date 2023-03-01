
import sys
import argparse
import pickle
import numpy as np
import scipy.stats
import multiprocessing

import utils


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_resamples", type=int,
                        help="number of resamples for permutation test",
                        default=1000)
    parser.add_argument("--n_samples", type=int, help="number of samples",
                        default=1000)
    parser.add_argument("--kappa", type=float, help="kappa", default=1.0)
    parser.add_argument("--loc_intercept", type=float,
                        help="location intercept coefficient", default=0.0)
    parser.add_argument("--loc_slope", type=float,
                        help="location slope coefficient", default=2*np.pi)
    parser.add_argument("--data_filename_pattern", type=str,
                        help="results filename pattern",
                        default=("../../results/data_nSamples_{:d}_"
                                 "kappa_{:.02f}_locIntercept_{:.02f}_locSlope_"
                                 "{:02f}.csv"))
    parser.add_argument("--results_filename_pattern", type=str,
                        help="results filename pattern",
                        default=("../../results/kennethPermutationTest_"
                                 "nSamples_{:d}_kappa_{:.02f}_"
                                 "locIntercept_{:.02f}_"
                                 "locSlope_{:02f}_"
                                 "nResamples_{:d}.pickle"))
    args = parser.parse_args()

    n_resamples = args.n_resamples
    n_samples = args.n_samples
    kappa = args.kappa
    loc_intercept = args.loc_intercept
    loc_slope = args.loc_slope
    data_filename_pattern = args.data_filename_pattern
    results_filename_pattern = args.results_filename_pattern

    data_filename = data_filename_pattern.format(
        n_samples, kappa, loc_intercept, loc_slope)
    data = np.genfromtxt(data_filename)

    obs_stat_value = utils.kenneth_statistic(x=data[:, 0], theta=data[:, 1])

    shuffled_stat_values = np.empty(n_resamples, dtype=np.double)
    for i in range(n_resamples):
        shuffled_stat_values[i] = utils.kenneth_statistic(
            x=data[:, 0], theta=np.random.permutation(data[:, 1]))
    shuffled_stat_values = np.sort(shuffled_stat_values)
    index_sorted = np.searchsorted(shuffled_stat_values, obs_stat_value)
    p_value = (n_resamples - index_sorted) / n_resamples
    print(f"p_value: {p_value}")

    results = {"obs_stat_value": obs_stat_value,
               "shuffled_stat_values": shuffled_stat_values,
               "p_value": p_value}
    results_filename = results_filename_pattern.format(n_resamples, kappa,
                                                       loc_intercept,
                                                       loc_slope, n_resamples)
    with open(results_filename, "wb") as f:
        pickle.dump(results, f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
