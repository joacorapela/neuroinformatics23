
import sys
import argparse
import pickle
import numpy as np
import scipy.stats
import multiprocessing

import utils


def weight_func(x, scale):
    weight = scipy.stats.norm.pdf(x, scale=scale)
    return weight


def logpdf(value, params):
    loc = params[0]
    kappa = params[1]
    answer = scipy.stats.vonmises.logpdf(value, kappa, loc=loc)
    return answer


def task(i, shuffled_data, test_xs, fit_func, weight_func, weight_func_scale,
         logpdf, x_dist_thr):
    print(f"Processing resample {i}")
    shuffled_stat_value = utils.lwll_statistic(
        data=shuffled_data, test_xs=test_xs, fit_func=fit_func,
        weight_func=weight_func, weight_func_scale=weight_func_scale,
        logpdf=logpdf, x_dist_thr=x_dist_thr)
    print(f"Done with resample {i} "
          f"with shuffled_stat_value: {shuffled_stat_value}")
    return shuffled_stat_value


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
    parser.add_argument("--weight_func_scale", type=float,
                        help="scale for the weight funct", default=0.01)
    parser.add_argument("--x_dist_thr", type=float,
                        help=("distance threshold for the definition of close "
                              "x points"), default=0.01)
    parser.add_argument("--n_xs", type=int, help="number of x values",
                        default=10)
    parser.add_argument("--data_filename_pattern", type=str,
                        help="results filename pattern",
                        default=("../../results/data_nSamples_{:d}_"
                                 "kappa_{:.02f}_locIntercept_{:.02f}_locSlope_"
                                 "{:02f}.csv"))
    parser.add_argument("--results_filename_pattern", type=str,
                        help="results filename pattern",
                        default=("../../results/lwllPermutationTest_"
                                 "nSamples_{:d}_kappa_{:.02f}_"
                                 "locIntercept_{:.02f}_"
                                 "locSlope_{:02f}_"
                                 "nXs_{:d}_"
                                 "nResamples_{:d}.pickle"))
    args = parser.parse_args()

    n_resamples = args.n_resamples
    n_samples = args.n_samples
    kappa = args.kappa
    loc_intercept = args.loc_intercept
    loc_slope = args.loc_slope
    weight_func_scale = args.weight_func_scale
    x_dist_thr = args.x_dist_thr
    n_xs = args.n_xs
    data_filename_pattern = args.data_filename_pattern
    results_filename_pattern = args.results_filename_pattern

    data_filename = data_filename_pattern.format(
        n_samples, kappa, loc_intercept, loc_slope)
    data = np.genfromtxt(data_filename)

    fit_func = scipy.stats.vonmises.fit
    test_xs = np.linspace(0, 1, n_xs)
    print("Started computing obs_stat_value")
    obs_stat_value = utils.lwll_statistic(
        data=data, test_xs=test_xs,
        fit_func=fit_func, weight_func=weight_func,
        weight_func_scale=weight_func_scale, logpdf=logpdf,
        x_dist_thr=x_dist_thr)
    print(f"Done with computing obs_stat_value: {obs_stat_value}")

    shuffled_data_col = [None for i in range(n_resamples)]
    for i in range(n_resamples):
        shuffled_data_col[i] = np.column_stack(
            [np.random.permutation(data[:, 0]), data[:, 1]]
        )
    # def task(i, data, test_xs, fit_func, weight_func, weight_func_scale,
    #          logpdf, x_dist_thr):
    items = [(i, shuffled_data_col[i], test_xs, fit_func, weight_func,
              weight_func_scale, logpdf, x_dist_thr)
             for i in range(n_resamples)]
    shuffled_stat_values = np.empty(n_resamples, dtype=np.double)
    with multiprocessing.Pool() as pool:
        for i, shuffled_stat_value in enumerate(pool.starmap(task, items)):
            shuffled_stat_values[i] = shuffled_stat_value
    shuffled_stat_values = np.sort(shuffled_stat_values)
    index_sorted = np.searchsorted(shuffled_stat_values, obs_stat_value)
    p_value = (n_resamples - index_sorted) / n_resamples
    print(f"p_value: {p_value}")

    results = {"obs_stat_value": obs_stat_value,
               "shuffled_stat_values": shuffled_stat_values,
               "p_value": p_value}
    results_filename = results_filename_pattern.format(n_resamples, kappa,
                                                       loc_intercept,
                                                       loc_slope, n_xs,
                                                       n_resamples)
    with open(results_filename, "wb") as f:
        pickle.dump(results, f)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
