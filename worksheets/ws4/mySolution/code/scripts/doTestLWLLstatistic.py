
import sys
import argparse
import numpy as np
import scipy.stats

import utils


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_samples", type=int, help="number of samples",
                        default=1000)
    parser.add_argument("--kappa", type=float, help="kappa", default=1.0)
    parser.add_argument("--loc_intercept", type=float,
                        help="location intercept coefficient", default=0.0)
    parser.add_argument("--loc_slope", type=float,
                        help="location slope coefficient", default=2*np.pi)
    parser.add_argument("--weight_func_scale", type=float,
                        help="scale for the weight funct", default=0.01)
    parser.add_argument("--cos_close_threshold", type=float,
                        help="cosine threshold for two angle to be close",
                        default=0.005)
    parser.add_argument("--data_filename_pattern", type=str,
                        help="results filename pattern",
                        default="../../results/data_nSamples_{:d}_kappa_{:.02f}_locIntercept_{:.02f}_locSlope_{:02f}.csv")
    args = parser.parse_args()

    cos_close_threshold = args.cos_close_threshold
    n_samples = args.n_samples
    kappa = args.kappa
    loc_intercept = args.loc_intercept
    loc_slope = args.loc_slope
    weight_func_scale = args.weight_func_scale
    data_filename_pattern = args.data_filename_pattern

    data_filename = data_filename_pattern.format(
        n_samples, kappa, loc_intercept, loc_slope)
    data = np.genfromtxt(data_filename)

    def weight_func(x):
        weight = scipy.stats.norm.pdf(x, scale=weight_func_scale)
        return weight

    def logpdf(value, params):
        loc = params[0]
        kappa = params[1]
        answer = scipy.stats.vonmises.logpdf(value, kappa, loc=loc)
        return answer

    def find_indices_data_close_to_test_point(test_point, data,
                                              threshold=cos_close_threshold):
        def distance(x, y):
            the_mean = (np.exp(1j*x)+np.exp(1j*y))/2
            R = np.abs(the_mean)
            distance = 1 - R
            return distance
        indices = [i for i in range(len(data))
                   if distance(x=test_point, y=data[i]) < threshold]
        return indices

    fit_func = scipy.stats.vonmises.fit
    test_xs = np.linspace(0, 1, 10)
    stat_value = utils.lwll_statistic(
        data=data, test_xs=test_xs,
        find_indices_data_close_to_test_point=
         find_indices_data_close_to_test_point,
        fit_func=fit_func, weight_func=weight_func, logpdf=logpdf)
    shuffled_data = np.column_stack([np.random.permutation(data[:, 0]),
                                     data[:, 1]])
    shuffled_stat_value = utils.lwll_statistic(
        data=shuffled_data, test_xs=test_xs,
        find_indices_data_close_to_test_point=
         find_indices_data_close_to_test_point,
        fit_func=fit_func, weight_func=weight_func, logpdf=logpdf)
    print(f"stat_value: {stat_value}, "
          f"shuffled_stat_value: {shuffled_stat_value}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
