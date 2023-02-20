
import sys
import argparse
import numpy as np
import scipy.stats

import utils


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--x_test", type=float, help="x value to test", default=0.5)
    parser.add_argument("--theta_test", type=float, help="theta value to test",
                        default=2*np.pi*0.5)
    parser.add_argument("--n_samples", type=int, help="number of samples",
                        default=1000)
    parser.add_argument("--kappa", type=float, help="kappa", default=1.0)
    parser.add_argument("--loc_intercept", type=float,
                        help="location intercept coefficient", default=0.0)
    parser.add_argument("--loc_slope", type=float,
                        help="location slope coefficient", default=2*np.pi)
    parser.add_argument("--weight_func_scale", type=float,
                        help="scale for the weight funct", default=0.01)
    parser.add_argument("--results_filename_pattern", type=str,
                        help="results filename pattern",
                        default="../../results/data_nSamples_{:d}_kappa_{:.02f}_locIntercept_{:.02f}_locSlope_{:02f}.csv")
    args = parser.parse_args()

    x_test = args.x_test
    theta_test = args.theta_test
    n_samples = args.n_samples
    kappa = args.kappa
    loc_intercept = args.loc_intercept
    loc_slope = args.loc_slope
    weight_func_scale = args.weight_func_scale
    results_filename_pattern = args.results_filename_pattern

    results_filename = results_filename_pattern.format(
        n_samples, kappa, loc_intercept, loc_slope)
    data = np.genfromtxt(results_filename)

    kappa_test = kappa

    def weight_func(x):
        weight = scipy.stats.norm.pdf(x, scale=weight_func_scale)
        return weight

    def logpdf(value, params):
        loc = params[0]
        kappa = params[1]
        answer = scipy.stats.vonmises.logpdf(value, kappa, loc=loc)
        return answer

    lwll = utils.locallyWeightedLogLikelihood(weight_func=weight_func,
                                              logpdf=logpdf,
                                              x=x_test,
                                              params_x=(theta_test,
                                                        kappa_test),
                                              data=data)
    print(f"lwll: {lwll}, x_test: {x_test}, theta_test: {theta_test}, "
          f"loc_intercept: {loc_intercept}, loc_slope: {loc_slope}, "
          f"kappa: {kappa}")

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
