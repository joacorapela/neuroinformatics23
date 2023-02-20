
import sys
import argparse
import numpy as np
import scipy.stats


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_samples", type=int, help="number of samples",
                        default=1000)
    parser.add_argument("--kappa", type=float, help="kappa", default=1.0)
    parser.add_argument("--loc_intercept", type=float,
                        help="location intercept coefficient", default=0.0)
    parser.add_argument("--loc_slope", type=float,
                        help="location slope coefficient", default=2*np.pi)
    parser.add_argument("--results_filename_pattern", type=str,
                        help="results filename pattern",
                        default="../../results/data_nSamples_{:d}_kappa_{:.02f}_locIntercept_{:.02f}_locSlope_{:02f}.csv")
    args = parser.parse_args()

    n_samples = args.n_samples
    kappa = args.kappa
    loc_intercept = args.loc_intercept
    loc_slope = args.loc_slope
    results_filename_pattern = args.results_filename_pattern

    x = np.random.uniform(size=n_samples)
    theta = np.array([
        scipy.stats.vonmises.rvs(kappa=kappa, loc=loc_intercept+loc_slope*xi)
        for xi in x])
    results = np.column_stack([x, theta])

    results_filename = results_filename_pattern.format(
        n_samples, kappa, loc_intercept, loc_slope)
    np.savetxt(results_filename, results)



if __name__ == "__main__":
    main(sys.argv)
