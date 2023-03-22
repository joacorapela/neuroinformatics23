
import sys
import numpy as np

import utils


def main(argv):
    np.random.seed(0)
    C = sim_data(T=10, p=.5, q=.5)
    mi = utils.mi(C)
    print(mi)
    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
