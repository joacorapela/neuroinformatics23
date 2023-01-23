#!/bin/csh
# ipython --pdb doPlotTwoGaussians.py -- --mean1 2.5 --n_samples 100
# ipython --pdb doPlotTwoGaussians.py -- --mean1 3.5 --n_samples 100
# ipython --pdb doPlotTwoGaussians.py -- --mean1 2.5 --n_samples 250
# ipython --pdb doPlotTwoGaussians.py -- --mean1 0.1 --std 1.0 --n_samples 10000 --xmin -0.07 --xmax 0.20 --xdt 0.001
ipython --pdb doPlotTwoGaussians.py -- --mean1 0.01 --std 1.0 --n_samples 10000 --xmin -0.07 --xmax 0.20 --xdt 0.001
