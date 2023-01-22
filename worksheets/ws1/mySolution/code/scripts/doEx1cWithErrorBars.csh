#!/bin/csh

ipython --pdb doEx1WithErrorBars.py -- --distribution StdCauchy --n_samples 10000
ipython --pdb doEx1WithErrorBars.py -- --distribution StdCauchy --n_samples 3
