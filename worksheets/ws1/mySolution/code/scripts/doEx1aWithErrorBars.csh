#!/bin/csh

ipython --pdb doEx1WithErrorBars.py -- --distribution Normal --normal_mean 0 --n_samples 10000
ipython --pdb doEx1WithErrorBars.py -- --distribution Normal --normal_mean 0 --n_samples 3
