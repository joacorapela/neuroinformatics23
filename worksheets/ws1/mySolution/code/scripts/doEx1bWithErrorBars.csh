#!/bin/csh

ipython --pdb doEx1WithErrorBars.py -- --distribution Normal --normal_mean 0.1
ipython --pdb doEx1WithErrorBars.py -- --distribution Normal --normal_mean 0.01
ipython --pdb doEx1WithErrorBars.py -- --distribution Normal --normal_mean 0.1 --n_samples 500
ipython --pdb doEx1WithErrorBars.py -- --distribution Normal --normal_mean 0.01 --n_samples 30000
