#!/bin/csh

ipython --pdb doEx1WithErrorBars.py -- --distribution Rademacher --n_samples 10000
ipython --pdb doEx1WithErrorBars.py -- --distribution Rademacher --n_samples 3
