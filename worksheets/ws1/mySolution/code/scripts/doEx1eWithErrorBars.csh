#!/bin/csh

ipython --pdb doEx1WithErrorBars.py -- --distribution VerySkewed --popmean 1e-3 --n_samples 100
ipython --pdb doEx1WithErrorBars.py -- --distribution VerySkewed --popmean 0.0 --n_samples 100
