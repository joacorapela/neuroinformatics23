#!/bin/csh

ipython --pdb doEx1WithErrorBars.py -- --distribution Normal --mean 0.1
ipython --pdb doEx1WithErrorBars.py -- --distribution Normal --mean 0.01
