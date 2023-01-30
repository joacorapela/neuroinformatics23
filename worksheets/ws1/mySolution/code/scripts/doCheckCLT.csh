#!/bin/csh

ipython --pdb doCheckCLT.py -- --distribution Rademacher
ipython --pdb doCheckCLT.py -- --distribution StdCauchy

