#!/bin/csh

echo "Started on node " $SLURMD_NODENAME

python -u doPlotCoherence.py

echo "Done on node " $SLURMD_NODENAME

