#!/bin/csh

echo "Started with segment_len: " $segment_len " on node " $SLURMD_NODENAME

python -u doComputeAndPlotSpectrogram.py --segment_len=$segment_len

echo "Done with segment_len: " $segment_len " on node " $SLURMD_NODENAME

