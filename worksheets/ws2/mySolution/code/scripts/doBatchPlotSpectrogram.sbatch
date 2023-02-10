#!/bin/csh

echo "Started with duration: " $duration " segment_len: " $segment_len " on node " $SLURMD_NODENAME

python -u doPlotSpectrogram.py --duration=$duration --segment_len=$segment_len

echo "Done with duration: " $duration " segment_len: " $segment_len " on node " $SLURMD_NODENAME

