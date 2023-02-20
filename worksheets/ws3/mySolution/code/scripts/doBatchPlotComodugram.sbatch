#!/bin/csh

echo "Started with segment_len " $segment_len ", start_time " $start_time ", duration: " $duration " on node " $SLURMD_NODENAME

python -u doPlotComodugram.py --start_time $start_time --duration $duration --segment_len=$segment_len

echo "Done with segment_len " $segment_len ", start_time " $start_time ", duration: " $duration " on node " $SLURMD_NODENAME

