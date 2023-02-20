#!/bin/csh

set durations = (10.0 50.0)
set segments_lengths = (256 512 1024 2048 4096 8192 16384)

foreach duration ($durations)
    foreach segment_length ($segments_lengths)
        echo ipython doEstimatePxxWelch.py -- --duration $duration --segment_length $segment_length
        ipython doEstimatePxxWelch.py -- --duration $duration --segment_length $segment_length
    end
end
