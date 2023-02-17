#!/bin/csh

ipython --pdb doPlotSomeChannelsLFPs.py -- --selected_channels "[0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368]" --duration 5.0
ipython --pdb doPlotSomeChannelsLFPs.py -- --selected_channels "[170,200,250,300]" --duration 1.0

