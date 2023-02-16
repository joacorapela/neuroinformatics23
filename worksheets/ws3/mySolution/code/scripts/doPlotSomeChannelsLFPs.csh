#!/binc/csh

ipython --pdb doPlotSomeChannelsLFPs.py
ipython --pdb doPlotSomeChannelsLFPs.py -- --selected_channels [150,200,250,300] --duration 1.0

