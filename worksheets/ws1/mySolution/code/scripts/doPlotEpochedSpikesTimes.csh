#!/bin/csh

python doEpochAndPlotNeuronSpikesTimes.py --epoch_event stimOn_times --sorting_event_name response_times --colors_event_name choice
python doEpochAndPlotNeuronSpikesTimes.py --epoch_event stimOn_times --sorting_event_name response_times --colors_event_name feedbackType
