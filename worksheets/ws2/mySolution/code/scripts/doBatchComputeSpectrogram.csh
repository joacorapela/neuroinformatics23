#!/bin/csh

setenv duration 5800
setenv segment_len 16384
# setenv duration 100
# setenv segment_len 64

sbatch \
--job-name=computeSpectrogram \
--output=../../outputs/computeSpectrogram_%A_%a.out \
--error=../../outputs/computeSpectrogram_%A_%a.err \
--time=8:00:00 \
--mem=40G \
./doBatchComputeSpectrogram.sbatch 
