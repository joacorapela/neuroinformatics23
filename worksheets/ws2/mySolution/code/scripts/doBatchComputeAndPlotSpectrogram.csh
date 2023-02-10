#!/bin/csh

setenv segment_len 8192

sbatch \
--job-name=calcSxx \
--output=../../outputs/calcSxx_%A_%a.out \
--error=../../outputs/calcSxx_%A_%a.err \
--time=8:00:00 \
--mem=40G \
./doBatchComputeAndPlotSpectrogram.sbatch 
