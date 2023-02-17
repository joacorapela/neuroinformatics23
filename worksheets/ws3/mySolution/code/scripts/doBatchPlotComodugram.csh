#!/bin/csh

setenv segment_len 8192
setenv start_time 0.0
setenv duration 5500.0

sbatch \
--job-name=comodugram \
--output=../../outputs/comodugram_%A_%a.out \
--error=../../outputs/comodugram_%A_%a.err \
--time=8:00:00 \
--mem=40G \
./doBatchPlotComodugram.sbatch 
