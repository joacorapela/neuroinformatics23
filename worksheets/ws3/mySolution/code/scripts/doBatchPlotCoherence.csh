#!/bin/csh

sbatch \
--job-name=coh \
--output=../../outputs/coh_%A_%a.out \
--error=../../outputs/coh_%A_%a.err \
--time=8:00:00 \
--mem=40G \
./doBatchPlotCoherence.sbatch 
