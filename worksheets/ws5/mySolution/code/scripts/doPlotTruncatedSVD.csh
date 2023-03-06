#!/bin/csh

foreach n_components ( `seq 1 20` )
    echo python doTruncatedSVD.py --n_components $n_components
    python doTruncatedSVD.py --n_components $n_components
end
