#!/bin/bash

set -x

CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}') # brew install coreutils
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate mask_check
rm webcam_demo.py.lprof

# run kernprof if you want to use decorator @profile
# gtimeout 60s kernprof -l -v webcam_demo.py

python webcam_demo.py
python -m line_profiler webcam_demo.py.lprof | less
