#!/bin/bash

set -x

# set conda env
CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}') # brew install coreutils
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate mask_check

# profile the script
python webcamfps.py --display 1  -n 100

# run kernprof if you want to use decorator @profile
# kernprof -l -v webcamfps.py

# show the profile
python -m line_profiler  webcamfps.py.lprof

