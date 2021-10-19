#!/bin/bash

# profile the script
kernprof -l -v webcamfps.py

# show the profile
python -m line_profiler  webcamfps.py.lprof

