#!/bin/bash

python generate.py -c third_model_5 -e third_model_5 -t sn
python evaluate.py -p save -g ../datasets/eecs442challenge/train/normal/
