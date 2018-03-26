#!/bin/bash

python generate.py -c debug -e debug -t sn
python evaluate.py -p save -g ../datasets/eecs442challenge/train/normal/
