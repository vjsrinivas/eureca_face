#!/bin/bash

mkdir NOISES
mkdir CORRECTIONS

# Preprocessing datasets:
python preprocessing.py --type wider --data_root ./WIDERFACE --noise_list noise.txt --correction_list corrections.txt

# Dataset analysis:
python dataset_analysis.py
