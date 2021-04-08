#!/bin/bash

# Preprocessing datasets:
#python preprocessing.py --type wider --data_root ./WIDERFACE --noise_list noise.txt --correction_list corrections.txt
python preprocessing.py --type wider --data_root ./WIDERFACE --noise_list noise_2.txt --correction_list corrections.txt

# Dataset analysis:
#python dataset_analysis.py
