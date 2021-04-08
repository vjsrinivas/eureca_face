#!/bin/sh

# median for retinaface:
MEDIAN_PARAM=( 2.000000 3.000000 4.000000 5.000000 )
SALT_PEPPER_PARAM=( 0.000000 0.100000 0.150000 0.200000 0.250000 0.300000 0.350000 0.400000 0.450000 0.500000 )
for correct in "${MEDIAN_PARAM[@]}"
do
    for noise in "${SALT_PEPPER_PARAM[@]}"
    do
        echo "Running Salt and Pepper @ $noise with Median correction @ $correct"
        cd ./retinaface
        export MXNET_CUDNN_AUTOTUNE_DEFAULT=0;python original_wf_test.py --val_file ../NOISES/salt_pepper/$noise/val.txt --correction median --correction_param $correct --noise salt_pepper  --noise_param $noise
    done
done