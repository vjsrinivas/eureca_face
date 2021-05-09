#!/bin/sh

# median for retinaface:
MEDIAN_PARAM=( 3.000000 5.000000 7.000000 9.000000 )
HE_PARAM=(0.000000)
SALT_PEPPER_PARAM=( 0.000000 0.100000 0.150000 0.200000 0.250000 0.300000 0.350000 0.400000 0.450000 0.500000 )
GAMMA_NOISE_PARAM=(0.000000 40.000000 80.000000 120.000000 160.000000 200.000000 240.000000 280.000000)
POISSON_PARAM=(0.000000 0.100000 0.200000 0.300000 0.400000 0.500000 0.600000 0.700000 0.800000 0.900000 1.000000)
GAUSS_NOISE_PARAM=( 0.000000 10.000000 20.000000 30.000000 40.000000 50.000000 60.000000 70.000000 80.000000 90.000000 100.000000 )

# RETINAFACE CORRECTIONS:

# median + salt and pepper param on retinaface:
for correct in "${MEDIAN_PARAM[@]}"
do
    for noise in "${SALT_PEPPER_PARAM[@]}"
    do
        echo "Running Salt and Pepper @ $noise with Median correction @ $correct"
        export MXNET_CUDNN_AUTOTUNE_DEFAULT=0;python original_wf_test.py --val_file ../NOISES/salt_pepper/$noise/val.txt --correction median --correction_param $correct --noise salt_pepper  --noise_param $noise
    done
done

# median + gaussian noise on retinaface
for correct in "${MEDIAN_PARAM[@]}"
do
    for noise in "${GAUSS_NOISE_PARAM[@]}"
    do
        echo "Running Gaussian Noise @ $noise with Median @ $correct"
        export MXNET_CUDNN_AUTOTUNE_DEFAULT=0;python original_wf_test.py --val_file ../NOISES/gaussian_noise/$noise/val.txt --correction median --correction_param $correct --noise gaussian_noise --noise_param $noise
    done
done

# median + poisson on retinaface:
for correct in "${MEDIAN_PARAM[@]}"
do
    for noise in "${POISSON_PARAM[@]}"
    do
        echo "Running Possion @ $noise Median @ $correct"
        export MXNET_CUDNN_AUTOTUNE_DEFAULT=0;python original_wf_test.py --val_file ../NOISES/poisson/$noise/val.txt --correction median --correction_param $correct --noise poisson --noise_param $noise
    done
done

# histogram + gama on retinaface:
for correct in "${HE_PARAM[@]}"
do
    for noise in "${GAMMA_NOISE_PARAM[@]}"
    do
        echo "Running Gamma @ $noise with Histogram Equalization"
        export MXNET_CUDNN_AUTOTUNE_DEFAULT=0;python original_wf_test.py --val_file ../NOISES/gamma/$noise/val.txt --correction he --correction_param $correct --noise gamma --noise_param $noise
    done
done


# DSFD CORRECTIONS:

cd ../dsfd

# median + salt and pepper param on dsfd:
for correct in "${MEDIAN_PARAM[@]}"
do
    for noise in "${SALT_PEPPER_PARAM[@]}"
    do
        echo "Running Salt and Pepper @ $noise with Median correction @ $correct"
        export MXNET_CUDNN_AUTOTUNE_DEFAULT=0;python eval.py --val_file ../NOISES/salt_pepper/$noise/val.txt --correction median --correction_param $correct --noise salt_pepper  --noise_param $noise
    done
done

# median + gaussian noise on dsfd
for correct in "${MEDIAN_PARAM[@]}"
do
    for noise in "${GAUSS_NOISE_PARAM[@]}"
    do
        echo "Running Gaussian Noise @ $noise with Median correction @ $correct"
        export MXNET_CUDNN_AUTOTUNE_DEFAULT=0; python eval.py --val_file ../NOISES/gaussian_noise/$noise/val.txt --correction median --correction_param $correct --noise gaussian_noise --noise_param $noise
    done
done

# median + poisson on dsfd:
for correct in "${MEDIAN_PARAM[@]}"
do
    for noise in "${POISSON_PARAM[@]}"
    do
        echo "Running Possion @ $noise Median @ $correct"
        export MXNET_CUDNN_AUTOTUNE_DEFAULT=0;python eval.py --val_file ../NOISES/poisson/$noise/val.txt --correction median --correction_param $correct --noise poisson --noise_param $noise
    done
done

# histogram equilization + gamma noise on dsfd
for correct in "${HE_PARAM[@]}"
do
    for noise in "${GAMMA_NOISE_PARAM[@]}"
    do
        echo "Running Gamma @ $noise with Histogram Equalization"
        export MXNET_CUDNN_AUTOTUNE_DEFAULT=0; python eval.py --val_file ../NOISES/gamma/$noise/val.txt --correction he --correction_param $correct --noise gamma --noise_param $noise
    done
done


cd ../