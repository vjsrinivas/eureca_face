cd ./scripts

# Degradations:
bash ./run_retinaface_wider_gaussian_noise.sh; \
bash ./run_retinaface_wider_poisson.sh; \
bash ./run_retinaface_wider_salt_pepper.sh; \
bash ./run_retinaface_wider_speckle.sh \
bash ./run_retinaface_wider_gamma.sh \
bash ./run_tinaface_wider_gaussian_noise.sh; \
bash ./run_tinaface_wider_poisson.sh; \
bash ./run_tinaface_wider_salt_pepper.sh; \
bash ./run_tinaface_wider_speckle.sh; \
bash ./run_tinaface_wider_gamma.sh; \
bash ./run_dsfd_wider_gaussian_noise.sh; \
bash ./run_dsfd_wider_poisson.sh; \
bash ./run_dsfd_wider_salt_pepper.sh; \
bash ./run_dsfd_wider_speckle.sh \
bash ./run_dsfd_wider_gamma.sh

# Corrections:
cd ../WIDERFACE/eval_tools/
ROOT="/home/vijay/Documents/devmk4/eureca/eureca_face"
MATLAB=/usr/local/MATLAB/R2020b/bin/matlab

MEDIAN_PARAM=( 3.000000 5.000000 7.000000 9.000000 )
HE_PARAM=(0.000000)
SALT_PEPPER_PARAM=( 0.000000 0.100000 0.150000 0.200000 0.250000 0.300000 0.350000 0.400000 0.450000 0.500000 )
GAUSS_NOISE_PARAM=( 0.000000 10.000000 20.000000 30.000000 40.000000 50.000000 60.000000 70.000000 80.000000 90.000000 100.000000 )
POISSON_PARAM=( 0.000000 0.100000 0.200000 0.300000 0.400000 0.500000 0.600000 0.700000 0.800000 0.900000)
GAMMA_PARAM=(0.000000 40.000000 80.000000 120.000000 160.000000 200.000000 240.000000 280.000000)

# RetinaFace evaluations:

# salt pepper + median:
for correct in "${MEDIAN_PARAM[@]}"
do
    for noise in "${SALT_PEPPER_PARAM[@]}"
    do
        combined=$noise"_"$correct
        echo "Running Salt and Pepper @ $noise with Median correction @ $correct"
        sudo $MATLAB -nodisplay -nosplash -r \
        "wider_eval(\
        \"$ROOT/CORRECTIONS/salt_pepper_median/$combined/retinaface\", \
        \"retinaface_salt_pepper_median_$combined\", \
        \"$ROOT/results/retinaface/salt_pepper_median_"$combined"_map.txt\"); \
        exit()"
    done
done

# gaussian_noise + median
for correct in "${MEDIAN_PARAM[@]}"
do
    for noise in "${GAUSS_NOISE_PARAM[@]}"
    do
        combined=$noise"_"$correct
        echo "Running Gaussian Noise @ $noise with Median correction @ $correct"
        sudo $MATLAB -nodisplay -nosplash -r \
        "wider_eval(\
        \"$ROOT/CORRECTIONS/gaussian_noise_median/$combined/retinaface\", \
        \"retinaface_gaussian_noise_median_$combined\", \
        \"$ROOT/results/retinaface/gaussian_noise_median_"$combined"_map.txt\"); \
        exit()"
    done
done

# poisson + median
for correct in "${MEDIAN_PARAM[@]}"
do
    for noise in "${POISSON_PARAM[@]}"
    do
        combined=$noise"_"$correct
        echo "Running Poisson @ $noise with Median correction @ $correct"
        echo $ROOT/CORRECTIONS/poisson_median/$combined/retinaface
        sudo $MATLAB -nodisplay -nosplash -r \
        "wider_eval(\
        \"$ROOT/CORRECTIONS/poisson_median/$combined/retinaface\", \
        \"retinaface_poisson_median_$combined\", \
        \"$ROOT/results/retinaface/poisson_median_"$combined"_map.txt\"); \
        exit()"
    done
done

# gamma + he 
for correct in "${HE_PARAM[@]}"
do
    for noise in "${GAMMA_PARAM[@]}"
    do
        combined=$noise"_"$correct
        echo "Running Gammma @ $noise with HE correction @ $correct"
        sudo $MATLAB -nodisplay -nosplash -r \
        "wider_eval(\
        \"$ROOT/CORRECTIONS/gamma_he/$combined/retinaface\", \
        \"retinaface_gamma_he_$combined\", \
        \"$ROOT/results/retinaface/gamma_he_"$combined"_map.txt\"); \
        exit()"
    done
done

# DSFD evaluations:

# salt pepper + median:
for correct in "${MEDIAN_PARAM[@]}"
do
    for noise in "${SALT_PEPPER_PARAM[@]}"
    do
        combined=$noise"_"$correct
        echo "Running Salt and Pepper @ $noise with Median correction @ $correct"
        sudo $MATLAB -nodisplay -nosplash -r \
        "wider_eval(\
        \"$ROOT/CORRECTIONS/salt_pepper_median/$combined/dsfd\", \
        \"dsfd_salt_pepper_median_$combined\", \
        \"$ROOT/results/dsfd/salt_pepper_median_"$combined"_map.txt\"); \
        exit()"
    done
done

# gaussian noise + median
for correct in "${MEDIAN_PARAM[@]}"
do
    for noise in "${GAUSS_NOISE_PARAM[@]}"
    do
        combined=$noise"_"$correct
        echo "Running Gaussian Noise @ $noise with Median correction @ $correct"
        sudo $MATLAB -nodisplay -nosplash -r \
        "wider_eval(\
        \"$ROOT/CORRECTIONS/gaussian_noise_median/$combined/dsfd\", \
        \"dsfd_gaussian_noise_median_$combined\", \
        \"$ROOT/results/dsfd/gaussian_noise_median_"$combined"_map.txt\"); \
        exit()"
    done
done

# poission + median
for correct in "${MEDIAN_PARAM[@]}"
do
    for noise in "${POISSON_PARAM[@]}"
    do
        combined=$noise"_"$correct
        echo "Running Poisson @ $noise with Median correction @ $correct"
        sudo $MATLAB -nodisplay -nosplash -r \
        "wider_eval(\
        \"$ROOT/CORRECTIONS/poisson_median/$combined/dsfd\", \
        \"dsfd_poisson_median_$combined\", \
        \"$ROOT/results/dsfd/poisson_median_"$combined"_map.txt\"); \
        exit()"
    done
done

# gamma + he
for correct in "${HE_PARAM[@]}"
do
    for noise in "${GAMMA_PARAM[@]}"
    do
        combined=$noise"_"$correct
        echo "Running Gammma @ $noise with HE correction @ $correct"
        sudo $MATLAB -nodisplay -nosplash -r \
        "wider_eval(\
        \"$ROOT/CORRECTIONS/gamma_he/$combined/dsfd\", \
        \"dsfd_gamma_he_$combined\", \
        \"$ROOT/results/dsfd/gamma_he_"$combined"_map.txt\"); \
        exit()"
    done
done

cd $ROOT