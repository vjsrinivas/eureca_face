cd ./vedadet

GAUSS_NOISE_PARAM=( 0.000000 10.000000 20.000000 30.000000 40.000000 50.000000 60.000000 70.000000 80.000000 90.000000 100.000000 )
for noise in "${GAUSS_NOISE_PARAM[@]}"
do
    echo $noise gaussian_noise
    CUDA_VISIBLE_DEVICES="0" python tools/infer_wf.py --val_file ../NOISES/gaussian_noise/$noise/val.txt configs/infer/tinaface/tinaface.py 
done

# poisson:
POISSON_PARAM=( 0.000000 0.100000 0.200000 0.300000 0.400000 0.500000 0.600000 0.700000 0.800000 0.900000 1.000000 )
for noise in "${POISSON_PARAM[@]}"
do
    echo $noise poisson
    CUDA_VISIBLE_DEVICES="0" python tools/infer_wf.py --val_file ../NOISES/poisson/$noise/val.txt configs/infer/tinaface/tinaface.py 
done

# salt_pepper:
SALT_PEPPER_PARAM=( 0.000000 0.100000 0.150000 0.200000 0.250000 0.300000 0.350000 0.400000 0.450000 0.500000 )
for noise in "${SALT_PEPPER_PARAM[@]}"
do
    echo $noise salt_pepper
    CUDA_VISIBLE_DEVICES="0" python tools/infer_wf.py --val_file ../NOISES/salt_pepper/$noise/val.txt configs/infer/tinaface/tinaface.py 
done

# speckle:
#SPECKLE_PARAM=( -1.000000 -0.750000 -0.500000 -0.250000 0.000000 0.250000 0.500000 0.750000 1.000000 )
#for noise in "${SPECKLE_PARAM[@]}"
#do
#    CUDA_VISIBLE_DEVICES="0" python tools/infer_wf.py --val_file ../NOISES/speckle/$noise/val.txt configs/infer/tinaface/tinaface.py 
#done

# gamma
GAMMA_PARAM=( 0.000000 40.000000 80.000000 120.000000 160.000000 200.000000 240.000000 280.000000)
for noise in "${GAMMA_PARAM[@]}"
do
    CUDA_VISIBLE_DEVICES="0" python tools/infer_wf.py --val_file ../NOISES/gamma/$noise/val.txt configs/infer/tinaface/tinaface.py 
done

cd ../