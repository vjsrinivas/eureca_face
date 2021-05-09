cd ../
cd ./WIDERFACE/eval_tools/

ROOT="/home/vijay/Documents/devmk4/eureca/eureca_face"
MATLAB=/usr/local/MATLAB/R2020b/bin/matlab

NOISE_PARAM=( 0.000000 0.100000 0.150000 0.200000 0.250000 0.300000 0.350000 0.400000 0.450000 0.500000 )
for noise in "${NOISE_PARAM[@]}"
do
	echo "Running " $noise "..."
	sudo $MATLAB -nodisplay -nosplash -r \
	"wider_eval(\
	\"$ROOT/NOISES/salt_pepper/$noise/detections/dsfd\", \
	\"dsfd_salt_pepper_$noise\", \
	\"$ROOT/results/dsfd/salt_pepper_"$noise"_map.txt\"); \
	 exit()"
done

cd ../../