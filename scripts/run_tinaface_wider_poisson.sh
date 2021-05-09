cd ../
cd ./WIDERFACE/eval_tools/

ROOT="/home/vijay/Documents/devmk4/eureca/eureca_face"
MATLAB=/usr/local/MATLAB/R2020b/bin/matlab

NOISE_PARAM=( 0.000000 0.100000 0.200000 0.300000 0.400000 0.500000 0.600000 0.700000 0.800000 0.900000 1.000000 )
for noise in "${NOISE_PARAM[@]}"
do
	echo "Running " $noise "..."
	sudo $MATLAB -nodisplay -nosplash -r \
	"wider_eval(\
	\"$ROOT/NOISES/poisson/$noise/detections/tinaface\", \
	\"tinaface_tinaface_poisson_$noise\", \
	\"$ROOT/results/tinaface/poisson_"$noise"_map.txt\"); \
	 exit()"
done

cd ../../