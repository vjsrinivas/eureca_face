cd ../
cd ./WIDERFACE/eval_tools/

ROOT="/home/vijay/Documents/devmk4/eureca/eureca_face"
MATLAB=/usr/local/MATLAB/R2020b/bin/matlab

NOISE_PARAM=( 0.000000 10.000000 20.000000 30.000000 40.000000 50.000000 60.000000 70.000000 80.000000 90.000000 100.000000 )
for noise in "${NOISE_PARAM[@]}"
do
	echo "Running " $noise "..."
	sudo $MATLAB -nodisplay -nosplash -r \
	"wider_eval(\
	\"$ROOT/NOISES/gaussian_noise/$noise/detections/retinaface\", \
	\"RetinaFace_retinaface_gaussian_noise_$noise\", \
	\"$ROOT/results/retinaface/gaussian_noise_"$noise"_map.txt\"); \
	 exit()"
done

cd ../../