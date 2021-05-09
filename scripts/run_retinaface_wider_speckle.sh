cd ../
cd ./WIDERFACE/eval_tools/

ROOT="/home/vijay/Documents/devmk4/eureca/eureca_face"
MATLAB=/usr/local/MATLAB/R2020b/bin/matlab

#NOISE_PARAM=( -1.000000 -0.750000 -0.500000 -0.250000 0.000000 0.250000 0.500000 0.750000 1.000000 )
NOISE_PARAM=( -1.000000 -0.750000 -0.500000 -0.250000 )
for noise in "${NOISE_PARAM[@]}"
do
	echo "Running " $noise "..."
	sudo $MATLAB -nodisplay -nosplash -r \
	"wider_eval(\
	\"$ROOT/NOISES/speckle/$noise/detections/retinaface\", \
	\"RetinaFace_retinaface_speckle_$noise\", \
	\"$ROOT/results/retinaface/speckle_"$noise"_map.txt\"); \
	 exit()"
done

cd ../../