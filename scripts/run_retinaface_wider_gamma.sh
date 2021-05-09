cd ../
cd ./WIDERFACE/eval_tools/

ROOT="/home/vijay/Documents/devmk4/eureca/eureca_face"
MATLAB=/usr/local/MATLAB/R2020b/bin/matlab

#NOISE_PARAM=( 0.000000 40.000000 80.000000 120.000000 160.000000 200.000000 )
NOISE_PARAM=( 240.000000 280.000000 )
for noise in "${NOISE_PARAM[@]}"
do
	echo "Running " $noise "..."
	sudo $MATLAB -nodisplay -nosplash -r \
	"wider_eval(\
	\"$ROOT/NOISES/gamma/$noise/detections/retinaface\", \
	\"RetinaFace_gamma_$noise\", \
	\"$ROOT/results/retinaface/gamma_"$noise"_map.txt\"); \
	 exit()"
done

cd ../../