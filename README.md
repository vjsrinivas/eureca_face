# EUReCA 2021 - Face Detector Sensitivity Analysis

##Face Detectors In Use:
- [x] TinaFace
- [x] RetinaFace
- [-] DSFD
- [-] S3FD
- [ ] **PyramidBox** - Not sure if I can use this one, it's completely in Chinese and the alternative ports are not accurate enough.

## Structure:
- The direct children folders represent just the raw source codes from the detectors. I will have to take them apart and run them seperately inside the folder `eureca_face`.
- `eureca_face` will contain all the major code that can/will be runnable from a fresh installation. It will hold the evaluation results and processed information.
- `WIDERFACE` folder will hold ALL of the preprocessed images

## Notes:
- Correction processing will be done during inferencing

## Types of processing:

### Degradation:
	- Noise:
		- Pepper and Salt (reference Matthew's paper)
		- Gaussian Noise (control noise)
		- Speckle Noise (granular interference)
		- Poisson's Noise

### Recovery:
	- Noises:
		- Non-linear Filtering:
			- Median Filtering (Gaussian and Speckle Noise)
			- Max/Min Filter (Salt&Pepper)
