DWI_shell can be used for reconstructing radially acquired DWIs, obtaining apparent diffusion coefficient (ADC) and kurtosis index (KI) maps from a set of DWIs, and for obtaining metrics for regions of interest (ROIs).

Requires the following libraries:
numpy
matplotlib
scipy
math
cmath
pandas
os

Setup:
- DWI_shell.ipynb and function_file.py must be in same directory (your working directory)


Image reconstruction:
1. Place k-space data file in working directory
2. Change parameters in "parameters to change" section
3. Run first and second cells

reconstruct_radial_DWIs():
Inputs (* required):
	- input_file*		: k-space data file
	- img_dim*			: list containing image dimensions
	- kspace_dim*		: list containing k-space dimensions
Output:
	- dwis 			: a numpy array of the reconstructed DWIs
	- AllSlicesBvalues.bin 	: a 64-bit floating point raw image file of the reconstructed DWIs


Parametric fitting:
- diffusion_fits() can be run using either a variable in the current runtime containing the DWIs (the output of reconstruct_radial_DWIs() ) or from raw image files in the working directory
Inputs (* required):
	- b_array* 			: a numpy array containing the b-values of the DWIs
	- SNR_threshold		: minimum SNR to be identified as "not background" and thus fitted
	- noise_region		: region to be defined as "noise" [x1,x2,y1,y2]; top left corner by default
	and either*
	- dwis			: a numpy array containing the DWIs
	or
	- input_file		: raw image file containing DWIs in working directory
	- img_dims			: list containing image dimensions
Outputs:
	- dwi_fits: 4D numpy array containing parametric fits and fitting information
		dwi_fits[0,:,:,:] : ADC maps
		dwi_fits[1,:,:,:] : KI maps
		dwi_fits[2,:,:,:] : fitting information;
		  if 0: fitting converges, if 1: SNR below threshold, if 0.5: fitting does not converge
	ADCMaps.bin			: a 64-bit floating point raw image file of the ADC maps
	KurtosisMaps.bin		: a 64-bit floating point raw image file of the Kurtosis Index (KI) maps
	ErrorMaps.bin		: a 64-bit floating point raw image file of the fitting information above


ROI analysis:
- ROI_analysis() can be run using either a varible in the current runtime containing the parametric fits (the output of diffusion_fit() ) or from raw image files in the working directory
- The mask file is a raw, 8-bit image file of the same dimensions as a DWI (ie. [slices, yres, xres]), where non-ROI voxels = 0 and ROIs are integers starting at 1.

Inputs (required*):
	- mask_file*		: raw image file containing tissue masks
	- img_dims*		: list containing image dimensions
	- index			: list containing mask index (if given, must have label for every ROI)
	and either*
	- dwi_fits		: 4D numpy array containing parametric fits (output from diffusion_fits() )
	or
	- ADCMaps_file		: raw image file of ADC maps
	- KurtosisMaps_file	: raw image file of kurtosis index maps
