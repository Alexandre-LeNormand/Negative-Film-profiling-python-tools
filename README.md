# Negative-Film-profiling-python-tools

Automated Python tools to help with gathering, calibrating and profiling negative film characteristics in RGB data. I didn't invent this method for profiling. Rather,
these are simple tools to skip some manual labour involved in film profiling. All the tools as based on Nico Fink's method from the Film Profile Journey tutorial series.
Go check his website : https://www.demystify-color.com/.

## Contents

- [Installation](#installation)
- [Tools](#Tools)
  - Film_Colorchecker_Calibration_&_Library
  - Get_Colourchecker_RGB_Data
  - RBF_Solver
- [Usage](#usage)
- [Requirements](#requirements)
- [Contributing](#contributing)

## Installation

Clone this repository:
```
git clone https://github.com/Alexandre-LeNormand/Negative-Film-profiling-python-tools.git
cd Negative-Film-profiling-python-tools
```

Install dependencies:
```
pip install -r requirements.txt
```

## Tools

### Film_Colorchecker_Calibration_&_Library

This script acts as a library for characteristic curve data and a calibration tool for digital scans of film negatives. It takes real-world
Cineon 10-bit float scan values of a ColorChecker Classic 24 (6 neutral patches) and finds the best-fit position of those patches on a film stock manufacturer
characteristic curve. Outputs a .cube correction 1D LUT that brings the given scan closer to the manufacturer's intent. 

Each characteristic curve is added into a film stock library and can be viewed, modified or deleted. New film stocks can be added to the library by entering 17 data points for 
each axis of the characteristic curve. The script supports two types of values for the X axis: Stops and Log Exposure. For the Y axis, using the method developped by Nico Fink,
the script assumes density values aligned to 0. 

The tool assumes the film scan is inverted to a positive and is in Cineon Log Gamma. 

The pipeline the script uses to find the best fit of the curve is as follow : 
  1. Convert scan Cineon floats → Density Above DMin
  2. Use the GREEN channel across all 6 patches to find X0 —
     the single log exposure ANCHOR that best aligns the measured
     densities to the emulsion curve (least-squares fit).
     The fit uses the same exposure-shifted spacing as the ideal-output
     lookup, so the exposure offset is encoded only once.
  4. From X0, derive the ideal log exposure for every patch:
         patch_logexp[i] = X0 + EXPOSURE_PATCH_SPACINGS[stop][i]
     For a bracketed shot the whole ladder shifts: e.g. at +1 stop the
     six patches occupy curve positions one stop higher than normal.
     X0 is the virtual anchor of whichever shifted ladder is active —
     not the absolute log exposure of middle grey (except when stop=0).
  5. Look up ideal density on each channel's emulsion curve.
  6. Convert ideal density → Cineon float output.
  7. Report: scan IN values vs ideal OUT values for DaVinci Resolve.

The tool supports manually typed cineon values and CSVs in single or bracketed exposures of up to -5 to +5EV. In this case, the (#Get_Colourchecker_RGB_Data) script
featured in this repo is usefull for automatically measuring and placing the values in separate CSVs for each exposure. The patches are read from brightest to darkest.

### Get_Colourchecker_RGB_Data
Reads X-Rite ColorChecker Classic (24-patch, 4×6) and/or
X-Rite ColorChecker SG (140-patch, 10×14) from a single image,
and/or any number of manually-placed square readings.

Workflow :
1. Load Image
2. (Optional) Click 4 corners of the 24-patch chart  → TL, TR, BR, BL
3. (Optional) Click 4 corners of the 140-patch chart → TL, TR, BR, BL
4. Set the sample-square size (pixels) per chart
5. (Optional) Set manual-patch size, click "Place Patches", then click
   anywhere on the image.  Right-click removes the last placed patch.
6. Click "Extract & Save CSV"

CSV layout  →  R,G,B  (one row per patch)
            SG 140-patch rows come first, then 24-patch rows,
            then manually-placed patches in the order they were placed.

### RBF_Solver

This tool creates a .cube 3D LUT based on two CSV datasets. The (#Get_Colourchecker_RGB_Data) script featured in this repo is usefull for creating the CSVs.

Radial Basis Function (RBF) interpolation enables creation of a smooth 3D LUT to map complex color datasets, capturing nonlinear responses which is ideal for matching
film to digital.

The user can define, the kernel used, smoothing parameter and size of the output LUT.

There are multiple kernel options available to compute the LUT. Thin_plate_splines gives with a smoothing of 0.4 gives nice results for film-digital matching. 

## Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib
- OpenCV or scikit-image for image processing
- SciPy for curve fitting

See `requirements.txt` for full list.
