# Lorentzian-Fit-kit

## Description:

This repository contains code for fitting Lorentzian curves with different types of baseline function. 
The Lorentzian curve is a common model used in numerous scientific domains to characterize spectral lines, resonances, and other phenomena exhibiting a distinctive peak shape. The inclusion of various baseline functions enables users to account for and model the underlying background signal inherent in experimental data, which may vary due to factors such as noise, drift or systematic effects.

The Lorentzian function $L(x)$ is defined as

$L(x) = k\frac{(\Gamma/2)^2}{(x-x_0)^2+(\Gamma/2)^2}$

where $k$ is the amplitude, $x_0$ is the center and $\Gamma$ is the full-width at half-maxima (FWHM).

 

## Installation:
To use this code, follow these steps:

1. Clone the repository

   git clone https://github.com/PreethaSaha/Lorentzian-Fit-kit.git

3. Navigate to the project directory

   cd Lorentzian-Fit-kit

5. Install the required dependencies

   pip install -r requirements.txt

## Usage:

Currently the types of baseline include and the respective codes are linked below:

1. [Constant: LorentzFit.py](https://github.com/PreethaSaha/Lorentzian-Fit-kit/blob/main/LorentzFit.py) 
2. [Cubic: LorentzFit_cubic.py](https://github.com/PreethaSaha/Lorentzian-Fit-kit/blob/main/LorentzFit_cubic.py)

The above codes are designed to find a true Loretzian peak/dip and also handle noisy data by easily customisable parameters.

## Input Data Format:

The input data should contain two columns of equal length: one for the independent variable X (e.g., wavelength, frequency) and another for the corresponding dependent variable Y (e.g., intensity, amplitude). 

Note that the code assumes an input data consisting of a minimum 20 (X,Y) datapoints.

## Output:

The script prints the parameters of the fitted Lorentzian curve: $k$, $x_0$, $\Gamma$ and the respective parameters of the fitted baseline. Optionally, it generates a plot showing the original data points along with the fitted Lorentzian curve and the selected baseline, in an user-defined path and an output file name 'test.png'.

## License:

This project is licensed under the MIT License - see the [LICENSE](https://github.com/PreethaSaha/Lorentzian-Fit-kit/blob/main/LICENSE) file for details.



