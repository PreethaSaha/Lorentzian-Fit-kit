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

<div>
  <button class="copy-button" onclick="copyToClipboard(this.parentElement.nextElementSibling.textContent)"></button>
  <pre><code>git clone https://github.com/PreethaSaha/Lorentzian-Fit-kit.git
  </code></pre>
</div>

2. Navigate to the project directory

<div>
  <button class="copy-button" onclick="copyToClipboard(this.parentElement.nextElementSibling.textContent)"></button>
  <pre><code>cd Lorentzian-Fit-kit
  </code></pre>
</div>
   

3. Install the required dependencies

<div>
  <button class="copy-button" onclick="copyToClipboard(this.parentElement.nextElementSibling.textContent)"></button>
  <pre><code> pip install -r requirements.txt
  </code></pre>
</div>
  

## Usage:

Currently the types of baseline include and the respective codes are linked below:

1. [Constant: LorentzFit.py](https://github.com/PreethaSaha/Lorentzian-Fit-kit/blob/main/LorentzFit.py) 
2. [Cubic: LorentzFit_cubic.py](https://github.com/PreethaSaha/Lorentzian-Fit-kit/blob/main/LorentzFit_cubic.py)

The above codes are designed to find a true Loretzian peak/dip and also handle noisy data by easily customisable parameters.
```
import csv
import numpy as np

a=np.genfromtxt('test3-2.csv',delimiter=',')

wavelength=a[:,0]
intensity=a[:,1]

from LorentzFit_cubic import cubic_lorentzian, initial_params, fit_lorentzian_cubic
Yfit, params, covar, perr, r2 = fit_lorentzian_cubic(wavelength, intensity, plot=True, plot_path=None)
```

```
---------Data Metrics------------
Your input X,Y data : <class 'numpy.ndarray'>, <class 'numpy.ndarray'>
Total no. of X,Y datapoints: 250, 250
No. of datapoints in first 1/10th of data = 25
Standard deviation of data = 0.16640427151687562 	 considering the first 1/10th of the data, supposedly with no peak or dip.

No. of datapoints used for smoothening = 10 	 #Users can modify this number according to their need! This smoothening does not affect the final fit procedure.

Value of X = 14.184738955823294 	 at zero-crossing index = 67
Value of Y = 1.2261337636095613 	 at zero-crossing index = 67

Found a dip!!

Initial params [k0, gamma0, x00, a0, b0, c0, d0]:  [0.21212039406917071, 2.164658634538153, 14.184738955823294, -1.8662518305369105e-05, 0.0027543089854331257, -0.016160172446802554, 1.3412823769712963]
Bounds[min], Bounds[max] = ([-inf, -24.5, 1.0, -inf, -inf, -inf, -inf], [inf, 24.5, 50.0, inf, inf, inf, inf])

1sigma-err= [1.03629152e-01 9.08462191e-02 2.89567591e-02 4.53900487e-06
 1.54059016e-04 1.27617690e-03 2.05140555e-02]
k_fit or fit value of amplitude: -1.820282341538805
gamma_fit or fit value of gamma: 1.0176288925150385
x0_fit or fit value of resonance: 15.182843697708726
a_fit or fit value of cubic bkgd: 1.578617047797091e-06
b_fit or fit value of cubic bkgd: 0.0019507837595691625
c_fit or fit value of cubic bkgd: -0.011186826924691739
d_fit or fit value of background level: 1.4569093814119933
r2: 0.952
Done plotting Lorentzian fit with data!
Done saving the plot!
```

## Input Data Format:

The input data should contain two columns of equal length: one for the independent variable X (e.g., wavelength, frequency) and another for the corresponding dependent variable Y (e.g., intensity, amplitude). 

Note that the code assumes an input data consisting of a minimum 20 (X,Y) datapoints.

## Output:

The script prints the parameters of the fitted Lorentzian curve: $k$, $x_0$, $\Gamma$ and the respective parameters of the fitted baseline. Optionally, it generates a plot showing the original data points along with the fitted Lorentzian curve and the selected baseline, in an user-defined path and an output file name 'test.png'.


## License:

This project is licensed under the MIT License - see the [LICENSE](https://github.com/PreethaSaha/Lorentzian-Fit-kit/blob/main/LICENSE) file for details.



