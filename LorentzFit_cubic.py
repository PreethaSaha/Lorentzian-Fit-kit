####-----------Made by Preetha Saha--------------------------------####

#To start: Please keep this script in the working directory!!
# This script does a Lorentzian fit with a cubic baseline or background. It assumes that the input data are numpy arrays(float 64) having a minimum of 20 (X,Y) datapoints. If not 20 datapoints, the user should change N accordingly.

# Users can import the functions: lorentzian, initial_params, fit_lorentzian
#using the syntax given below. Please check their corresponding return variables.


#For example:
# from LorentzFit import cubic_lorentzian, initial_params, fit_lorentzian_cubic
# Yfit, params, covar, perr, r2 = fit_lorentzian(X, Y,  p0=p0, plot=False, plot_path=None)

# If plot=True, plot_path=user_defined_path, it will plot the data and its fit and save with the name 'test.png' in the given path.

# Y_fit : Fitted Y array
# popt : fit estimates
# pcov : var-cov matrix
# perr : 1-sigma error of the fit estimates
# r2   : goodness of fit


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Callable, List, Optional, Tuple
from pathlib import Path
#from argparse import ArgumentParser, Namespace


                    
          
######### moving_average : function which does the work as the name suggests :) ##################

def moving_average(n, array):
    temp = []
    for i in range(len(array)-n):
        temp.append(np.mean(array[i:i+n]))
        
        
    return np.array(temp)
    
################ -------------o------------------ ######################################

############ fit functions defined ##################
def cubic(x, a, h, b, c, d):
    return a*((x-h)**3)+ b*((x-h)**2) + c*(x-h) + d
    
    
def cubic_lorentzian(x, k, gamma, x0, a, b, c ,d):
        return (k * ((gamma / 2)**2) / ((gamma / 2) ** 2 + (x - x0) ** 2) +
                cubic(x, a, x0, b, c, d)
                )
                
#####--------------o-----------------#######################

##### Bounds defined for the fit function #########################
def Bounds(X):
    
    kMin, kMax = -np.inf ,np.inf
    gammaMin, gammaMax = -0.5 * (np.max(X) - np.min(X)), 0.5 * (np.max(X) - np.min(X))
    x0Min, x0Max = np.min(X), np.max(X)
    aMin, aMax = -np.inf, np.inf
    bMin, bMax = -np.inf, np.inf
    cMin, cMax = -np.inf, np.inf
    dMin, dMax = -np.inf, np.inf
    
    return ([kMin, gammaMin, x0Min, aMin, bMin, cMin, dMin], [kMax, gammaMax, x0Max, aMax, bMax, cMax, dMax])
#######----------------o-------------------------########################

############ initial_params: function which finds the true maxima or minima using moving_average and differentiating the original data. Subsequently, it fits a cubic poly to the baseline which are then passed as the informed initial guess for Lorentzian fit with cubic baseline. #########################

def initial_params(X: np.array, Y: np.array) -> List[float]:

    print("---------Data Metrics------------")
    print(f"Your input X,Y data : {type(X)}, {type(Y)}")
    if type(X) is not np.ndarray or type(Y) is not np.ndarray:
        print(f"Please change the input data type to numpy array of float64!!")
    #Check for float type
    print(f"Total no. of X,Y datapoints: {len(X)}, {len(Y)}")
    if len(X) != len(Y):
        print(f"The length of the input numpy arrays should be equal to work with!")
    
    N=int(len(X)/10) #If less than 20 data points, need to modify this accordingly!
    print(f"No. of datapoints in first 1/10th of data = {N}")
    sigma = np.std(Y[0:N])
    print(f"Standard deviation of data = {sigma} \t considering the first 1/10th of the data, supposedly with no peak or dip.\n")
    
    '''User can modify this depending on how noisy the data is and the degree to smoothen it'''
    n_moving = 20
    print(f"No. of datapoints used for smoothening = {n_moving} \t #Users can modify this number according to their need! This smoothening does not affect the final fit procedure.\n")
    foo = np.diff(moving_average(n_moving,Y))
    
    #-----Plot or save to see/compare the smoothened & differentiated data----------------#
    plt.plot(X[:-n_moving-1],np.diff(moving_average(n_moving,Y)))
    plt.plot(X[:-n_moving-1],Y[:-n_moving-1]) # Original data
    plt.show()
    #plt.savefig("random.png",dpi=300)
    
    #-------Zero-crossing index --------------------------------------------#
    min_index_foo = np.argmin(foo)
    max_index_foo = np.argmax(foo)
    max_index = int((max_index_foo + min_index_foo)/2) # zero-crossing index
    

    print(f"Value of X = {X[max_index]} \t at zero-crossing index = {max_index}")
    print(f"Value of Y = {Y[max_index]} \t at zero-crossing index = {max_index}\n")
    
    
    
    N = 5                           # N=int(0.1/step_size)
    X_poly = X[0:max_index-N]
    X_poly = np.append(X_poly,X[max_index+N:])
    X_poly = X_poly - X[max_index]

    Y_poly = Y[0:max_index-N]
    Y_poly = np.append(Y_poly,Y[max_index+N:])
    X_poly = np.array(X_poly,dtype=np.float64)
    Y_poly = np.array(Y_poly,dtype=np.float64)
    z = np.polyfit(X_poly, Y_poly, 3)
    p3 = np.poly1d(z)
    
    
    gamma0 = np.abs(X[max_index_foo]-X[min_index_foo])
    
    offset0 = 0
    
    x00 = X[max_index]
      
    
    k0 = np.max(Y)-p3(X_poly[np.argmax(Y_poly)])            #refer to the documentation of polyfit and poly1d.
                                                    
    a0 = z[0]
    b0 = z[1]
    c0 = z[2]
    d0 = z[3] #np.max(Y)
    
    if (min_index_foo < max_index_foo) and (Y[max_index]<30.0*sigma):
        print(f"Found a dip!!\n")
        #k0 = np.max(Y)-np.mean(Y[0:20])
        #k0 = Y[max_index]-np.mean(Y[0:N])
        k0 = np.max(Y)-p3(X_poly[np.argmax(Y_poly)])

    elif (min_index_foo > max_index_foo) and (Y[max_index]>30.0*sigma):
        print(f"Found a peak!!\n")
        #k0 = np.max(Y)-np.mean(Y[0:20])
        #k0 = Y[max_index]-np.mean(Y[0:N])
        k0 = np.max(Y)-p3(X_poly[np.argmax(Y_poly)])
        
    else:
        print("Found neither a peak nor a dip!!\n")
        k0 = 0.0
    
    p0 = [k0, gamma0, x00, a0,b0,c0,d0]
    print("Initial params [k0, gamma0, x00, a0, b0, c0, d0]: ", p0)
    
    
    return p0
   
########### end of initial_params func ###########################
  
    
 ############### fit_lorentzian: func which fits a lorentzian to the data ###########################

#--------Fit optimisation starts here ------------------------#

def fit_lorentzian_cubic(
    X: np.ndarray,
    Y: np.ndarray,
    p0: List[float],
    plot: bool = True,
    plot_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    
    
    p0=initial_params(X,Y)
    bounds=Bounds(X)
    print(f"Bounds[min], Bounds[max] = {Bounds(X)}\n")
    
    
    
    
    try:
        for temp_var in range(5):
            #print(temp_var,p0)
            popt = curve_fit(f=cubic_lorentzian, xdata=X, ydata=Y, p0=p0,maxfev = 10000)[0] # cubic_lorentzian defined in fitFunctions.py, called here as fit
            pcov = curve_fit(f=cubic_lorentzian, xdata=X, ydata=Y, p0=p0,maxfev = 10000)[1] # cubic_lorentzian defined in fitFunctions.py, called here as fit
            #print("check popt",popt)
            p0=popt
            
    except:
        print("Fitting fails")
        popt = [0.0]*7
        pcov = [0.0]*7
        
#------------ends here -------------------------------------#

#---------Checks for Bound errors!-------------------------------------#
    if any(popt <= Bounds(X)[0]) == True:
        print(f"WARNING: Value of the parameter estimates {popt} less than or equal to the minm bound {Bounds(X)[0]}\n")
        popt[0] = 0.0
        popt[1] = 0.0
        popt[2] = 0.0
        popt[3] = 0.0
        popt[4] = 0.0
        popt[5] = 0.0
        pcov[0] = 0.0
        pcov[1] = 0.0
        pcov[2] = 0.0
        pcov[3] = 0.0
        pcov[4] = 0.0
        pcov[5] = 0.0
       
    if any(popt >= Bounds(X)[1]) == True:
        print(f"WARNING: Value of the parameter estimates {popt} greater than or equal to the maxm bound {Bounds(X)[1]}\n")
        popt[0] = 0.0
        popt[1] = 0.0
        popt[2] = 0.0
        popt[3] = 0.0
        popt[4] = 0.0
        popt[5] = 0.0
        pcov[0] = 0.0
        pcov[1] = 0.0
        pcov[2] = 0.0
        pcov[3] = 0.0
        pcov[4] = 0.0
        pcov[5] = 0.0

#--------------------------o-----------------------------------------#

#--------Print the parameter estimates and goodness of fit --------------------#
        
        
    
    [k_fit, gamma_fit, x0_fit, a_fit, b_fit, c_fit, d_fit] = popt

    #---------1-sigma error of the estimates--------------------------#
    perr = np.sqrt(np.diag(pcov))
    print("1sigma-err=", perr)
    
    # -----------calculate goodness of fit using R^2----------------#
    X_fit = np.linspace(X[0],X[-1],int(1000*len(X)))
    Y_fit = cubic_lorentzian(X, *popt)
    # calculate R^2, the length of X and Y arrays should be equal !
    r2 = round(1 - np.var(Y - Y_fit) / np.var(Y), 3)
    
    #res_freq = 2.9979e8 / (x0_fit * 1e-9) * 1e-12 # convert wavelength to THz
    FWHM = np.abs(gamma_fit)
    print("k_fit or fit value of amplitude:", k_fit)
    print("gamma_fit or fit value of gamma:", np.abs(gamma_fit))
    print("x0_fit or fit value of resonance:", x0_fit)
    print("a_fit or fit value of cubic bkgd:", a_fit)
    print("b_fit or fit value of cubic bkgd:", b_fit)
    print("c_fit or fit value of cubic bkgd:", c_fit)
    print("d_fit or fit value of background level:", d_fit)
    Amp_fit = k_fit
    print("r2:", r2)

#---------------------------o-------------------------------------------------#
    
    
#------OPTIONAL - if the Plot bool is True ------------------------------------#
    
    Y_fit = cubic_lorentzian(X_fit, *popt)
    
    if plot:
        # plot the original data
        plt.scatter(X, Y, color="black", marker=".",alpha = 0.5, label="Data")
        # plot Lorentzian fit
        plt.plot(
                 X_fit,
                 Y_fit,
                 color="red",
                 alpha=0.5,
                 label=f"Lorentzian fit \n x0={round(x0_fit,3)} \n R2={r2} \n Bkgd fit type: Cubic",
                        )
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        #plt.grid(True)
        plt.legend()
        #plt.show()
        print("Done plotting Lorentzian fit with data!")
        plt.savefig(fname="test.png", dpi=300)
        #print(filename_dev+str(i).jpg)
        print("Done saving the plot!\n")
        
    return Y_fit, popt, pcov, perr, r2

############### returns the fit Y values, fitting parameters estimates, var-cov matrix, 1-sigma error of parameter estimates and goodness of fit(r2) ###########################






    

