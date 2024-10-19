#To start: Please keep this script in the working directory!!
# This script does a Lorentzian fit with a constant baseline or background. It assumes that the input data are numpy arrays(float 64) having a minimum of 20 (X,Y) datapoints. If not 20 datapoints, the user should change N accordingly.

# Users can import the functions: lorentzian, initial_params, fit_lorentzian
#using the syntax given below. Please check their corresponding return variables.


#For example:
# from LorentzFit import lorentzian, initial_params, fit_lorentzian
# Yfit, params, covar, perr, r2 = fit_lorentzian(X, Y, plot=False, plot_path=None)

# If plot=True, plot_path=user_defined_path, it will plot the data and its fit and save with the name 'test.png' in the given path. 

# Y_fit : Fitted Y array
# popt : fit estimates
# pcov : var-cov matrix
# perr : 1-sigma error of the fit estimates
# r2   : goodness of fit


########## Import the required dependencies #############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Callable, List, Optional, Tuple
from pathlib import Path

import warnings
from scipy.optimize import OptimizeWarning
# Configure warnings to raise exceptions
warnings.simplefilter("error", OptimizeWarning)


#import xlsxwriter
#import csv
#import os



                    
          
######### moving_average : function which does the work as the name suggests :) ##################

def moving_average(n, array):
    temp = []
    for i in range(len(array)-n):
        temp.append(np.mean(array[i:i+n]))
        
        
    return np.array(temp)
    
################ -------------o------------------ ######################################


###### fit functions defined #############################
                
def lorentzian(x, k, gamma, x0, d):
        return (k * ((gamma / 2)**2) / ((gamma / 2) ** 2 + (x - x0) ** 2) + d
                )
                
##########----------o----------------##########################

##### Bounds defined for the fit function #########################
def Bounds(X):
    
    kMin, kMax = -np.inf ,np.inf
    gammaMin, gammaMax = -0.5 * (np.max(X) - np.min(X)), 0.5 * (np.max(X) - np.min(X))
    x0Min, x0Max = np.min(X), np.max(X)
    dMin, dMax = -np.inf, np.inf
    
    return ([kMin, gammaMin, x0Min, dMin], [kMax, gammaMax, x0Max, dMax])
#######----------------o-------------------------########################
    

############ finds the true maxima or minima using moving_average and differentiating the spectrum. A set of initial guesses are passed for Lorentzian fit with constant baseline or background. #########################

def initial_params(X: np.array, Y: np.array,n_moving) -> List[float]:

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
   # n_moving = 20
    print(f"No. of datapoints used for smoothening = {n_moving} \t #Users can modify this number according to their need! This smoothening does not affect the final fit procedure.\n")
    foo = np.diff(moving_average(n_moving,Y))
    
    #-----Plot or save the smoothened and differentiated data----------------#
    plt.plot(X[:-n_moving-1],np.diff(moving_average(n_moving,Y)))
    plt.plot(X[:-n_moving-1],Y[:-n_moving-1])
   # plt.show()
    #plt.savefig("random.png",dpi=300)
    
    #-------Zero-crossing index --------------------------------------------#
    min_index_foo = np.argmin(foo)
    max_index_foo = np.argmax(foo)
    max_index = int((max_index_foo + min_index_foo)/2) # zero-crossing index
    #print(X[min_index_foo], X[max_index_foo])

    
    print(f"Value of X = {X[max_index]} \t at zero-crossing index = {max_index}")
    print(f"Value of Y = {Y[max_index]} \t at zero-crossing index = {max_index}\n")
    '''
    Y_excluded=Y[max_index-50:max_index+50]
    #print(Y_excluded)
    #Y_excluded = np.delete(Y, Y_excluded)
    print(Y_excluded)
    mask = np.ones(Y.shape, dtype=bool)
    mask[Y_excluded] = False
    Y_excluded = Y[mask]
    #print(Y_excluded)
    sigma = np.std(Y_excluded)
    #print(f"Standard deviation of data = {sigma} \t considering the first 1/5th of the data, supposedly with no peak or dip.\n")
    '''
    #-----------initial guesses ---------------------------#
    
    
    gamma0 = np.abs(X[max_index_foo]-X[min_index_foo]) # always positive
    
    
    x00 = X[max_index]
    
    #d0 = 0.0
    d0 = np.mean(Y[0:N]) #considering the first 1/10th of the data
    
        ##----True peak or dip should be greater or lesser than (30 times the sigma ---> maybe needs modification depending on quality of data) respectively----------------###
    
    if (min_index_foo < max_index_foo) and (Y[max_index]<30.0*sigma):
        print(f"Found a dip!!\n")
        #k0 = np.max(Y)-np.mean(Y[0:20])
        k0 = Y[max_index]-np.mean(Y[0:N])

    elif (min_index_foo > max_index_foo) and (Y[max_index]>30.0*sigma):
        print(f"Found a peak!!\n")
        #k0 = np.max(Y)-np.mean(Y[0:20])
        k0 = Y[max_index]-np.mean(Y[0:N])
        
    else:
        print("Found neither a peak nor a dip!!\n")
        k0 = 0.0
       
    
    
    p0 = [k0, gamma0,x00, d0]
    print(f"Initial params [k0, gamma0, x00, d0]: {p0}")
    

    return p0
    
 ############### fit_lorentzian: func which fits a lorentzian the peak of the spectrum ###########################
#Y_excluded=[]

#--------Fit optimisation starts here ------------------------#
def fit_lorentzian(
    X: np.ndarray,
    Y: np.ndarray,
    plot: bool = True,
    plot_path: Optional[Path] = None,n_moving=10
) -> Tuple[np.ndarray, np.ndarray, float]:
    
    p0=initial_params(X,Y,n_moving)
    bounds=Bounds(X)
    print(f"Bounds[min], Bounds[max] = {Bounds(X)}\n")
    try:
        for temp_var in range(5): # Not important----Just to check whether it returns the same fit estimates for greater iterations!!
            
            #print(temp_var,p0)
            popt,pcov = curve_fit(f=lorentzian, xdata=X, ydata=Y, p0=p0, maxfev = 10000)
            
            p0=popt # to ensure that at each iteration that estimates of the previous iteration is taken as initial guesses
            
            #pass
    #except OptimizeWarning as e:
       # raise RuntimeError("Optimization warning occurred") from e
    except:
        print("Fitting fails\n")
        popt = [0.0]*4
        pcov = [0.0]*4
     
#------------ends here -------------------------------------#
        
#---------Checks for Bound errors!-------------------------------------#
    if any(popt <= Bounds(X)[0]) == True:
        print(f"WARNING: Value of the parameter estimates {popt} less than or equal to the minm bound {Bounds(X)[0]}\n")
        popt[0] = 0.0
        popt[1] = 0.0
        popt[2] = 0.0
        pcov[0] = 0.0
        pcov[1] = 0.0
        pcov[2] = 0.0
       
    if any(popt >= Bounds(X)[1]) == True:
        print(f"WARNING: Value of the parameter estimates {popt} greater than or equal to the maxm bound {Bounds(X)[1]}\n")
        popt[0] = 0.0
        popt[1] = 0.0
        popt[2] = 0.0
        pcov[0] = 0.0
        pcov[1] = 0.0
        pcov[2] = 0.0

#--------------------------o-----------------------------------------#
       
#--------Print the parameter estimates and goodness of fit --------------------#

    [k_fit, gamma_fit, x0_fit, d_fit] = popt
    #print(f"popt={popt}")
    
    #---------1-sigma error of the estimates--------------------------#
    perr = np.sqrt(np.diag(pcov))
    print("1sigma-err=", perr)
    
    
    # -----------calculate goodness of fit using R^2----------------#
    X_fit = np.linspace(X[0],X[-1],int(1000*len(X)))
    Y_fit = lorentzian(X, *popt)
    # calculate R^2 , the length of X and Y arrays should be equal !
    r2 = round(1.0 - (np.var(Y - Y_fit) / np.var(Y)), 3)
    
    
    
    FWHM = np.abs(gamma_fit)
    print("k_fit or fit value of amplitude:", k_fit)
    print("gamma_fit or fit value of gamma:", np.abs(gamma_fit))
    print("x0_fit or fit value of resonance:", x0_fit)
    print("d_fit or fit value of background level:", d_fit)
    Amp_fit = k_fit
    print("r2:", r2)
    
    
#---------------------------o-------------------------------------------------#
    
    
#------OPTIONAL - if the Plot bool is True ------------------------------------#
    Y_fit = lorentzian(X_fit, *popt)
    
    if plot:
        # plot the original data
        plt.scatter(X, Y, color="black", marker=".",alpha = 0.5, label="Data")
        # plot Lorentzian fit
        plt.plot(
                 X_fit,
                 Y_fit,
                 color="red",
                 alpha=0.5,
                 label=f"Lorentzian fit \n x0={round(x0_fit,3)} \n R2={r2} \n Bkgd fit type: Constant",
                        )
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Transmitted Power(mW)")
        #plt.grid(True)
        plt.legend()
        #plt.show()
        print("Finished plotting Lorentzian")
        plt.savefig(fname="test.png", dpi=300)
        print("Finished saving plot\n")
#----------------------o--------------------------------------------#
        
    return Y_fit, popt, pcov, perr, r2

############### returns the fit Y values, fitting parameters estimates, var-cov matrix, 1-sigma error of parameter estimates and goodness of fit(r2) ###########################






    

