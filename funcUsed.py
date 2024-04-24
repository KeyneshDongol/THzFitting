import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial
import json
import time
from math import pi
from matplotlib.gridspec import GridSpec
# from Error_function import E_air_exp_f, E_air_theo_f, E_air_theo_t, E_sample_exp_f, E_TMM_new, calculate_error
from Functions.Read_material import Material
from Functions import exp_pulse, fourier, tools
from scipy.optimize import minimize
from Functions.mainfunction import E_TMM, Error_func,  noise_remove
# from toIntroduce import E_AirInOriginal
from setup_attributes import Attributes, ErrorFunction
from MaterialLayer import MaterialData, LayerStack
# from Error_function import E_TMM_new_

if __name__ == '__main__':

    '''
    Initialize Atrributes and Experimental Data
    '''
    # Assuming you have the classes Attributes, ExperimentalData, etc., already defined and imported
    attributes = Attributes('inputs.json')

    # Process experimental data, which is done during initialization of Attributes
    print("Experimental Data Processed:")
    print("Omega (Frequency Range):", attributes.experimental_data.omega)
    print("Processed Signals (Example Outputs):", attributes.experimental_data.E_air_f)
    
    '''
    Error Calculation and Optimzation
    '''
    # Initialize ErrorFunction
    error_func = ErrorFunction(
        experimental_data=attributes.experimental_data,
        eps0=attributes.eps0,
        mu=attributes.mu,
        to_find=attributes.to_find,
        sub_layer=attributes.sub_layer,
        echoes_removed=attributes.echoes.Removed,
        layer_thicknesses=attributes.layer_thicknesses
        )   

    # Optimize parameters for the best fit model
    n_range = (2, 3)  # Hypothetical range for the refractive index
    k_range = (0, 0.01)  # Hypothetical range for the extinction coefficient
    optimal_params = error_func.optimize_parameters(n_range=n_range, k_range=k_range)
    print("Optimization Results: n and k values along with error matrix.")
    print("n Values:", optimal_params[0])
    print("k Values:", optimal_params[1])
    print("Error Matrix:", optimal_params[2])
    
    
    layer_stack = attributes.layer_stack
    print(layer_stack)