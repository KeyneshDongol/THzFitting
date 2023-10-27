import numpy as np
from scipy.optimize import minimize, Bounds
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial
import json
import time
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from Functions.Read_material import Material
from Functions.TMM import SpecialMatrix
from Functions import exp_pulse, fourier, tools

from scipy.optimize import minimize, LinearConstraint

# returns the theoretically transmitted pulse (in both time and frequency domain)
def E_TMM(layers, to_find, omega, eps0, mu, d, f_in,sub_layer, echoes_removed, unknown):  
    epsilon = []
    for l in layers:
        if type(l) == str:  
            if to_find[0] == True:                                        # permittivity
                epsilon.append(unknown*eps0)
            if to_find[1] == True:                                        # plasma and damping
                eps = Material(omega).drude(unknown[0], unknown[1]);  eps= 1*eps;  eps[0] = eps[1]
                epsilon.append(eps*eps0)
            if to_find[2] == True:                                        # n and k
                epsilon.append(Material(omega).epsilon(unknown)*eps0)
        else:
            epsilon.append(l)

    '''Transfer matrix'''
    TMM = SpecialMatrix(omega, mu, epsilon, d)
    _, _, T_0inf = TMM.Transfer_Matrix()
    T_0s  = TMM.Transfer_Matrix_special_0N(sub_layer)
    T_sinf  = TMM.Transfer_Matrix_special_Ninf(sub_layer)

    '''Tranmission & reflection coefficients'''
    t_0s        = TMM.Transmission_coeff(T_0s)
    t_sinf      = TMM.Transmission_coeff(T_sinf)
    t           = TMM.Transmission_coeff(T_0inf)
    t_noecho    = np.multiply(t_0s, t_sinf)

    '''Remove echo or not'''    
    if echoes_removed[0]==True:
        f_inf_R      = f_in * t_noecho 
    else:
        f_inf_R      = f_in*t
        
    '''Transmitted wave in freq domain'''
    trans = fourier.ift(f_inf_R)    
    return trans, f_inf_R


''' error function without penalty term'''
# def Error_func(layers, to_find, omega, eps0, mu, d, E_air_f, E_exp_f,sub_layer,echoes_removed, unknown):
#     if to_find[0] == True:
#         unknown = unknown[:len(unknown)//2] + 1j*unknown[len(unknown)//2:]
#     if to_find[2] == True:
#         unknown = np.array(np.array_split(unknown, 2))    
#     E_theo_f = E_TMM(layers, to_find, omega, eps0, mu, d, E_air_f, sub_layer,echoes_removed,unknown)[1]    
#     return np.sum(np.abs(E_theo_f - E_exp_f))


''' error function with penalty term'''
def Error_func(layers, to_find, omega, eps0, mu, d, E_air_f, E_exp_f, sub_layer, echoes_removed, unknown):
    if to_find[0] == True:
        unknown = unknown[:len(unknown)//2] + 1j*unknown[len(unknown)//2:]
    if to_find[2] == True:
        unknown = np.array(np.array_split(unknown, 2))

    # Additional penalty term to enforce positive unknown[1]
    penalty = 0.0
    if to_find[2] == True:
        negative_k = np.where(unknown[1] < 0)[0]
        if len(negative_k) > 0:
            penalty = np.sum(np.exp(-unknown[1][negative_k]))

    E_theo_f = E_TMM(layers, to_find, omega, eps0, mu, d, E_air_f, sub_layer, echoes_removed, unknown)[1]
    
    return np.sum(np.abs(E_theo_f - E_exp_f)) + penalty


'''result smoothing'''
def moving_average(data, window_size):
    # Pad the data at the beginning and end to avoid edge effects
    padded_data = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
    
    # Apply the moving average filter
    smoothed_data = np.convolve(padded_data, np.ones(window_size)/window_size, mode='valid')
    
    return smoothed_data


if __name__ == '__main__':
    
    '''inputs'''
    f = open(Path.cwd()/'inputs.json')
    config = json.load(f)
    to_find = list(config['to_find'].values()) 
    input_num = list(config['input'].values())
    mat_data = config['layers']
    echoes_removed =  list(config['Echoes'].values())

    '''Material and Geometry of the sample'''
    mu = 12.57e-7
    eps0 = 8.85e-12
    d = [i['thickness'] for i in mat_data]       #list
    eps_data = [i['eps_data'] for i in mat_data] #list 
    is_known = [i['is_known'] for i in mat_data] #list
    is_substrate = [i['is_substrate'] for i in mat_data] #list
    sub_layer = np.where(is_substrate)[0][0]
    
    '''Experimental data'''
    pulse_res = config['pulse_res']  
    n, tmin, tmax, tpos = pulse_res.values()
    pulse_path = input_num[0]
    exp_in_pulse = Path.cwd()/'experimental_data'/pulse_path[0]  # current working directory
    exp_out_pulse = Path.cwd()/'experimental_data'/pulse_path[1] # current working directory
    t_grid, E_air_in, E_sample_out = exp_pulse.fitted_pulse(exp_in_pulse, exp_out_pulse, tmin, tmax, tpos, d, n) # data reading (propagated through air)
    omega, E_air_f = fourier.ft(t_grid, E_air_in)
    E_exp_f = fourier.ft(t_grid, E_sample_out)[1]

    '''Inputing the permittivities'''
    layers = []
    for j, k in zip(eps_data, is_known):
        if k == True:     # if known, find eps
            if type(j[0]) == str: ##
                layers.append(Material(omega).known_nk(j[0], j[1])*eps0)
            else:
                drude = Material(omega).drude(j[0], j[1]);     drude = 1*drude;  drude[0] = drude[1]
                layers.append(drude*eps0)
        else:
            layers.append('unknown')
            if to_find[0] == True:     # permittivity(real and imaginary)
                if type(j[0]) == str:
                    unknown = Material(omega).known_nk(j[0], j[1])
                else:
                    drude = Material(omega).drude(j[0], j[1]);     drude = 1*drude;  drude[0] = drude[1]
                    unknown = drude
            if to_find[1] == True:     # plasma and damping 
                unknown = np.array(j)
            if to_find[2] == True:     # n and k
                unknown = Material(omega).read_nk(j[0], j[1])
    
    
    '''Theoretical calculated transmitted pulse '''
    E_Theory = partial(E_TMM, layers, to_find, omega, eps0, mu, d, E_air_f,sub_layer,echoes_removed)    
    E_theo_t, E_theo_f = E_Theory(unknown)


    '''splitting the material properties into real and imaginary parts'''
    if to_find[0] == True:       # permittivity(real and imaginary)
        new_unknown = np.hstack((np.real(unknown), np.imag(unknown))) 
    if to_find[1] == True:       #drude
        new_unknown = unknown
    if to_find[2] == True:       # n and k
        new_unknown = np.hstack((unknown[0], unknown[1]))

       
    '''Doing the fitting '''
    min_func = partial(Error_func, layers, to_find, omega, eps0, mu, d, E_air_f, E_exp_f,sub_layer,echoes_removed)
        
    if to_find[1] == True:
        bounds = Bounds(unknown*0.4, unknown*0.4)
    else:
        bounds = None
    start = time.time()
    res = minimize(min_func, new_unknown, method='Powell', bounds=bounds, options= {'disp' : True, 'adaptive': True, 'maxiter': 10000000, 'maxfev': 10000000})

    end = time.time()
    

    


    print(f'Elapsed time: {end - start}s')
    print(f'Before: error function =  {min_func(new_unknown)}')
    if to_find[0] == True:    
        result = res['x'][:len(new_unknown)//2] + 1j*res['x'][len(new_unknown)//2:]
        print(f'After: {min_func(np.hstack((np.real(result), np.imag(result))))}')
              
    if to_find[1] == True:
        result = res['x']
        print(f'After: error function =  {min_func(result)}')        
        ##==============yplotting permittivity ============         
        drudePt = Material(omega).drude(5.145, 69.2e-3)
        drudePt_fit = Material(omega).drude(result[0], result[1])
        plt.figure('Plasma_damping')        
        plt.plot(omega,drudePt.real,label = 'input real drude')
        plt.plot(omega,drudePt.imag,label = 'input imag drude')        
        plt.plot(omega,drudePt_fit.real,label = 'output real drude')
        plt.plot(omega,drudePt_fit.imag,label = 'output imag drude')
        plt.legend()
    if to_find[2] == True:
        result = np.array(np.array_split(res['x'], 2))
        
        # Set the window size for the moving average filter (adjust as needed)
        window_size = 9
        
        # Set a threshold for peak prominence (adjust as needed)
        prominence_threshold = 6.0
        
        # Find prominent peaks using the threshold
        peaks, _ = find_peaks(result[0], prominence=prominence_threshold)
        
        # Remove the prominent peaks
        result[0][peaks] = 2
        
        # Apply moving average smoothing to the modified data
        smoothed_data = moving_average(result[0], window_size)
        
        # Generate indices for the smoothed data
        indices = np.arange(len(smoothed_data))
        
        # Create an interpolation function (cubic spline interpolation)
        interp_func = interp1d(indices, smoothed_data, kind='cubic')
        
        # Create a finer grid of indices for the interpolated data
        # fine_indices = np.linspace(0, len(smoothed_data) - 1, 10 * (len(smoothed_data) - 1))
        fine_indices = np.linspace(0, omega[-1],500)
        
        # Interpolate the smoothed data on the finer grid
        interpolated_data = interp_func(fine_indices)

        print(f'After: {min_func(np.hstack((result[0], result[1])))}')
        ##============== plotting n and k============ 
        n_k = Material(omega).read_nk("SiO2.txt", "eV")      
        plt.figure('n_k')     
        plt.plot(omega,n_k[0],label = 'input n')
        plt.plot(omega,n_k[1],label = 'input k')     
        plt.plot(omega,result[0].real,label = 'output n')
        plt.plot(fine_indices,smoothed_data.real,label = 'output n smoothed')
        
        # plt.plot(omega,result[1].real,label = 'output k')
        plt.legend()
    print(f'Result: {result}')
    
    E_theo_fit_t, E_theo_fit_f = E_Theory(result)

    '''Minimization plot - time'''
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(8, 4))
    fig.suptitle('Minimized - time domain')
    ax1.set_title('Before')
    ax2.set_title('After')
    ax1.plot(t_grid, E_air_in, alpha=0.4,label = 'E_air_in')
    ax1.plot(t_grid, E_sample_out,label = 'E_sample_out')
    ax1.plot(t_grid, E_theo_t,label = 'E_theory_time')
    ax1.legend()
    ax2.plot(t_grid, E_air_in, alpha=0.4,label = 'E_air_in')
    ax2.plot(t_grid, E_sample_out, label='E_sample_out')
    ax2.plot(t_grid, E_theo_fit_t, label='Best fit E')
    ax2.legend()
    plt.show()

    '''Minimization plot - freq'''
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(8, 4))
    fig.suptitle('Minimized - freq')
    ax1.set_title('Before')
    ax2.set_title('After')
    ax1.plot(omega, np.abs(E_exp_f))
    ax1.plot(omega, np.abs(E_theo_f))
    ax2.plot(omega, np.abs(E_exp_f), label='exp')
    ax2.plot(omega, np.abs(E_theo_fit_f), label='sim')
    ax2.legend()
    plt.show()

   





