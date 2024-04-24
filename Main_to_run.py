import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial
import json
import time
from math import pi
from matplotlib.gridspec import GridSpec
from Error_function import E_TMM_new_, calculate_error
from Functions.Read_material import Material
from Functions import exp_pulse, fourier, tools
from scipy.optimize import minimize
from Functions.mainfunction import E_TMM, Error_func,  noise_remove
# from toIntroduce import E_AirInOriginal
# from Plotting_original_pulse import echoes_removed

if __name__ == '__main__':

    '''inputs'''
    f = open(Path.cwd()/'inputs.json')  #Text_IO_Wrapper
    config = json.load(f) #{dict:5}
    to_find = list(config['to_find'].values()) #{list: 3} ->  "permittivity", "plasma_damping", "n_k"
    input_num = list(config['input'].values()) #{list: 1} -> "pulse"
    mat_data = config['layers'] #{list: 6} -> material data
    echoes_removed =  config['Echoes'].values()

    '''Material and Geometry of the sample'''
    mu = 12.57e-7
    eps0 = 8.85e-12
    d = [i['thickness'] for i in mat_data]       #list of thicknesses
    eps_data = [i['eps_data'] for i in mat_data] #TODO: Shift below to inputing the permittivities
    is_known = [i['is_known'] for i in mat_data] #list of bool
    is_substrate = [i['is_substrate'] for i in mat_data] #list
    sub_layer = np.where(is_substrate)[0][0]

    '''Experimental data'''
    pulse_res = config['pulse_res']  #{dict: 4} -> pulse_res: {'n', 'tmax', 'tmin', 'tpos'}
    n, tmin, tmax, tpos = pulse_res.values()
    pulse_path = input_num[0]

    exp_in_pulse = Path.cwd()/'experimental_data'/pulse_path[0]  # current working directory {PosixPath}
    exp_out_pulse = Path.cwd()/'experimental_data'/pulse_path[1] # current working directory
    t_grid, E_air_in, E_sample_out = exp_pulse.fitted_pulse(exp_in_pulse, exp_out_pulse, tmin, tmax, tpos, d, n) # data reading (propagated through air)
    omega, E_air_f = fourier.ft(t_grid, E_air_in)
    E_exp_f = fourier.ft(t_grid, E_sample_out)[1]

    tGridOriginal, E_AirInOriginal = exp_pulse.read_pulse(exp_in_pulse)
    tGridVariable, E_SampleOutOriginal = exp_pulse.read_pulse(exp_out_pulse)

    omega, E_air_exp_f = fourier.ft(t_grid, E_air_in)
    t_grid_variable, E_SampleExpFourier = fourier.ft(t_grid, E_sample_out)


    TransferFuncExp = E_SampleExpFourier/E_air_f

    epsAir = np.array([1] * len(omega))
    layers1 = [epsAir * eps0, epsAir * eps0, epsAir * eps0, epsAir * eps0, epsAir * eps0, epsAir * eps0]
    E_air_theo_t, E_air_theo_f = E_TMM_new_(layers1, to_find, omega, eps0, mu, d, E_air_f, sub_layer, echoes_removed)

    #set up the rangesfor n and k
    nValues = np.linspace(2, 6, 50)  #  TODO: Introduce a way to keep the 2-6 variable dynamic.
    kValues = np.linspace(0, 0.05, 50)  # TODO: Same for kValues x-y axis values

    errorValues = np.zeros((len(nValues), len(kValues)))

    for i, n in enumerate(nValues):
        for j, k in enumerate(kValues):
            errorValues[i,j] = calculate_error(n,k)

    # print(errorValues)
    
    '''
    Error funciton values will replace eps_data in inputs.json file. 
    '''


    '''Inputing the permittivities'''
    layers = []
    # zip joins 2 functions together
    for j, k in zip(eps_data, is_known): # j = {list: 2}, k = {bool}
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
                
            #TODO: save the above new list titlted layer as unknown_initiL.txt file


    '''Theoretical calculated transmitted pulse '''
    E_Theory = partial(E_TMM, layers, to_find, omega, eps0, mu, d, E_air_f, sub_layer, echoes_removed)
    E_theo_t, E_theo_f = E_Theory(unknown)  # unknown = {ndarray: (2,61)}
    # unknown are values from material, aka they are new_n and new_k


    #TODO: find a way to connect this with the Error func script.

    '''Splitting the material properties into real and imaginary parts'''
    if to_find[0] == True:       # permittivity(real and imaginary)
        new_unknown = np.hstack((np.real(unknown), np.imag(unknown)))
    if to_find[1] == True:       #drude
        new_unknown = unknown
    if to_find[2] == True:       # n and k
        new_unknown = np.hstack((unknown[0], unknown[1]))


    '''Doing the fitting '''
    min_func = partial(Error_func, layers, to_find, omega, eps0, mu, d, E_air_f, E_exp_f, sub_layer, echoes_removed)

    if to_find[1] == True:
        bounds = None #Bounds(unknown*0.65, unknown*0.65)
    else:
        bounds = None
    start = time.time()
    res = minimize(min_func, new_unknown, method='L-BFGS-B', bounds = bounds,  options= {'disp' : True, 'adaptive': True, 'maxiter': 200000, 'maxfev': 200000})

    end = time.time()



    freq = omega*1e-12/2*pi
    t_grid = t_grid*1e12

    print(f'Elapsed time: {end - start}s')
    print(f'Before: error function =  {min_func(new_unknown)}')

    ## Fitting  permittivity
    if to_find[0] == True:
        result = res['x'][:len(new_unknown)//2] + 1j*res['x'][len(new_unknown)//2:]
        per_real = result.real
        per_imag = result.imag
        print(f'After: {min_func(np.hstack((np.real(result), np.imag(result))))}')
        drudePt = Material(omega).drude(5.145, 69.2e-3)
      # Define global style settings
        fontsize = 5
        title_fontsize = 6
        linewidth = 1

        plt.figure('Plasma_damping', figsize=(8.6/2.54, 7/2.54), dpi=200)
        gs = GridSpec(2, 1, height_ratios=[1, 1])  # height ratios for subplots

        # Subplot 1: Real data
        plt.subplot(gs[0])
        plt.plot(freq, drudePt.real, linewidth=linewidth, label='$\epsilon$ real input', color='tab:green')
        plt.plot(freq, per_real, linewidth=linewidth, label='$\epsilon$ real fit', color='brown')
        plt.xlabel('Freq. (THz)', fontsize=fontsize)
        plt.ylabel('$\epsilon$ Real', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.gca().yaxis.get_offset_text().set_size(fontsize)


        # Subplot 2: Imaginary data
        plt.subplot(gs[1])
        plt.plot(freq, drudePt.imag, linewidth=linewidth, label='$\epsilon$ imag input', color='tab:green')
        plt.plot(freq, per_imag, linewidth=linewidth, label='$\epsilon$ imag fit', color='brown')
        plt.xlabel('Freq. (THz)', fontsize=fontsize)
        plt.ylabel('$\epsilon$ Imaginary', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.gca().yaxis.get_offset_text().set_size(fontsize)

        plt.tight_layout()
        plt.show()

    ## Fitting  plasma and damping frequency
    if to_find[1] == True:
        result = res['x']
        print(f'After: error function =  {min_func(result)}')
        ##==============yplotting permittivity ============
        drudePt = Material(omega).drude(5.145,0.069)                         #TODO: Mission1: change a way that auto plot the input vlues
        drudePt_fit = Material(omega).drude(result[0], result[1])
     # Define global style settings
        fontsize = 5
        title_fontsize = 6
        linewidth = 1

        # Create figure and specify grid specs
        plt.figure('Plasma_damping', figsize=(8.6/2.54, 7/2.54), dpi=200)
        gs = GridSpec(3, 1, height_ratios=[1, 1, 0.1])  # height ratios for subplots

        # Subplot 1: Real data
        plt.subplot(gs[0])
        plt.plot(freq, drudePt.real, linewidth=linewidth, label='$\epsilon$ real input', color='tab:green')
        plt.plot(freq, drudePt_fit.real, linewidth=linewidth, label='$\epsilon$ real fit', color='brown')
        plt.xlabel('Freq. (THz)', fontsize=fontsize)
        plt.ylabel('$\epsilon$ Real', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.gca().yaxis.get_offset_text().set_size(fontsize)

        # Subplot 2: Imaginary data
        plt.subplot(gs[1])
        plt.plot(freq, drudePt.imag, linewidth=linewidth, label='$\epsilon$ imag input', color='tab:green')
        plt.plot(freq, drudePt_fit.imag, linewidth=linewidth, label='$\epsilon$ imag fit', color='brown')
        plt.xlabel('Freq. (THz)', fontsize=fontsize)
        plt.ylabel('$\epsilon$ Imaginary', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.gca().yaxis.get_offset_text().set_size(fontsize)

        # Subplot 3: Display result
        plt.subplot(gs[2])
        plt.axis('off')
        result_text1 = f"Plasma frequency =  {result[0]} ev"
        result_text2 = f"Damping frequency =  {result[1]} ev"
        plt.text(0.2, 1.2, result_text1, fontsize=fontsize, ha='center', va='center')
        plt.text(0.2, 0, result_text2, fontsize=fontsize, ha='center', va='center')

        plt.tight_layout()
        plt.show()



        ##==============PLOTTING RESULTS ============

        E_theo_fit_t, E_theo_fit_f = E_Theory(result)

                        # Define global style settings
        fontsize = 5
        title_fontsize = 6
        linewidth = 1
        linewidth2 = 2
        # Section 1: Minimization plot - time
        plt.figure('Minimization Plot - Time', figsize=(8.6/2.54, 5/2.54), dpi=200)

        # Subplot 1
        plt.subplot(121)
        plt.title('Before', fontsize=title_fontsize)
        plt.plot(t_grid, E_air_in, linewidth=linewidth, label='E_air_in', color='lightblue')
        plt.plot(t_grid, E_sample_out, linewidth=linewidth2, label='E_sample_out', color='tab:green')
        plt.plot(t_grid, E_theo_t, linewidth=linewidth, label='E_theory_out', color='brown')
        plt.xlabel('Time (ps)', fontsize=fontsize)
        plt.ylabel('E(t) arb.units', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.tight_layout()
        # Subplot 2
        plt.subplot(122)
        plt.title('After', fontsize=title_fontsize)
        plt.plot(t_grid, E_air_in, linewidth=linewidth, label='E_air_in', color='lightblue')
        plt.plot(t_grid, E_sample_out, linewidth=linewidth2, label='E_sample_out', color='tab:green')
        plt.plot(t_grid, E_theo_fit_t, linewidth=linewidth, label='E_theory_fit', color='brown')
        plt.xlabel('Time (ps)', fontsize=fontsize)
        plt.ylabel('E(t) arb.units', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)


        plt.tight_layout()

        # Section 2: Minimization plot - freq
        plt.figure('Minimization Plot - Freq', figsize=(8.6/2.54, 5/2.54), dpi=200)

        # Subplot 1
        plt.subplot(121)
        plt.title('Before', fontsize=title_fontsize)
        plt.plot(freq, np.abs(E_exp_f), linewidth=linewidth2, label='E_sample_out', color='tab:green')
        plt.plot(freq, np.abs(E_theo_f), linewidth=linewidth, label='E_theory_fit', color='brown')
        plt.xlabel('Freq.(THz)', fontsize=fontsize)
        plt.ylabel('E($\omega$) arb.units', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.xlim(0,50)
        plt.tight_layout()
        # Subplot 2
        plt.subplot(122)
        plt.title('After', fontsize=title_fontsize)
        plt.plot(freq, np.abs(E_exp_f), linewidth=linewidth2, label='E_sample_out', color='tab:green')
        plt.plot(freq, np.abs(E_theo_fit_f), linewidth=linewidth, label='E_theory_out', color='brown')
        plt.xlabel('Freq.(THz)', fontsize=fontsize)
        plt.ylabel('E($\omega$) arb.units', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.xlim(0,50)
        plt.tight_layout()



    ## Fitting  n and k
    if to_find[2] == True:

        ##============== noise removing (pure data smoothing)============
        result = np.array(np.array_split(res['x'], 2))     # original raw data
        fine_x,smoothed_data1,smoothed_data2,interpolated_data1,interpolated_data2,results = noise_remove(res, omega, window_size=21, prominence_threshold1=5,prominence_threshold2=0.5)    #TODO: Do something about this
        freq_new = fine_x*1e-12/2*3.1415926   #THz
        inter_new = np.array([smoothed_data1,smoothed_data2])
        new_results = tools.save_nk('fitted_data_quartz_B.txt', freq,smoothed_data1,smoothed_data2)
        # new_results = tools.save_nk('fitted_data_test.txt', freq,result[0],result[1])

        print(f'After: {min_func(np.hstack((result[0], result[1])))}')

        ##============== plotting n and k============
        for known, data in zip(is_known, eps_data):
            if not known:
                filename, units = data
                n_k = Material(omega).read_nk(filename, units)



        # Define global style settings
        fontsize = 5
        title_fontsize = 6
        linewidth1 = 1
        linewidth2 = 2

        # Plot n_k figure
        plt.figure('n_k figure', figsize=(8.6/2.54, 7/2.54), dpi=200)

        # Subplot 1
        plt.subplot(211)
        plt.plot(freq, n_k[0], linewidth=linewidth1, label='input n', color='lightblue')
        plt.plot(freq, result[0].real, linewidth=linewidth1, label='output n', color='lightgreen')
        plt.plot(freq, results[0].real, linewidth=linewidth1, label='output n dropped', color='bisque')
        plt.plot(freq, smoothed_data1, linewidth=linewidth2, label='smoothed', color='orange')
        plt.plot(freq_new, interpolated_data1, linewidth=linewidth1, label='interpolated', color='purple')
        plt.xlabel('Freq. (THz)', fontsize=fontsize)
        plt.ylabel('n', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)


        # Subplot 2
        plt.subplot(212)
        plt.plot(freq, n_k[1], linewidth=linewidth1, label='input k', color='lightblue')
        plt.plot(freq, result[1].real, linewidth=linewidth1, label='output k', color='lightgreen')
        plt.plot(freq, results[1].real, linewidth=linewidth1, label='output k dropped', color='bisque')
        plt.plot(freq, smoothed_data2, linewidth=linewidth2, label='smoothed', color='orange')
        plt.plot(freq_new, interpolated_data2, linewidth=linewidth1, label='interpolated', color='purple')
        plt.xlabel('Freq.(THz)', fontsize=fontsize)
        plt.ylabel('k', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.show()

        plt.tight_layout()

        ##==============PLOTTING RESULTS ============

        E_theo_fit_t, E_theo_fit_f = E_Theory(result)


                # Define global style settings
        fontsize = 5
        title_fontsize = 6
        linewidth = 1
        linewidth2 = 2
        # Section 1: Minimization plot - time
        plt.figure('Minimization Plot - Time', figsize=(8.6/2.54, 5/2.54), dpi=200)

        # Subplot 1
        plt.subplot(121)
        plt.title('Before', fontsize=title_fontsize)
        plt.plot(t_grid, E_air_in, linewidth=linewidth, label='E_air_in', color='lightblue')
        plt.plot(t_grid, E_sample_out, linewidth=linewidth2, label='E_sample_out', color='tab:green')
        plt.plot(t_grid, E_theo_t, linewidth=linewidth, label='E_theory_out', color='brown')
        plt.xlabel('Time (ps)', fontsize=fontsize)
        plt.ylabel('E(t) arb.units', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.tight_layout()
        # Subplot 2
        plt.subplot(122)
        plt.title('After', fontsize=title_fontsize)
        plt.plot(t_grid, E_air_in, linewidth=linewidth, label='E_air_in', color='lightblue')
        plt.plot(t_grid, E_sample_out, linewidth=linewidth2, label='E_sample_out', color='tab:green')
        plt.plot(t_grid, E_theo_fit_t, linewidth=linewidth, label='E_theory_fit', color='brown')
        plt.xlabel('Time (ps)', fontsize=fontsize)
        plt.ylabel('E(t) arb.units', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.tight_layout()



        # Section 2: Minimization plot - freq
        plt.figure('Minimization Plot - Freq', figsize=(8.6/2.54, 5/2.54), dpi=200)

        # Subplot 1
        plt.subplot(121)
        plt.title('Before', fontsize=title_fontsize)
        plt.plot(freq, np.abs(E_exp_f), linewidth=linewidth2, label='E_sample_out', color='tab:green')
        plt.plot(freq, np.abs(E_theo_f), linewidth=linewidth, label='E_theory_fit', color='brown')
        plt.xlabel('Freq.(THz)', fontsize=fontsize)
        plt.ylabel('E($\omega$) arb.units', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.xlim(0,50)
        plt.tight_layout()
        # Subplot 2
        plt.subplot(122)
        plt.title('After', fontsize=title_fontsize)
        plt.plot(freq, np.abs(E_exp_f), linewidth=linewidth2, label='E_sample_out', color='tab:green')
        plt.plot(freq, np.abs(E_theo_fit_f), linewidth=linewidth, label='E_theory_out', color='brown')
        plt.xlabel('Freq.(THz)', fontsize=fontsize)
        plt.ylabel('E($\omega$) arb.units', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.xlim(0,50)
        plt.tight_layout()

        print(f'Result: {result}')






    # Load the existing JSON data
    with open('inputs.json') as f:
        data = json.load(f)

    # Modify the layer's eps_data only when 'is_known' is False
    for layer in data['layers']:
        if not layer['is_known']:
            # Check if the first element of eps_data is a string
            if isinstance(layer['eps_data'][0], str):
                layer['eps_data'] = ['fitted_data_quartz_B.txt', 'THz']
            # Check if the first element of eps_data is a number (int or float)
            elif isinstance(layer['eps_data'][0], (int, float)):
                layer['eps_data'] = [result[0],result[1]]

    # Save the modified data back to a new JSON file
    with open('results.json', 'w') as f:
        json.dump(data, f, indent=4)
