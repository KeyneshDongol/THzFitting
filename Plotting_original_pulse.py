#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:43:18 2023

@author: yingshuyang
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from Functions import exp_pulse

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



pulse_res = config['pulse_res']  
n, tmin, tmax, tpos = pulse_res.values()
pulse_path = input_num[0]
exp_in_pulse = Path.cwd()/'experimental_data'/pulse_path[0]  # current working directory
exp_out_pulse = Path.cwd()/'experimental_data'/pulse_path[1] # current working directory
t_grid, E_air_in= exp_pulse.read_pulse(exp_in_pulse) 
_, E_sample_out= exp_pulse.read_pulse(exp_out_pulse) 
t_grid_new, E_air_in_new, E_sample_out_new = exp_pulse.fitted_pulse(exp_in_pulse, exp_out_pulse, tmin, tmax, tpos, d, n) # data reading (propagated through air)


plt.figure('test')
plt.plot(t_grid,E_air_in)
plt.plot(t_grid,E_sample_out)

plt.figure('test new')
plt.plot(t_grid_new,E_air_in_new)
plt.plot(t_grid_new,E_sample_out_new)




# import numpy as np
# from pathlib import Path


# def save_arrays_to_file(freq, n, k, path):
#     # Stack the arrays horizontally
#     data = np.column_stack((freq, n, k,))

#     # Determine the absolute path
#     file_path = Path.cwd() / 'material_data' / path

#     # Ensure the directory exists
#     file_path.parent.mkdir(parents=True, exist_ok=True)

#     # Save the data to a text file
#     np.savetxt(file_path, data, fmt='%d', delimiter='\t', comments='')


# # Sample data arrays
# array1 = np.array([1, 2, 3, 4])
# array2 = np.array([5, 6, 7, 8])
# array3 = np.array([9, 10, 11, 12])

# save_arrays_to_file(array1, array2, array3, 'fitted_data.txt')

# import numpy as np

# def find_pulse_start(filename, threshold=0.1):
#     # Load data from the file
#     time, amplitude = np.loadtxt(filename, unpack=True)

#     # Compute the rate of change of the amplitude
#     derivative = np.diff(amplitude)

#     # Find the first point where the derivative exceeds the threshold
#     start_index = np.argmax(derivative > threshold)
    
#     # Return the corresponding time value
#     return time[start_index]

# # filename = "path_to_your_file.txt"
# filename = Path.cwd()/'experimental_data'/pulse_path[0] 
# pulse_start_time = find_pulse_start(filename)
# print(f"The pulse starts at {pulse_start_time} ps")