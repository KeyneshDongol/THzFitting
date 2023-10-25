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


plt.figure('test')
plt.plot(t_grid,E_air_in)
plt.plot(t_grid,E_sample_out)

