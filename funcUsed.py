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
from setup_attributes import Attributes

if __name__ == '__main__':
    # input = 'inputs.
    f = open(Path.cwd()/'inputs.json')  #Text_IO_Wrapper


    attributes = Attributes(f)

    # print(attributes.layers)
            