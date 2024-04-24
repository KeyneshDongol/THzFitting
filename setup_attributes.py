import json
import numpy as np
from pathlib import Path
from Functions import exp_pulse, fourier, tools
from Error_function import E_TMM_new_
from Functions.Read_material import Material
from MaterialLayer import LayerStack



class ToFind:
    def __init__(self, permittivity = False, plasma_damping = False, n_k = False):
        self.permittivity = permittivity
        self.plasma_damping = plasma_damping
        self.n_k = n_k
    def __str__(self):
        return (f"ToFind(permittivity={self.permittivity}, "
                f"plasma_damping={self.plasma_damping}, n_k={self.n_k})")

class Input:
    def __init__(self, pulse = None):
        if pulse is None:
            raise ValueError("Pulse must be provided.")
        self.pulse = pulse
    def __str__(self):
        return f"Input(pulse={self.pulse})"

class PulseRes:
    def __init__(self, n = 0, tmin = 0, tmax = 0, tpos = 0):
        self.n = n
        self.tmin = tmin
        self.tmax = tmax
        self.tpos = tpos
    def __str__(self):
        return (f"PulseRes(n={self.n}, tmin={self.tmin}, tmax={self.tmax}, tpos={self.tpos})")


class Layer:
    def __init__(self, omega, thickness=[], eps_data=None, is_substrate=False, is_known=True):
        if eps_data is None:
            raise ValueError("Error: No eps_data was provided.")
        self.material = Material(omega)  # Initialize a Material object with the provided frequency
        self.thickness = thickness
        self.eps_data = eps_data
        self.is_substrate = is_substrate
        self.is_known = is_known
        self.permittivity = self.calculate_permittivity()

    def calculate_permittivity(self):
        if self.is_known:
            if isinstance(self.eps_data[0], str):  # Assuming path and unit are provided
                return self.material.known_nk(self.eps_data[0], self.eps_data[1])
            else:  # Assuming plasma frequency and damping are provided for Drude model
                return self.material.drude(self.eps_data[0], self.eps_data[1])
        return 'unknown'   

    def __str__(self):
        return (f"Layer(thickness={self.thickness}, eps_data={self.eps_data}, "
                f"is_substrate={self.is_substrate}, is_known={self.is_known}, "
                f"permittivity={self.permittivity})")  # Removed () after self.permittivity

class Echoes:
    def __init__(self, Removed = False):
        self.Removed = Removed
    def __str__(self):
        return f"Echoes(Removed={self.Removed})"

class ExperimentalData:
    def __init__(self, config):
        self.config = config
        self.base_path = Path.cwd() / 'experimental_data'
        self.pulse_res = config.get('pulse_res', {})
        self.n, self.tmin, self.tmax, self.tpos = self.pulse_res.values()
        input_data = config.get('input', {})
        self.pulse_path = [self.base_path / path for path in input_data.get('pulse', [])]
        # self.process_experimental_data()

    def process_experimental_data(self, layer_thicknesses, n=None):
        if len(self.pulse_path) < 2:
            raise ValueError("Insufficient pulse paths provided")

        exp_in_pulse, exp_out_pulse = self.pulse_path
        # Assuming layer_thicknesses is correctly formatted and ready to be used as 'd'
        t_grid, E_air_in, E_sample_out = exp_pulse.fitted_pulse(exp_in_pulse, exp_out_pulse, self.tmin, self.tmax, self.tpos, layer_thicknesses, n or self.n)
        omega, E_air_f = fourier.ft(t_grid, E_air_in)
        _, E_exp_f = fourier.ft(t_grid, E_sample_out)

        self.t_grid = t_grid
        self.E_air_in = E_air_in
        self.E_sample_out = E_sample_out
        self.omega = omega
        self.E_air_f = E_air_f
        self.E_exp_f = E_exp_f
        

class Attributes:
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as file:
            config = json.load(file)
        
        required_sections = ['to_find', 'input', 'pulse_res', 'layers', 'Echoes']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Initialize constants
        self.mu = 12.57e-7
        self.eps0 = 8.85e-12

        # Initialize JSON data
        self.to_find = ToFind(**config.get('to_find', {}))
        self.input = Input(**config.get('input', {}))
        self.pulse_res = PulseRes(**config.get('pulse_res', {}))
        self.echoes = Echoes(**config.get('Echoes', {}))

        # Pass the whole config or just the relevant part to ExperimentalData
        self.experimental_data = ExperimentalData(config)
        
        # Extract layer thicknesses from layers data
        self.layer_thicknesses = [layer['thickness'] for layer in config.get('layers', [])]

        # Initialize layers, note omega is fetched after experimental_data is fully prepared
        self.experimental_data.process_experimental_data(self.layer_thicknesses)
        
        self.omega = self.experimental_data.omega
        if self.experimental_data.omega is None:
            raise ValueError("Omega has not been initialized properly.")
        
        # Now it's safe to initialize layers with self.omega, as it's guaranteed to be set
        self.layers = [Layer(self.omega, **layer) for layer in config.get('layers', [])]

        # Other initializations as needed
        self.layer_thicknesses = [layer.thickness for layer in self.layers]  # Collect all layer thicknesses
        self.process_material_data()
        
        self.layer_stack = LayerStack(self.omega, self.eps_data, self.is_known, self.to_find)



    def process_material_data(self):
        # Process the layers data (e.g., extracting thicknesses and permittivity data)
        self.d = [layer.thickness for layer in self.layers]
        self.eps_data = [layer.eps_data for layer in self.layers]
        self.is_known = [layer.is_known for layer in self.layers]
        self.is_substrate = [layer.is_substrate for layer in self.layers]
        self.sub_layer = np.where(self.is_substrate)[0][0] if any(self.is_substrate) else None


class ErrorFunction:
    def __init__(self, experimental_data, eps0, mu, to_find, sub_layer, echoes_removed, layer_thicknesses):
        self.experimental_data = experimental_data
        self.eps0 = eps0
        self.mu = mu
        self.to_find = to_find
        self.sub_layer = sub_layer
        self.echoes_removed = echoes_removed
        self.d = layer_thicknesses 
        self.initialize_data()

    def initialize_data(self):
        '''
        Read original pulses and perform Fourier Transform
        '''
        self.tGridOriginal, self.E_AirInOriginal = exp_pulse.read_pulse(self.experimental_data.pulse_path[0])
        self.tGridVariable, self.E_SampleOutOriginal = exp_pulse.read_pulse(self.experimental_data.pulse_path[1])
        self.omega, self.E_air_exp_f = fourier.ft(self.tGridOriginal, self.E_AirInOriginal)
        _, self.E_sample_exp_f = fourier.ft(self.tGridVariable, self.E_SampleOutOriginal)
        self.TF_exp = self.E_sample_exp_f / self.E_air_exp_f

    def setup_theoretical_layers(self, n, k):
        eps_Air = np.ones_like(self.omega)
        index_n = np.full_like(self.omega, n)
        index_k = np.full_like(self.omega, k)
        epsS = (index_n + 1j * index_k) ** 2
        return [eps_Air, epsS * self.eps0, eps_Air, eps_Air, eps_Air, eps_Air]

    def calculate_error(self, n, k):
        layers = self.setup_theoretical_layers(n, k)
         
        # Utilize E_TMM_new_ function here
        E_sample_theo_t, E_sample_theo_f = E_TMM_new_(layers, self.to_find, self.omega, self.eps0, self.mu, self.d, self.E_air_exp_f, self.sub_layer, self.echoes_removed)
        TF_theo = E_sample_theo_f / self.E_air_exp_f
        error = np.sum(np.abs(TF_theo - self.TF_exp)**2)
        return error

    def optimize_parameters(self, n_range=(2, 6), k_range=(0, 0.05), steps=50):
        nValues = np.linspace(*n_range, steps)
        kValues = np.linspace(*k_range, steps)
        errorValues = np.zeros((len(nValues), len(kValues)))

        for i, n in enumerate(nValues):
            for j, k in enumerate(kValues):
                errorValues[i, j] = self.calculate_error(n, k)

        return nValues, kValues, errorValues