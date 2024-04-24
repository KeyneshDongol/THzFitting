import numpy as np
from Functions.Read_material import Material

class MaterialData:
    def __init__(self, omega, eps0=8.85e-12):
        self.omega = omega
        self.eps0 = eps0

    def known_nk(self, n, k):
        # Assuming n and k are the real and imaginary parts of the refractive index
        epsilon = (n + 1j * k) ** 2
        return epsilon

    def drude(self, plasma_freq, damping_freq):
        print(f"Processing Drude model with plasma_freq: {plasma_freq}, damping_freq: {damping_freq}")
        try:
            eV2Hz = 241.79893e12

            plas = plasma_freq * eV2Hz * 2 * np.pi
            return 1 - (plas ** 2) / (self.omega ** 2 + 1j * self.omega * damping_freq)
        except TypeError as e:
            print(f"Data type error in drude calculation: {e}")
            raise

    def read_nk(self, n, k):
        # Directly read n and k values to form a complex permittivity
        return n + 1j * k

class LayerStack:
    def __init__(self, omega, eps_data_list, is_known_list, to_find):
        self.layers = []
        self.material = Material(omega)
        for eps_data, is_known in zip(eps_data_list, is_known_list):
            layer = self.process_layer(eps_data, is_known, to_find)
            self.layers.append(layer)
            
    def handle_string_input(self, identifier1, identifier2):
    # Example: Read permittivity from a file named `identifier1`
    # where `identifier1` might include additional details like temperature or frequency
        try:
            with open(identifier1, 'r') as file:
                n, k = map(float, file.readline().split())
            return self.material.known_nk(n, k)
        except IOError:
            # Handle the error if file does not exist or other IO issues
            return "File not found or error reading file"
        except ValueError:
            # Handle the error if the data in the file isn't in the expected format
            return "Data format error in file"

    def process_layer(self, eps_data, is_known, to_find):
        if is_known:
            if isinstance(eps_data[0], str):
                # Assuming the strings are filenames or keys to look up permittivity
                # Here, you'd implement or call a method to handle these strings
                return self.handle_string_input(eps_data[0], eps_data[1])
            else:
                # Handling numerical inputs directly if ever present
                return self.material.drude(eps_data[0], eps_data[1])
        else:
            # Process based on what needs to be found
            unknown = 'unknown'
            if to_find.permittivity:
                return self.handle_string_input(eps_data[0], eps_data[1])
            if to_find.plasma_damping:
                return np.array(eps_data)
            if to_find.n_k:
                return self.handle_string_input(eps_data[0], eps_data[1])
            return unknown
    

    def save_layers(self, filename='unknown_initial.txt'):
        with open(filename, 'w') as f:
            for layer in self.layers:
                f.write(str(layer) + '\n')

