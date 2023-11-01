import numpy as np
from pathlib import Path


def save_eps(path, omega, eps):
    eps_real = np.real(eps)
    eps_imag = np.imag(eps)
    eps_data = Path.cwd()/'material_data'/path
    with open(eps_data, 'w') as f:
        for freq, real, imag in zip(omega, eps_real, eps_imag):
            f.write(str(freq)+' '+str(real)+' '+str(imag)+'\n')

# def save_nk(path, omega, n,k):
#     n_and_k = Path.cwd()/'material_data'/path
#     with open(n_and_k, 'w') as f:
#         for freq, real, imag in zip(omega, n, k):
#             f.write(str(freq)+' '+str(n)+' '+str(k)+'\n')
            
            
            
def save_nk(path,omega, n, k):
    # Stack the arrays horizontally
    data = np.column_stack((omega, n, k,))

    # Determine the absolute path
    file_path = Path.cwd() / 'material_data' / path

    # Ensure the directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the data to a text file
    np.savetxt(file_path, data)

