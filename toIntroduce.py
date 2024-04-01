from typing_extensions import Any
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from numpy._typing import _UnknownType
from numpy.typing import NDArray
from Functions.TMM import SpecialMatrix
from Functions import exp_pulse, fourier
from matplotlib import cm


if __name__ == '__main__':
    def E_TMM (layers: list, toFind: list[bool], omega: object, exp: float, _mu: float, _d: list[object], fourierIn: NDArray, subLayer: object, echoesRemoved: list[object]):# -> tuple[NDArray[np.floating[Any]], Any]:
        epsilon = layers

        # Transfer Matrix
        TMM = SpecialMatrix(omega, _mu, epsilon, _d)
        foo_,foo_,T_0inf = TMM.Transfer_Matrix()
        T_0s = TMM.Transfer_Matrix_special_0N(subLayer)
        T_sinFourier = TMM.Transfer_Matrix_special_Ninf(subLayer)

        # Transmission & Reflection Coefficients
        t_0s = TMM.Transmission_coeff(T_0s)
        t_sinFourier = TMM.Transmission_coeff(T_sinFourier)
        t = TMM.Transmission_coeff(T_0inf)
        t_noEcho = np.multiply(t_0s, t_sinFourier)

        # Remove Echo
        if echoesRemoved[0] == True:
            f_inFourierR = fourierIn * t_noEcho
        else:
            f_inFourierR = fourierIn * t

        # Transmitted wave in k-space
        transmitted = fourier.ift(f_inFourierR)
        return transmitted, f_inFourierR

    '''Inputs'''
    input = open(Path.cwd()/'inputs.json')
    config = json.load(input)
    toFind = list(config['to_find'].values())
    inputNum = list(config['input'].values())
    matData = config['layers']
    echoesRemoved = list(config['Echoes'].values())

    '''Material and Geometry of the sample'''
    _mu = 12.57e-7
    _eps = 8.85e-12
    _d = [i['thickness'] for i in matData]
    expData = [i['eps_data'] for i in matData]
    isKnown = [i['is_known'] for i in matData]
    isSubstrate = [i['is_substrate'] for i in matData]
    subLayer = np.where(isSubstrate)[0][0]

    '''Experimental Data'''
    pulseRes = config['pulse_res']
    _n, tMin, tMax, tPos = pulseRes.values()
    pulsePath = inputNum[0]

    expInPulse = Path.cwd()/'experimental_data'/pulsePath[0]
    expOutPulse = Path.cwd()/'experimental_data'/pulsePath[1]
    tGridOriginal, E_AirInOriginal = exp_pulse.read_pulse(expInPulse) #reading the original pulse E_ref
    tGridVariable,E_SampleOutOriginal = exp_pulse.read_pulse(expOutPulse)

    tGrid, E_AirIn, E_SampleOut = exp_pulse.fitted_pulse(expInPulse, expOutPulse, tMin, tMax, tPos, _d, _n) #data reading (propagated through air)
    _omega, E_AirFourier = fourier.ft(tGrid, E_AirIn)
    E_expFourier = fourier.ft(tGrid, E_SampleOut)[1]

    _omega, E_AirExpFourier = fourier.ft(tGrid, E_AirIn)
    tGridVariable, E_SampleExpFourier = fourier.ft(tGrid, E_SampleOut)


    '''Transfer Function Measured'''
    TransferFuncExp = E_SampleExpFourier/E_AirFourier

    '''Transfer Function Theory'''
    epsAir = np.array([1] * len(_omega))

    layers1 = [epsAir * _eps, epsAir * _eps, epsAir * _eps, epsAir * _eps, epsAir * _eps, epsAir * _eps]
    E_AirTheoryTransmistted, E_AirTheoryFourier = E_TMM(layers1, toFind, _omega, _eps, _mu, _d, E_AirFourier, subLayer, echoesRemoved)


    plt.figure('test')
    plt.plot(tGrid, E_AirIn)
    plt.plot(tGrid, E_AirTheoryTransmistted)
    # plt.show

    def calculateError(n: int, k: int) -> int:
        '''
        Return:
            Error
        '''
        epsAir = np.array([1] * len(_omega))
        nIndex = np.array([n] * len(_omega))
        # print("nIndex ->", nIndex)
        kIndex = np.array([k] * len(_omega))
        # print("kIndex ->", kIndex)
        epsS = (nIndex + 1j*kIndex )**2
        layers = [epsAir * _eps, epsS * _eps, epsAir * _eps, epsAir * _eps, epsAir * _eps, epsAir * _eps]

        # Replace with actual E_TMM function call
        E_SampleTheoryTransmitted, E_SampleTheoryFourier = E_TMM(layers, toFind, _omega, _eps, _mu, _d, E_AirFourier, subLayer, echoesRemoved)

        TransmittedExperimentalFourier = E_SampleExpFourier/E_AirExpFourier
        TransmittedTheoreticalFourier = E_SampleTheoryFourier/E_SampleTheoryFourier

        return np.sum(np.abs(E_SampleTheoryFourier - E_SampleExpFourier)**2)

    # Set up the ranges for n and k
    nValues = np.linspace(1, 6, 50)  #  steps from 0 to 6 for refractive index
    kValues = np.linspace(0, 0.05, 50)  # steps from 0 to 1 for extinction coefficient

    # Prepare a grid for error values
    errorValues = np.zeros((len(nValues), len(kValues)))

    # Calculate the error function for each combination of n and k
    for i, n in enumerate(nValues):
        for j, k in enumerate(kValues):
            errorValues[i, j] = calculateError(n, k)

    # test = errorValues.argmin()
    # minIndex = np.where(errorValues == np.min(errorValues))
    # minIndex = divmod(errorValues.argmin(), errorValues.shape[1])
    print("The index is: ")
    minIndex = np.unravel_index(errorValues.argmin(), errorValues.shape)

    print(minIndex)
    min_nValue = nValues[minIndex[0]]
    min_kValue = kValues[minIndex[1]]
    print(min_nValue)
    print(min_kValue)

    def getMin(errorValues: NDArray[np.float64]) -> tuple[int, int]:
        minIndex = np.unravel_index(errorValues.argmin(), errorValues.shape)
        min_nValue = nValues[minIndex[0]]
        min_kValue = kValues[minIndex[1]]

        return min_nValue, min_kValue

    test = getMin(errorValues)
    print(test)

    # # Create a figure for the plots
    # fig = plt.figure(figsize=(18, 8))

    # # 3D plot of error function - larger subplot
    # ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2, projection='3d')
    # N, K = np.meshgrid(nValues, kValues)
    # surf = ax1.plot_surface(N, K, errorValues.T, cmap=cm.get_cmap("Spectral"))
    # ax1.set_xlabel('Refractive Index (n)')
    # ax1.set_ylabel('Extinction Coefficient (k)')
    # ax1.set_zlabel('Error Function')
    # ax1.set_title('3D Error Function Surface Plot')
    # plt.colorbar(surf, ax=ax1, shrink=0.5)

    # # Common colormap for the 2D plots
    # cmap = cm.get_cmap("Spectral")

    # # 2D Projection along n-axis - smaller subplot
    # ax2 = plt.subplot2grid((2, 3), (0, 2))
    # for k_index, k in enumerate(kValues):
    #     color = cmap(k_index / len(kValues))
    #     ax2.plot(nValues, errorValues[:, k_index], color=color)
    # ax2.set_xlabel('Refractive Index (n)')
    # ax2.set_ylabel('Error Function')
    # ax2.set_title('Error Projection Along n-axis')

    # # 2D Projection along k-axis - smaller subplot
    # ax3 = plt.subplot2grid((2, 3), (1, 2))
    # for n_index, n in enumerate(nValues):
    #     color = cmap(n_index / len(nValues))
    #     ax3.plot(kValues, errorValues[n_index, :], color=color)
    # ax3.set_xlabel('Extinction Coefficient (k)')
    # ax3.set_ylabel('Error Function')
    # ax3.set_title('Error Projection Along k-axis')

    # plt.tight_layout()
    # plt.show()
