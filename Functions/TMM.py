from typing_extensions import Any
import numpy as np
from numpy._typing import _UnknownType
from numpy.typing import NDArray


# class SpecialMatrix:
#     def __init__(self, omega, mu, epsilon, z):
#         self.mu = mu
#         self.omega = omega
#         self.epsilon = epsilon
#         self.z = z

#     def a_matrix(self, material, position):
#         a11 = np.exp(
#             1j*self.omega*np.sqrt(self.epsilon[material]*self.mu)*self.z[position])
#         a12 = np.exp(-1j*self.omega *
#                       np.sqrt(self.epsilon[material]*self.mu)*self.z[position])
#         a21 = -np.sqrt(self.epsilon[material]/self.mu)*np.exp(
#             1j*self.omega*np.sqrt(self.epsilon[material]*self.mu)*self.z[position])
#         a22 = np.sqrt(self.epsilon[material]/self.mu)*np.exp(-1j*self.omega *
#                                                               np.sqrt(self.epsilon[material]*self.mu)*self.z[position])
#         am = np.array([[a11, a12], [a21, a22]]).transpose(2, 0, 1)
#         return am

#     def inv_a_matrix(self, material, position):
#         a = self.a_matrix(material, position)
#         a0 = np.linalg.inv(a)
#         return a0

#     def m_matrix(self, n):  # normal
#         m11 = np.cos(self.omega*np.sqrt(self.epsilon[n]*self.mu)*self.z[n])
#         m12 = (1j/np.sqrt(self.epsilon[n]/self.mu)) * \
#             np.sin(-self.omega*np.sqrt(self.epsilon[n]*self.mu)*self.z[n])
#         m21 = 1j*np.sqrt(self.epsilon[n]/self.mu)*np.sin(-self.omega *
#                                                           np.sqrt(self.epsilon[n]*self.mu)*self.z[n])
#         m22 = np.cos(self.omega*np.sqrt(self.epsilon[n]*self.mu)*self.z[n])
#         m = np.array([[m11, m12], [m21, m22]]).transpose(2, 0, 1)
#         return m

#     def transfer_matrix(self):
#         t_0inf = self.a_matrix(0, 0)  # a0_0
#         t_sinf = self.a_matrix(1, 1)  # as_d
#         for i in range(1, len(self.z)):
#             t_0inf = np.matmul(self.m_matrix(i), t_0inf)
#         for i in range(2, len(self.z)):
#             t_sinf = np.matmul(self.m_matrix(i), t_sinf)
#         T_0inf = np.matmul(self.inv_a_matrix(0, 0), t_0inf)
#         T_sinf = np.matmul(self.inv_a_matrix(0, 0), t_sinf)
#         T_0s = np.matmul(self.inv_a_matrix(1, 0), self.a_matrix(0, 0))
#         return T_0s, T_sinf, T_0inf

#     def transfer_matrix_special_sinf(self,excitedlayer):
#         ainf_0 = self.inv_a_matrix(0,0)
#         a_excite_d = self.a_matrix(excitedlayer,excitedlayer)
#         special_transfermatrix = a_excite_d
#         for i in range(excitedlayer+1,len(self.z)):
#             special_transfermatrix = np.matmul(self.m_matrix(i),special_transfermatrix)
#         special_transfermatrix = np.matmul(ainf_0,special_transfermatrix)
#         T_exciteinf = special_transfermatrix
#         return T_exciteinf

#     def transfer_matrix_special_0s(self,excitedlayer):
#         a0_0       = self.a_matrix(0,0)
#         a_excite_0 = self.inv_a_matrix(excitedlayer,0)
#         special_transfermatrix = a0_0
#         for i in range(1,excitedlayer):
#             special_transfermatrix = np.matmul(self.m_matrix(i),special_transfermatrix)
#         special_transfermatrix = np.matmul(a_excite_0,special_transfermatrix)
#         T_0excite = special_transfermatrix
#         return T_0excite

#     def transmission_coeff(self, t):
#         trans = (t[0:, 0, 0]*t[0:, 1, 1]-t[0:, 0, 1]*t[0:, 1, 0])/t[0:, 1, 1]
#         return trans

#     def reflection_coeff(self, t):
#         r = -t[0:, 1, 0]/t[0:, 1, 1]
#         return r



class SpecialMatrix(object):
    def __init__(self,omega,mu,epsilon,z):
        self.mu = mu
        self.omega = omega
        self.epsilon = epsilon
        self.z = z


    def a_Matrix(self,material,position):
        a11 = np.exp(1j*self.omega*np.sqrt(self.epsilon[material]*self.mu)*self.z[position])
        a12 = np.exp(-1j*self.omega*np.sqrt(self.epsilon[material]*self.mu)*self.z[position])
        a21 =-np.sqrt(self.epsilon[material]/self.mu)*np.exp(1j*self.omega*np.sqrt(self.epsilon[material]*self.mu)*self.z[position])
        a22 =np.sqrt(self.epsilon[material]/self.mu)*np.exp(-1j*self.omega*np.sqrt(self.epsilon[material]*self.mu)*self.z[position])
        am = np.array([[a11, a12],[a21, a22]]).transpose(2,0,1)
        return am


     #Compute the multiplicative inverse of a matrix
    def inv_a_Matrix(self,material,position):
        a = self.a_Matrix(material,position)
        a0 = np.linalg.inv(a)
        return a0


    def M_Matrix(self,n: Any) -> NDArray[Any]:    #normal
        m11 = np.cos(self.omega*np.sqrt(self.epsilon[n]*self.mu)*self.z[n])
        m12 = (1j/np.sqrt(self.epsilon[n]/self.mu))*np.sin(-self.omega*np.sqrt(self.epsilon[n]*self.mu)*self.z[n])
        m21 = 1j*np.sqrt(self.epsilon[n]/self.mu)*np.sin(-self.omega*np.sqrt(self.epsilon[n]*self.mu)*self.z[n])
        m22 = np.cos(self.omega*np.sqrt(self.epsilon[n]*self.mu)*self.z[n])
        m = np.array([[m11, m12],[m21, m22]]).transpose(2,0,1)
        return m





    def Transfer_Matrix(self) -> tuple[_UnknownType, _UnknownType, _UnknownType]:
        '''
        Returns:
            T_0s, T_sinF, T_0inf
        '''

        t_0inf = self.a_Matrix(0,0)  #a0_0
        t_sinf = self.a_Matrix(1,1)  #as_d
        for i in range(1,len(self.z)):
            t_0inf = np.matmul(self.M_Matrix(i),t_0inf)
        for i in range(2,len(self.z)):
            t_sinf = np.matmul(self.M_Matrix(i),t_sinf)
        T_0inf = np.matmul(self.inv_a_Matrix(0,0),t_0inf)
        T_sinf = np.matmul(self.inv_a_Matrix(0,0),t_sinf)
        T_0s   = np.matmul(self.inv_a_Matrix(1,0),self.a_Matrix(0,0))
        return T_0s,T_sinf,T_0inf



    def Transfer_Matrix_special_Ninf(self,excitedlayer):
        ainf_0 = self.inv_a_Matrix(0,0)
        a_excite_d = self.a_Matrix(excitedlayer,excitedlayer)
        special_transfermatrix = a_excite_d
        for i in range(excitedlayer+1,len(self.z)):
            special_transfermatrix = np.matmul(self.M_Matrix(i),special_transfermatrix)
        special_transfermatrix = np.matmul(ainf_0,special_transfermatrix)
        T_exciteinf = special_transfermatrix
        return T_exciteinf

    def Transfer_Matrix_special_0N(self,excitedlayer):
        a0_0       = self.a_Matrix(0,0)
        a_excite_0 = self.inv_a_Matrix(excitedlayer,0)
        special_transfermatrix = a0_0
        for i in range(1,excitedlayer):
            special_transfermatrix = np.matmul(self.M_Matrix(i),special_transfermatrix)
        special_transfermatrix = np.matmul(a_excite_0,special_transfermatrix)
        T_0excite = special_transfermatrix
        return T_0excite


    def f_x(self,t,f_R,f_L):
        f_xR = t[0:,0,0]*f_R+t[0:,0,1]*f_L
        f_xL = t[0:,1,0]*f_R+t[0:,1,1]*f_L
        return f_xR,f_xL



    def Transmission_coeff(self,t):
        trans = (t[0:,0,0]*t[0:,1,1]-t[0:,0,1]*t[0:,1,0])/t[0:,1,1]
        return trans



    def Reflection_coeff(self,t):
        r=-t[0:,1,0]/t[0:,1,1]
        return r


    def Jcoeff(self,t):
        Jcoeff = t[0:,0,1]/t[0:,1,1]
        return Jcoeff



    def J_matrix(self,f_k,f_omegaj):
        f_k = f_k.reshape(-1,1)
        J_ok = f_omegaj*f_k   #data matrix of the current
        return J_ok

    @staticmethod
    def E_TMM_new(layers: list, to_find: list[bool], omega:object, eps0: float, mu: float, d: list[object], f_in: NDArray, sub_layer: object, echoes_removed: list[object]) -> tuple[NDArray[np.floating[Any]], Any]:
        epsilon = layers

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