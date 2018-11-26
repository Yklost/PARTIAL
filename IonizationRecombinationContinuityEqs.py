# -*- coding: utf-8 -*-
"""
Created on Fri May 18 17:22:03 2018

@author: atkachen
"""

from math import sqrt, exp, pi 
import numpy as np
import matplotlib.pyplot as plt


################ input parameters ########################################################

n_t = 1000   #number of timesteps
T = 0.001   #full time interval
n_z = 100    #number of spatial steps


############## calculation of initial profiles of density ################################

h1 = 2.0 * 10**8 #cm
h2 = 1.0 * 10**8 #cm
n_n_0 =  1.0 * 10**13 #cm^-3
n_i_0 = n_n_0 * 0.001 #cm^-3
T_n = 6000.0 #K
T_i = 11600.0 #K
m = 1.6 * 10**(-24) #g 
k_b = 1.3807*10**(-16) #erg/K
m_e = 9.1094*10**(-28) #g
sigma = 2.0*10**(-17) #cm^2
e_ioniz = 2.18*10**(-11) #erg
g = 27400.0 #cm/s^2

x = np.linspace(h1, h2, n_z)
n_n = []
n_i = []

for h in x:
    n_n.append(n_n_0 * exp( - m * g * h / k_b / T_n))
    n_i.append(n_i_0 * exp( - m * g * h / k_b / T_i))

################# calculating initial profiles and ioniz. coef ###########################
            
u_0 = [[0.0 for u_01 in range(2)] for u_02 in range(n_z)]
tem_n_0 = []
tem_i_0 = []
coef_0 = []
pre_calc = []
ioniz_frac_0 = []
 
for w in range(n_z):
    u_0[w][0] = n_i[w]
    u_0[w][1] = n_n[w]
    tem_n_0.append(T_n)
    tem_i_0.append(T_i)
    coef_0.append(sqrt(8.0 * k_b * tem_i_0[w] / pi / m_e) * sigma * (e_ioniz / k_b / tem_i_0[w] + 2.0)\
    * exp(-e_ioniz / k_b / tem_i_0[w]) - 2.7 * 10**(-13) * (tem_i_0[w] / 11600.0)**(-0.75))
    pre_calc.append(2 * u_0[w][0] * k_b * tem_i_0[w])
    ioniz_frac_0.append(u_0[w][0]/u_0[w][1])

u_0 = np.array(u_0)
tem_i_0 = np.array(tem_i_0)
tem_n_0 = np.array(tem_n_0)
coef_0 = np.array(coef_0) 
pre_calc = np.array(pre_calc)   

######################### forward euler scheme ##############################################

def forward_euler(n_t, T, n_z, u_0, pre_calc, tem_i_0, tem_n_0, coef_0):
    dt = float(T/n_t)
    u = [[[0.0 for u_1 in range(2) ] for u_2 in range(n_z)] for u_3 in range(n_t)]
    tem_i = [[0.0 for tem_1 in range(n_z)] for tem_2 in range(n_t)]
    tem_n = [[0.0 for tem_1 in range(n_z)] for tem_2 in range(n_t)]
    pre_e = [[0.0 for pre_1 in range(n_z)] for pre_2 in range(n_t)]
    coef = [[0.0 for coef_1 in range(n_z)] for coef_2 in range(n_t)]
    ioniz_frac = [[0.0 for if_1 in range(n_z)] for if_2 in range(n_t)]
    u = np.array(u)
    tem_i = np.array(tem_i)
    tem_n = np.array(tem_n) 
    pre_e = np.array(pre_e)
    coef= np.array(coef)
    ioniz_frac = np.array(ioniz_frac)
    t = np.zeros(n_t+1)
    z = np.zeros(n_z+1)
    #V_i = 0
    u[0] = u_0
    tem_i[0] = tem_i_0
    tem_n[0] = tem_n_0
    pre_e[0] = pre_calc
    coef[0] = coef_0
    ioniz_frac[0] = ioniz_frac_0
    t[0] = 0
    z[0] = 0
    for i in range(n_t-1):
        for s_ind in range(n_z):
            for sp in range(2):
                t[i+1] = t[i] + dt
                u[i+1][s_ind][sp] = u[i][s_ind][sp] + (-1)**sp * dt * coef_0[s_ind] * u[i][s_ind][0] * u[i][s_ind][1] #tem  = const
            ioniz_frac[i+1][s_ind] = u[i][s_ind][0] / u[i][s_ind][1]
            #pre_e[i+1][s_ind] = 2 * u[i][s_ind][0] * k_b  * tem_i[i][s_ind]   
            #tem_i[i+1][s_ind] = pre_e[i][s_ind] / u[i][s_ind][0] / k_b / 2.0
            #coef[i+1][s_ind] = sqrt(8.0 * k_b * tem_i[i][s_ind] / pi / m_e) * sigma * (e_ioniz / k_b / tem_i[i][s_ind] + 2.0) \
            #* exp(-e_ioniz / k_b / tem_i[i][s_ind]) - 2.7 * 10**(-13) * (tem_i[i][s_ind] / 11600.0)**(-0.75)
		    
    print('dt = %f\n' %dt)
    return u, pre_e, tem_i, tem_n, coef, ioniz_frac, t 

########## plotting section ###################################################################

[u, pre_e, tem_i, tem_n, coef, ioniz_frac, t] = forward_euler(n_t, T, n_z, u_0, pre_calc, tem_i_0, tem_n_0, coef_0)
u = np.array(u)

plt.plot(x, np.log10(u[0][:][:]))
plt.plot(x, np.log10(u[n_t-1][:][:]), '--')
plt.legend(['protons', 'neutrals'])
plt.title('Altitude profiles of proton and hydrogen density')
plt.ylabel('Density, cm^-3, log scale')
plt.xlabel('Height, cm')

plt.figure()
plt.plot(x, np.log10(ioniz_frac[n_t-1][:]))
plt.xlabel('Height, cm')
plt.ylabel('n_i/n_n, log scale')
plt.title('Ionization rate')

#################################################################################################
# test if the number of particles in conserved
sum_part = []
for time_step in range(n_t):
    sum_part.append(np.sum(u[time_step][:][0])+np.sum(u[time_step][:][1]))
plt.figure()
plt.plot(np.diff(sum_part), '.')   
plt.xlabel('Number of timesteps')
plt.ylabel('Absolute value of deviation')
plt.title('Deviation from number of particles conservation (rounding)') 
