# -*- coding: utf-8 -*-
"""
Created on Fri May 18 17:22:03 2018

@author: atkachen
"""

from math import sqrt, exp, pi 
import numpy as np
import matplotlib.pyplot as plt

################ input parameters ########################################################

n_t = 100   #number of timesteps
T = 0.001        #full time interval
n_z = 100     #number of spatial steps

###################### reading from file #################################################

file_tem = open("Initial_model.txt", "r").readlines() # we are going to read the file as a list of strings

height = []                    #these are our lists of relevant parameters in file 
temperature = []
electron_dens = []
proton_dens = []
hydr_dens = []
press_el = []

for line_numb in range(2, len(file_tem)):               #we start reading the data and writing it into our lists,
    file_tem[line_numb] = file_tem[line_numb].split()   #we split the strings from 2nd to last into a list of values  
    height.append(float(file_tem[line_numb][0]))
    temperature.append(float(file_tem[line_numb][1]))
    electron_dens.append(float(file_tem[line_numb][2]))
    proton_dens.append(float(file_tem[line_numb][3]))
    hydr_dens.append(float(file_tem[line_numb][4]))
    press_el.append(float(file_tem[line_numb][5]))    

############## interpolation of the values density(height), temperature(height) ##########

n_z = 100    
x = np.linspace(8.0*10**7, 1.5*10**8, n_z)

tem_interp = []
n_e_interp = []
n_i_interp = []
n_n_interp = []
p_e_interp = []

for val in x:
    for i in range(len(height)):
        if(val < height[i] and val > height[i+1]):
            k = i
            tem_interp.append(temperature[k] + (temperature[k+1]-temperature[k])*(val-height[k])/(height[k+1]-height[k]))
            n_e_interp.append(electron_dens[k] + (electron_dens[k+1]-electron_dens[k])*(val-height[k])/(height[k+1]-height[k]))  #the interpolation formula       
            n_i_interp.append(proton_dens[k] + (proton_dens[k+1]-proton_dens[k])*(val-height[k])/(height[k+1]-height[k]))
            n_n_interp.append(hydr_dens[k] + (hydr_dens[k+1]-hydr_dens[k])*(val-height[k])/(height[k+1]-height[k]))
            p_e_interp.append(press_el[k] + (press_el[k+1]-press_el[k])*(val-height[k])/(height[k+1]-height[k]))
        elif(val == height[i]):
            k = i
            tem_interp.append(temperature[k])
            n_e_interp.append(electron_dens[k])
            n_i_interp.append(proton_dens[k])
            n_n_interp.append(hydr_dens[k])
            p_e_interp.append(press_el[k])

################# calculating initial profiles and ioniz. coef ##############################
            
u_0 = [[0.0 for u_01 in range(2)] for u_02 in range(len(n_i_interp))]
tem_0 = []
coef_0 = []
pre_calc = []
 
for w in range(len(n_i_interp)):
    u_0[w][0] = n_i_interp[w]
    u_0[w][1] = n_n_interp[w]

k_b = 1.3807*10**(-16) #erg/K
m_e = 9.1094*10**(-28) #g
sigma = 2.0*10**(-17) #cm^2
e_ioniz = 2.18*10**(-11) #erg
    
for el in range(len(tem_interp)):
    tem_0.append(tem_interp[el])
    coef_0.append(sqrt(8.0 * k_b * tem_0[el] / pi / m_e) * sigma * (e_ioniz / k_b / tem_0[el] + 2.0)\
    * exp(-e_ioniz / k_b / tem_0[el]) - 2.7 * 10**(-13) * (tem_0[el] / 11600.0)**(-0.75))
    pre_calc.append(2 * u_0[el][0] * k_b * tem_interp[el])

u_0 = np.array(u_0)
tem_0 = np.array(tem_0)
coef_0 = np.array(coef_0) 
pre_calc = np.array(pre_calc)   

######################### forward euler scheme ##############################################

def forward_euler(n_t, T, n_z, u_0, pre_calc, tem_0, coef_0):
    dt = float(T/n_t)
    #dz = float(H/n_z)
    u = [[[0.0 for u_1 in range(2) ] for u_2 in range(n_z)] for u_3 in range(n_t)]
    tem = [[0.0 for tem_1 in range(n_z)] for tem_2 in range(n_t)]
    pre_e = [[0.0 for pre_1 in range(n_z)] for pre_2 in range(n_t)]
    coef = [[0.0 for coef_1 in range(n_z)] for coef_2 in range(n_t)]
    u = np.array(u)
    tem = np.array(tem)
    pre_e = np.array(pre_e)
    coef= np.array(coef)
    t = np.zeros(n_t+1)
    z = np.zeros(n_z+1)
    #V_i = 0
    u[0] = u_0
    tem[0] = tem_0
    pre_e[0] = pre_calc
    coef[0] = coef_0
    t[0] = 0
    z[0] = 0
    for i in range(n_t-1):
        for s_ind in range(len(n_n_interp)):
            for sp in range(2):
                t[i+1] = t[i] + dt
                u[i+1][s_ind][sp] = u[i][s_ind][sp] + (-1)**sp * dt * coef[i][s_ind] * u[i][s_ind][0] * u[i][s_ind][1]
            pre_e[i+1][s_ind] = 2 * u[i][s_ind][0] * k_b  * tem[i][s_ind]   
            tem[i+1][s_ind] = pre_e[i][s_ind] / u[i][s_ind][0] / k_b / 2.0
            coef[i+1][s_ind] = sqrt(8.0 * k_b * tem[i][s_ind] / pi / m_e) * sigma * (e_ioniz / k_b / tem[i][s_ind] + 2.0) \
            * exp(-e_ioniz / k_b / tem[i][s_ind]) - 2.7 * 10**(-13) * (tem[i][s_ind] / 11600.0)**(-0.75)
		
    print('dt = %f\n' %dt)
    return u, pre_e, tem, coef, t 

########## plotting section ###################################################################

[u, pre_e, tem, coef, t] = forward_euler(n_t, T, n_z, u_0, pre_calc, tem_0, coef_0)
u = np.array(u)

plt.plot(x, np.log10(u[0][:][:]))
plt.plot(x, np.log10(u[n_t-1][:][:]), '--')
plt.legend(['protons', 'neutrals'])
plt.title('Altitude profiles of proton and hydrogen density')
plt.ylabel('Density, cm^-3, log scale')
plt.xlabel('Height, cm')

plt.figure()
plt.plot(tem[n_t-1][:])
#################################################################################################

