# -*- coding: utf-8 -*-
"""
Created on Fri May 18 17:22:03 2018

@author: atkachen
"""

from math import sqrt, exp, log, pi 
import scipy
import numpy as np
import matplotlib.pyplot as plt

file_tem = open("temperature_model1001_modified_reduced.txt", "r").readlines() # we are going to read the file as a list of strings

height = []                    #these are our lists of relevant parameters in file 
temperature = []
electron_dens = []
proton_dens = []
hydr_dens = []

for line_numb in range(2, len(file_tem)):               #we start reading the data and writing it into our lists,
    file_tem[line_numb] = file_tem[line_numb].split()   #we split the strings from 2nd to last into a list of values  
    height.append(float(file_tem[line_numb][0]))
    temperature.append(float(file_tem[line_numb][1]))
    electron_dens.append(float(file_tem[line_numb][2]))
    proton_dens.append(float(file_tem[line_numb][3]))
    hydr_dens.append(float(file_tem[line_numb][4]))

##############interpolation of the values dens(height)
n_z = 100    
x = np.linspace(8.0*10**7, 1.5*10**8, n_z)

tem_interp = []
n_e_interp = []
n_i_interp = []
n_n_interp = []

for val in x:
    for i in range(len(height)):
        if(val < height[i] and val > height[i+1]):
            k = i
            tem_interp.append(temperature[k] + (temperature[k+1]-temperature[k])*(val-height[k])/(height[k+1]-height[k]))
            n_e_interp.append(electron_dens[k] + (electron_dens[k+1]-electron_dens[k])*(val-height[k])/(height[k+1]-height[k]))  #the interpolation formula       
            n_i_interp.append(proton_dens[k] + (proton_dens[k+1]-proton_dens[k])*(val-height[k])/(height[k+1]-height[k]))
            n_n_interp.append(hydr_dens[k] + (hydr_dens[k+1]-hydr_dens[k])*(val-height[k])/(height[k+1]-height[k]))    
        elif(val == height[i]):
            k = i
            tem_interp.append(temperature[k])
            n_e_interp.append(electron_dens[k])
            n_i_interp.append(proton_dens[k])
            n_n_interp.append(hydr_dens[k])

################# calculating initial profiles and ioniz. coef ###################################
            
u_0 = [[0 for s in range(2)] for v in range(len(n_i_interp))]            
for w in range(len(n_i_interp)):
    u_0[w][0] = n_i_interp[w]
    u_0[w][1] = n_n_interp[w]           
u_0 = np.array(u_0)


k_b = 1.3807*10**(-16) #erg/K
m_e = 9.1094*10**(-28) #g
sigma = 2.0*10**(-17) #cm^2
e_ioniz = 2.18*10**(-11) #erg



temp = np.array(tem_interp)
coef = []
for el in temp:
    coef.append(sqrt(8.0 * k_b * el / pi / m_e) * sigma * (e_ioniz / k_b / el + 2.0) * exp(-e_ioniz / k_b / el) - 2.7 * 10**(-13) * (el / 11600.0)**(-0.75))
coef = np.array(coef) 


#####################################################################

def forward_euler(n_t, T, n_z, u_0, coef):
    dt = float(T/n_t)
    #dz = float(H/n_z)
    u = [[[0.0 for k in range(2) ] for l in range(n_z)] for m in range(n_t)]
    u = np.array(u)
    t = np.zeros(n_t+1)
    z = np.zeros(n_z+1)
    #V_i = 0
    u[0] = u_0
    t[0] = 0
    z[0] = 0
    for i in range(n_t-1):
        for s_ind in range(len(n_n_interp)):
            for sp in range(2):
                t[i+1] = t[i] + dt
        #z[i+1] = z[i] + dz
        #u[i+1] = u[i] + dt*dz*coef*u[i][0]*u[i][1]/(dz+dt*V_i)
                u[i+1][s_ind][sp] = u[i][s_ind][sp] + (-1)**sp*dt*coef[s_ind]*u[i][s_ind][0]*u[i][s_ind][1]
    print('dt = %f' %dt)
    return u, t#, z

########## customer interface ##############################
    
n_t = 1000   #number of timesteps
T = 1        #full time interval
n_z = 100     #number of spatial steps

[pl, t] = forward_euler(n_t, T, n_z, u_0, coef)
pl = np.array(pl)
#plt.figure()
plt.plot(x, np.log10(pl[0][:][:]))
plt.plot(x, np.log10(pl[999][:][:]), '--')
#plt.figure()
#plt.plot(x, np.log10(pl[500000][:][:]))
#plt.plot(x, np.log10(pl[999999][:][:]), '--')


#plt.plot(z, pl)
plt.legend(['protons', 'neutrals'])
plt.title('Altitude profiles of proton and hydrogen density')
plt.ylabel('Density, cm^-3, log scale')
plt.xlabel('Height, cm')
##############################################################

