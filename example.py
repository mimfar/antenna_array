#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 17:47:01 2022

@author: mimfar
"""
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
if current_path not in sys.path: sys.path.insert(0,current_path)

from antenna_array import calc_AF, calc_AF_, plot_pattern,polar_pattern ,calc_peak_sll_hpbw, db20
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.signal.windows import get_window , taylor, chebwin


num_elem = 10# number of elements
element_spacing = 0.5 
scan_angle = 30
theta_deg,AF_linear = calc_AF(num_elem,element_spacing,scan_angle)
G = db20(AF_linear)
fig,ax = plot_pattern(theta_deg,G,xlim = (-90,90),xlab = r'$\theta$',ylab = 'dB(AF^2)')
ax.set_xticks(np.linspace(-90,90,7))
peak, theta_peak, SLL, HPBW = calc_peak_sll_hpbw(G, theta_deg) 
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(peak, theta_peak, SLL, HPBW))
fig,ax = polar_pattern(theta_deg,G)
#%%
# An array of N element with random element spacing choosen from element spacing list
element_spacing = [0.25,0.5,0.75,1 ]
X = np.cumsum(random.choices(element_spacing,k=num_elem))
P = -2 * np.pi * X * np.sin(np.radians(scan_angle))
I = np.ones(X.shape)
I = get_window('hamming', num_elem)
Nt = 181 # length of theta vectoe
theta_deg,dtheta_deg = np.linspace(-90,90,Nt,retstep = True)
theta_deg = theta_deg.reshape(Nt,1)
AF_linear = calc_AF_(X,I,P,theta_deg)
G = db20(AF_linear)
peak, theta_peak, SLL, HPBW = calc_peak_sll_hpbw(G, theta_deg) 
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(peak, theta_peak, SLL, HPBW))
# polar_pattern(theta_deg,G)
plot_pattern(theta_deg,G)

plt.show()
#%% Linear Array Class Example
from linear_array import LinearArray
num_elem = 10# number of elements
element_spacing = 0.5 
scan_angle = 30
la = LinearArray(num_elem,element_spacing,scan_angle=scan_angle)
la.calc_AF
ff = plt.figure()
_,aa = la.plot_pattern(fig=ff,marker='-',xlim=(-90,90),ylim=(-30,10))
params = la.calc_peak_sll_hpbw_calc()
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(*params))
la.calc_envelope(theta1=0,theta2=60,delta_theta=10)
fig1 ,ax1 = la.plot_envelope(plot_all=True);
la.polar_pattern();
fig1 = la.polar_envelope(plot_all=True);

#%% Linear Array Class Example nonuniform spacing sparse array with low side lobe level
# The nonunifrom spacing removes the grating lobe
num_elem = 32
dx = sorted(random.choices([0.5,0.75,1.25,2],k=int(num_elem/2),weights = [1, .75, 0.5, 0.5]))
element_spacing = np.hstack((np.flip(dx),dx[1:])) 
la = LinearArray(num_elem,element_spacing,scan_angle=scan_angle)
la.calc_AF
ff = plt.figure()
_,aa = la.plot_pattern(fig=ff,marker='-',xlim=(-90,90))
params = la.calc_peak_sll_hpbw_calc()
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(*params))


element_spacing_uniform = np.mean(element_spacing)
la_uniform = LinearArray(num_elem,element_spacing_uniform,scan_angle=scan_angle)
la_uniform.calc_AF
la_uniform.plot_pattern(fig=ff,marker='--',xlim=(-90,90),ylim=(-20,25))
params = la_uniform.calc_peak_sll_hpbw_calc()
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(*params))
a2 = ff.add_axes([0.15, 0.75, 0.7, 0.1])
a2.plot(la.X,np.ones(la.X.shape),'.b')
a2.plot(la_uniform.X,-np.ones(la_uniform.X.shape),'xr')
a2.set_ylim(-2,2)
a2.patch.set_alpha(0.15)
# a2.set_yticklabels('');
#%% Linear array example construct from element position
X = 4 * np.random.randn(num_elem)
la = LinearArray.from_element_position(X)
la.calc_AF
ff = plt.figure()
_,aa = la.plot_pattern(fig=ff,marker='-',xlim=(-90,90),ylim=(-20,20))
params = la.calc_peak_sll_hpbw_calc()
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(*params))
a2 = ff.add_axes([0.2, 0.2, 0.6, 0.1])
a2.plot(la.X,np.zeros(la.X.shape),'.b')
a2.patch.set_alpha(0.15)
a2.set_yticklabels('');

#%%
# An array of num_elem element with amlitude tapering / winodowing to reduce side lobe level
num_elem = 16
element_spacing = 0.5
scan_angle = 0
X = np.linspace(0,num_elem-1,num_elem) * element_spacing
P = -2 * np.pi * X * np.sin(np.radians(scan_angle))

Nt = 721 # length of theta vectoe
theta_deg,dtheta_deg = np.linspace(-90,90,Nt,retstep = True)
theta_deg = theta_deg.reshape(Nt,1)
fig, ax = plt.subplots(figsize=(8,6))
fig1, ax1 = plt.subplots(figsize=(8,6))

# based on window type
window_list = ['boxcar','hamming','blackman','blackmanharris']
df = pd.DataFrame(columns = ['Peak','SLL','HPBW'])
for window in window_list:
    I = get_window(window, num_elem)

    plt.sca(ax1) 
    plt.plot(I,'-o')

    AF_linear = calc_AF_(X,I,P,theta_deg,element_pattern=False)
    G = db20(AF_linear)
    plot_pattern(theta_deg,G-10*np.log10(num_elem),ylim=(-110,0),fig=fig)

    peak, theta_peak, SLL, HPBW = calc_peak_sll_hpbw(G, theta_deg)
    df.loc[window] = [peak, SLL, HPBW]


# plt.show()
plt.legend(window_list)
plt.sca(ax1) 
plt.legend(window_list)
plt.grid()
print(df.round(1))

#based on SLL level
fig, ax = plt.subplots(figsize=(8,6))
fig1, ax1 = plt.subplots(figsize=(8,6))

for SLL_target in [20,30,40,50,70,100]:
    if SLL_target < 50:    
        I = taylor(num_elem, nbar=5, sll=SLL_target)
    else:
        I = chebwin(num_elem, SLL_target)
        
    plt.sca(ax1) 
    plt.plot(I,'-o')

    AF_linear = calc_AF_(X,I,P,theta_deg,element_pattern=False)
    G = db20(AF_linear)
    plot_pattern(theta_deg,G-10*np.log10(num_elem),ylim=(-110,0),fig=fig)

    peak, theta_peak, SLL, HPBW = calc_peak_sll_hpbw(G, theta_deg)
    df.loc[SLL_target] = [peak, SLL, HPBW]

print(df.round(1))

#%% Linear Array Class with amplitude tapering Example
from linear_array import LinearArray
num_elem = 16# number of elements
element_spacing = 0.5 
scan_angle = 0
la = LinearArray(num_elem,element_spacing,scan_angle=scan_angle,window='hann')
la.calc_AF
ff = plt.figure()
_,ax1 = la.plot_pattern(fig=ff,marker='-',xlim=(-90,90),ylim=(-60,15))
params = la.calc_peak_sll_hpbw_calc()
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(*params))
la.SLL = 55
la.calc_AF
_,ax2 = la.plot_pattern(fig=ff,marker='-',xlim=(-90,90),ylim=(-60,15),xlab='theta',ylab='Array Gain(dB)')
params = la.calc_peak_sll_hpbw_calc()
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(*params))
