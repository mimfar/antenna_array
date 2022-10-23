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
import matplotlib.pyplot as plt
import random



N = 10# number of elements
element_spacing = 0.5 
scan_angle = 30
theta_deg,AF_linear = calc_AF(N,element_spacing,scan_angle)
G = db20(AF_linear)
fig,ax = plot_pattern(theta_deg,G,xlim = (-90,90),xlab = r'$\theta$',ylab = 'dB(AF^2)')
ax.set_xticks(np.linspace(-90,90,7))
peak, theta_peak, SLL, HPBW = calc_peak_sll_hpbw(G, theta_deg) 
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(peak, theta_peak, SLL, HPBW))
fig,ax = polar_pattern(theta_deg,G)

# An array of N element with random element spacing choosen from element spacing list
element_spacing = [0.25,0.5,0.75,1 ]
X = np.cumsum(random.choices(element_spacing,k=N))
# X = np.linspace(0,N,N-1) * element_spacing
P = -2 * np.pi * X * np.sin(np.radians(scan_angle))
I = np.ones(X.shape)
Nt = 181 # length of theta vectoe
theta_deg,dtheta_deg = np.linspace(-90,90,Nt,retstep = True)
theta_deg = theta_deg.reshape(Nt,1)
AF_linear = calc_AF_(X,I,P,theta_deg)
G = db20(AF_linear)
peak, theta_peak, SLL, HPBW = calc_peak_sll_hpbw(G, theta_deg) 
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(peak, theta_peak, SLL, HPBW))
polar_pattern(theta_deg,G)
plt.show()
#%% Linear Array Class Example
from linear_array import LinearArray
N = 10# number of elements
element_spacing = 0.5 
scan_angle = 30
la = LinearArray(N,element_spacing,scan_angle=scan_angle)
la.calc_AF
ff = plt.figure()
_,aa = la.plot_pattern(fig=ff,marker='-',xlim=(-90,90),ylim=(-30,10))
params = la.calc_peak_sll_hpbw_calc()
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(*params))
la.calc_envelope(theta1=0,theta2=60,delta_theta=10)
fig1 ,ax1 = la.plot_envelope(plot_all=True);
la.polar_pattern();
fig1 = la.polar_envelope(plot_all=True);
