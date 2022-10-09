#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 17:47:01 2022

@author: mimfar
"""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

# from antenna_array import calc_AF, calc_AF_, plot_pattern ,peak_sll_hpbw_calc_I
import antenna_array as aa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import random



N = 16# number of elements
element_spacing = 0.5 
scan_angle = 30
theta_deg,AF_linear = aa.calc_AF(N,element_spacing,scan_angle)
G = 20 * np.log10(np.abs(AF_linear))
peak, theta_peak, SLL, HPBW = aa.calc_peak_sll_hpbw_calc(G, theta_deg) 
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(peak, theta_peak, SLL, HPBW))
peak_plot = 5 * (1 + int(peak/5))
aa.plot_pattern(theta_deg,G, xlim = (-90,90),ylim = (-20,peak_plot),xlab = r'$\theta$',ylab = 'dB(AF^2)')

# An array of N element with random element spacing choosen from element spacing list
element_spacing = [0.25,0.5,0.75,1 ]
X = np.cumsum(random.choices(element_spacing,k=N))
# X = np.linspace(0,N,N-1) * element_spacing
P = -2 * np.pi * X * np.sin(np.radians(scan_angle))
I = np.ones(X.shape)
Nt = 181 # length of theta vectoe
theta_deg,dtheta_deg = np.linspace(-90,90,Nt,retstep = True)
theta_deg = theta_deg.reshape(Nt,1)
AF_linear = aa.calc_AF_(X,I,P,theta_deg)
G = 20 * np.log10(np.abs(AF_linear))
peak, theta_peak, SLL, HPBW = aa.calc_peak_sll_hpbw_calc(G, theta_deg) 
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(peak, theta_peak, SLL, HPBW))
aa.plot_pattern(theta_deg,G, xlim = (-90,90),ylim = (-20,peak_plot),xlab = r'$\theta$',ylab = 'dB(AF^2)')
#

