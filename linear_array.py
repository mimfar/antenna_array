#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 23:21:26 2022

@author: mimfar
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

PI = np.pi

class LinearArray():
    ''' The class construct a linear antenna array object'''
    
    Version = '0.1'
    
    def __init__(self,num_elem,element_spacing,scan_angle=0,theta=[],element_pattern=True):
        self.num_elem = num_elem
        self.element_spacing = element_spacing
        self.scan_angle = scan_angle
        self.theta = theta
        self.element_pattern = element_pattern
        
    @property
    def calc_AF(self):
        '''AF_calc calculates the Array Factor (AF) of a linear antenna array with 
        uniform antenna element spacing 
        num_elem              :  # of elements
        scan_angle (deg): A progressive phase shift will be applied to array elements to scan the beam to scan angle
        theta (deg)     : spatial angle range -90:90 with braodside=0
        element_pattern :Applies cosine envelope on top of the array factor
        The Gain is calculated for the array factor only not array factor x element pattern '''

        L = (self.num_elem - 1) * self.element_spacing
        X = np.linspace(0,self.num_elem-1,self.num_elem) * self.element_spacing
        P = -2 * PI * X * np.sin(np.radians(self.scan_angle)) 
        I = np.ones(X.shape)
        if not any(self.theta):
            HPBW = 51 / L
            Nt = int(180 / (HPBW / 4))
            Nt = Nt + (Nt+1) % 2 # making Nt an odd number
            Nt = max(Nt,181) # 181 point is at least 1 degree theta resolution
            self.theta = np.linspace(-90,90,Nt)
            self.AF = self.calc_AF_(X,I,P)          
        else:
            self.AF = self.calc_AF_(X,I,P)
        return self.AF  
        
    def calc_AF_(self,X,I,P):

        '''AF_calc_ calculates the Array Factor (AF) of a linear antenna array
        X (wavelength) : Position of array elements
        I(Linear)      : Excitation amplitude at each element
        P(Radian)      : Excitation phase at each element
        theta (deg)    : spatial angle range -90:90 with braodside=0
        element_pattern:Applies cosine envelope on top of the array factor
        The Gain is calculated for the array factor only not array factor x element pattern '''
        
        X = X.reshape(1,-1)
        theta = self.theta.reshape(-1,1)
        P = P.reshape(X.shape)
        I = I.reshape(X.shape)
        AF = np.sum(I * np.exp(1j * P + 1j * 2 * np.pi 
                      * np.dot (np.sin(np.radians(theta)),X)),axis = 1).reshape(theta.shape)
          
        delta_theta = (theta[1] - theta[0]) * np.pi / 180
        AF_int = 0.5 * np.sum(np.abs(AF)**2 * np.sin(np.radians(theta + 90))) * delta_theta  # integral of AF^2
        AF = AF/ (AF_int ** 0.5)
        
        if self.element_pattern:
            AF = AF * np.cos(np.radians(theta))
        
        return AF

    def calc_peak_sll_hpbw_calc(self):
        '''Function calculates the Peak value and angle, SLL, and HPBW of G in dB
        assuming a pattern with a single peak (no grating lobes)'''
        G,theta_deg = np.ravel(20 * np.log10(np.abs(self.AF))),np.ravel(self.theta)
        peak,index_peak  = np.max(G), np.argmax(G) 
        theta_peak = theta_deg[index_peak]
        index_3dBR = index_peak + np.argmin(np.abs(G[index_peak:] - peak + 3))
        index_3dBL = np.argmin(np.abs(G[:index_peak] - peak + 3))
        HPBW = theta_deg[index_3dBR] - theta_deg[index_3dBL]
        
        dG = np.sign(np.diff(G))
        dG_cs = -dG[0:-1] * dG[1:]# change sign
        index_nullR = index_3dBR + 1
        while (dG_cs[index_nullR] == -1) & (index_nullR < len(dG_cs)):
            index_nullR += 1        
        index_nullL = index_3dBL - 1 
        while (dG_cs[index_nullL] == -1) & (index_nullL > 0):
            index_nullL -= 1
            
        SLL = peak - np.max([np.max(G[0:index_nullL]),np.max(G[index_nullR:]) ])
        # print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(peak, theta_peak, SLL, HPBW))
        pattern_params = namedtuple('pattern_params',['Gain','Peak_Angle','SLL','HPBW'])
        self.pattern_params = pattern_params(peak, theta_peak, SLL, HPBW)
        return self.pattern_params
    
    def plot_pattern(self,marker = '-',xlim = None, ylim = None, xlab = 'x',ylab = 'y'):
        AF_dB = 20 * np.log10(np.abs((self.AF)))
        peak_plot = 5 * (int(np.max(AF_dB) / 5) + 1)
        plt.plot(self.theta,AF_dB,marker)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        if xlim:
            plt.xlim(xlim)
        else:
            plt.xlim(np.min(self.theta),np.max(self.theta))
                
        if ylim:
            plt.ylim(ylim)
        else:
            plt.ylim(peak_plot-30,peak_plot)
        plt.grid(True)



