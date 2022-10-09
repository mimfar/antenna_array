#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:04:17 2022

@author: mimfar
"""

import numpy as np
import matplotlib.pyplot as plt


def calc_AF_(X,I,P,theta,element_pattern=True):
    
    '''AF_calc_ calculates the Array Factor (AF) of a linear antenna array
    X (wavelength) : Position of array elements
    I(Linear)      : Excitation amplitude at each element
    P(Radian)      : Excitation phase at each element
    theta (deg)    : spatial angle range -90:90 with braodside=0
    element_pattern:Applies cosine envelope on top of the array factor
    The Gain is calculated for the array factor only not array factor x element pattern '''
    
    X = X.reshape(1,-1)
    theta = theta.reshape(-1,1)
    P = P.reshape(X.shape)
    I = I.reshape(X.shape)
    AF = np.sum(I * np.exp(1j * P + 1j * 2 * np.pi 
                  * np.dot (np.sin(np.radians(theta)),X)),axis = 1).reshape(theta.shape)
      
    delta_theta = (theta[1] - theta[0]) * np.pi / 180
    AF_int = 0.5 * np.sum(np.abs(AF)**2 * np.sin(np.radians(theta + 90))) * delta_theta  # integral of AF^2
    AF = AF/ (AF_int ** 0.5)
    
    if element_pattern:
        AF = AF * np.cos(np.radians(theta))
    
    return AF

def calc_AF(N,element_spacing,scan_angle,theta=None,element_pattern=True):
    
    '''AF_calc calculates the Array Factor (AF) of a linear antenna array with 
    uniform antenna element spacing 
    N               :  # of elements
    scan_angle (deg): A progressive phase shift will be applied to array elements to scan the beam to scan angle
    theta (deg)     : spatial angle range -90:90 with braodside=0
    element_pattern :Applies cosine envelope on top of the array factor
    The Gain is calculated for the array factor only not array factor x element pattern '''

    L = (N-1) * element_spacing
    X = np.linspace(0,N-1,N) * element_spacing
    P = -2 * np.pi * X * np.sin(np.radians(scan_angle)) 
    I = np.ones(X.shape)
    if theta == None:
        HPBW = 51 / L
        Nt = int(180 / (HPBW / 4))
        Nt = Nt + Nt % 2 # making Nt an odd number
        theta_deg,dtheta_deg = np.linspace(-90,90,Nt,retstep = True)
        return  theta_deg,calc_AF_(X,I,P,theta_deg,element_pattern=element_pattern)
    else:
        
        return  calc_AF_(X,I,P,theta_deg,element_pattern=element_pattern)
    
def plot_pattern(x,y,marker = '-',xlim = None, ylim = None, xlab = 'x',ylab = 'y'):
    plt.plot(x,y,marker)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim(np.min(x),np.max(x))
            
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim((np.min(y),np.max(y)))
    plt.grid(True)

def calc_peak_sll_hpbw_calc(G, theta_deg):
    '''Function calculates the Peak value and angle, SLL, and HPBW of G in dB
    assuming a paatern with a single peak (no grating lobes)'''
    G,theta_deg = np.ravel(G),np.ravel(theta_deg)
    peak,index_peak  = np.max(G), np.argmax(G) 
    theta_peak = theta_deg[index_peak]
    index_3dBR = index_peak + np.argmin(np.abs(G[index_peak:] - peak + 3))
    index_3dBL = np.argmin(np.abs(G[:index_peak] -peak + 3))
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
    return peak, theta_peak, SLL, HPBW



