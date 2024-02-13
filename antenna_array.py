#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:04:17 2022

@author: mimfar
"""

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def db(m,x):
    return m * np.log10(np.abs(x))

db10 = partial(db,10)
db20 = partial(db,20)

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
        Nt = max(Nt,181) # 181 point is at least 1 degree theta resolution
        theta_deg,dtheta_deg = np.linspace(-90,90,Nt,retstep = True)
        return  theta_deg,calc_AF_(X,I,P,theta_deg,element_pattern=element_pattern)
    else:
        
        return  calc_AF_(X,I,P,theta_deg,element_pattern=element_pattern)
    

def calc_peak_sll_hpbw(G, theta_deg):
    '''Function calculates the Peak value and angle, SLL, and HPBW of G in dB
    assuming a pattern with a single peak (no grating lobes)'''
    G,theta_deg = np.ravel(G),np.ravel(theta_deg)
    peak,idx_peak  = np.max(G), np.argmax(G) 
    theta_peak = theta_deg[idx_peak]
    
    dG = np.sign(np.diff(G))
    dG_cs = -dG[0:-1] * dG[1:]# change sign in derivative (peaks & nulls)
    dG_cs = np.insert(np.append(dG_cs,1),0,1) 
    cs_idx = np.asarray(dG_cs == 1).nonzero()[0] # idx of peaks and nulls
    idx_ = np.asarray(cs_idx == idx_peak).nonzero()[0][0]
    idx_null_L, idx_null_R= cs_idx[idx_-1],cs_idx[idx_+1]
    
    idx_3dB_R = idx_peak + np.argmin(np.abs(G[idx_peak:idx_null_R] - peak + 3))
    idx_3dB_L = idx_null_L + np.argmin(np.abs(G[idx_null_L:idx_peak] - peak + 3))
    HPBW = theta_deg[idx_3dB_R] - theta_deg[idx_3dB_L]    
    SLL = peak - np.max([np.max(G[0:idx_null_L]),np.max(G[idx_null_R:])])
    return peak, theta_peak, SLL, HPBW

def plot_pattern(x,y,fig=None,marker = '-',xlim = None, ylim = None, xlab = 'x',ylab = 'y',title=''):
    peak_plot = 5 * (int(np.max(y) / 5) + 1)
    if not isinstance(fig, matplotlib.figure.Figure):
        fig, ax = plt.subplots(figsize=(6,4))
    if fig.axes[0]:
        plt.sca(fig.axes[0]) 

    else:
        plt.sca(fig.add_subplot(111))
        
    
    plt.plot(x,y,marker)
    ax = plt.gca()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim(np.min(x),np.max(x))
            
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim((peak_plot-30,peak_plot))
    plt.grid(True)
    
    return fig,ax

def polar_pattern(x,y,fig=None,marker = '-',rlim = None, tlim = None ,title=''):
    peak_plot = 5 * (int(np.max(y) / 5) + 1)
    if not isinstance(fig, matplotlib.figure.Figure):
        fig, ax = plt.subplots(figsize=(8,6),subplot_kw={'projection': 'polar'})
        
    ax.plot(np.radians(x), y)
    ax.set_thetagrids(angles=np.linspace(-90,90,13))
    if tlim:
        ax.set_thetamin(tlim[0])
        ax.set_thetamax(tlim[1])
    else:
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
            
    if rlim:
        ax.set_rmax(rlim[1])
        ax.set_rmin(rlim[0])
    else:
        ax.set_rmax(peak_plot)
        ax.set_rmin(peak_plot-30)

    ax.grid(True)
    ax.set_theta_zero_location("N")
    ax.set_rlabel_position(180)  # Move radial labels away from plotted line
    ax.set_theta_direction('clockwise')
    
    return fig,ax





