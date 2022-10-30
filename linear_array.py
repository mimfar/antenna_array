#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 23:21:26 2022

@author: mimfar
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, Iterable
from functools import partial
import matplotlib
PI = np.pi

def db(m,x):
    return m * np.log10(np.abs(x))

db10 = partial(db,10)
db20 = partial(db,20)
    

class LinearArray():
    ''' The class construct a linear antenna array object'''
    
    Version = '0.1'
    
    def __init__(self,num_elem,element_spacing,scan_angle=0,theta=[],element_pattern=True):
        
        '''AF_calc calculates the Array Factor (AF) of a linear antenna array with 
        uniform antenna element spacing 
        num_elem              :  # of elements
        scan_angle (deg): A progressive phase shift will be applied to array elements to scan the beam to scan angle
        theta (deg)     : spatial angle range -90:90 with braodside=0
        element_pattern :Applies cosine envelope on top of the array factor
        The Gain is calculated for the array factor only not array factor x element pattern '''
        
        assert num_elem > 0 , ('array length can not be zero')
        self.num_elem = num_elem
        self.element_spacing = element_spacing
        if isinstance(self.element_spacing,(int,float)):   
            self.X = np.linspace(0,self.num_elem-1,self.num_elem) * self.element_spacing
        elif isinstance(self.element_spacing,Iterable): 
            self.X = np.insert(np.cumsum(element_spacing),0,0)
            self.num_elem = len(self.X)
        self.X = self.X - np.mean(self.X)
        self.scan_angle = scan_angle
        self.theta = theta
        self.element_pattern = element_pattern
        
    @classmethod
    def from_element_position(cls,X,**kwargs):
        return cls(len(X),np.diff(sorted(X)),**kwargs)
        
    @property
    def calc_AF(self):
        

        array_length = max(self.X) - min(self.X)
        self.P = -2 * PI * self.X * np.sin(np.radians(self.scan_angle)) 
        self.I = np.ones(self.X.shape)
        if not any(self.theta):
            HPBW = 51 / array_length
            Nt = int(180 / (HPBW / 4))
            Nt = Nt + (Nt+1) % 2 # making Nt an odd number
            Nt = max(Nt,181) # 181 point is at least 1 degree theta resolution
            self.theta = np.linspace(-90,90,Nt)
            self.AF = self.calc_AF_()          
        else:
            self.AF = self.calc_AF_()
        return self.AF  
        
    def calc_AF_(self):
        
        self.X = self.X.reshape(1,-1)
        theta = self.theta.reshape(-1,1)
        self.P = self.P.reshape(1,-1)
        self.I = self.I.reshape(1,-1)
        AF = np.sum(self.I * np.exp(1j * self.P + 1j * 2 * np.pi 
                      * np.dot (np.sin(np.radians(theta)),self.X)),axis = 1).reshape(theta.shape)
          
        delta_theta = (theta[1] - theta[0]) * np.pi / 180
        AF_int = 0.5 * np.sum(np.abs(AF)**2 * np.sin(np.radians(theta + 90))) * delta_theta  # integral of AF^2
        AF = AF/ (AF_int ** 0.5)
        
        if self.element_pattern:
            AF = AF * np.cos(np.radians(theta))
        
        return AF.ravel()

    def calc_peak_sll_hpbw_calc(self):
        '''Function calculates the Peak value and angle, SLL, and HPBW of G in dB
        assuming a pattern with a single peak (no grating lobes)'''
        G,theta_deg = np.ravel(db20(self.AF)),np.ravel(self.theta)
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
        pattern_params = namedtuple('pattern_params',['Gain','Peak_Angle','SLL','HPBW'])
        self.pattern_params = pattern_params(peak, theta_peak, SLL, HPBW)
        return self.pattern_params
    
    @staticmethod
    def _plot(x,y,fig=None,marker = '-',xlim = None, ylim = None, xlab = 'x',ylab = 'y',title = ''):
        peak_plot = 5 * (int(np.max(y) / 5) + 1)
        if not isinstance(fig, matplotlib.figure.Figure):
            fig, ax = plt.subplots(figsize=(8,6))

        plt.plot(x,y,marker)
        ax = plt.gca()
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        if not xlim:
           xlim = (np.min(x),np.max(x))
        if not ylim:
            ylim = ((peak_plot-30,peak_plot))
        
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(True)
        return  fig, ax

    @staticmethod
    def _polar(t,r,fig=None,marker = '-',tlim = None, rlim = None ,title=''):
        peak_plot = 5 * (int(np.max(r) / 5) + 1)
        if not isinstance(fig, matplotlib.figure.Figure):
            fig, ax = plt.subplots(figsize=(8,6),subplot_kw={'projection': 'polar'})
        else:
            ax = fig.add_axes([0, 0, 1.6, 1.2], polar=True)
            
        ax.plot(np.radians(t), r)
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
        
    def plot_pattern(self,**kwargs):
        return self._plot(self.theta,db20(self.AF),**kwargs)
        
    def plot_envelope(self,plot_all=True,**kwargs):
        if plot_all:    
            return self._plot(self.theta,self.envelopes,**kwargs)
        else:
            return self._plot(self.theta,self.envelopes[:,-1 ],**kwargs)

    def polar_pattern(self,**kwargs):
        return self._polar(self.theta,db20(self.AF),**kwargs)  
    
    def polar_envelope(self,plot_all=True,**kwargs):
        if plot_all:    
            return self._polar(self.theta,self.envelopes,**kwargs)
        else:
            return self._polar(self.theta,self.envelopes[:,-1 ],**kwargs)
       
    def calc_envelope(self,theta1=0,theta2=45,delta_theta=5):
        N = int((theta2 - theta1)/delta_theta)
        self.scan_range = np.linspace(theta1,theta2,N+1)
        scan_angle_ = self.scan_angle
        self.envelopes = np.zeros((N+1,len(self.theta)))
        for idx,scan_angle in enumerate(self.scan_range):
            self.scan_angle = scan_angle
            self.envelopes[idx,:] = db20(self.calc_AF.ravel())
        self.envelopes[N,:] = np.max(self.envelopes[:-1,:],axis=0)
        self.envelopes = self.envelopes.T
        self.scan_angle = scan_angle_

