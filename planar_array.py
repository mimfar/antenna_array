#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 23:29:45 2023

@author: mimfar
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import get_window, taylor, chebwin

from collections import namedtuple, Iterable
from functools import partial
import matplotlib
from matplotlib import cm


PI = np.pi

def db(m,x):
    return m * np.log10(np.abs(x))

db10 = partial(db,10)
db20 = partial(db,20)
    

class PlanarArray():
    ''' The class construct a planr antenna array object'''
    
    Version = '0.1'
    
    def __init__(self,num_elem,element_spacing,scan_angle=(0,0),theta=[],phi=[],element_pattern=True,window=None,SLL=None):
        
        '''AF_calc calculates the Array Factor (AF) of a planar antenna array with either rect or tri grids
        
        num_elem              : a tuple of  # of elements in rows and columns
        scan_angle (deg): a tuple of angels (theta_scan,phi_scan) A progressive phase shift will 
                          be applied to array elements to scan the beam to scan angle
        theta (deg), phi(deg)  : spatial angle theta:0-180,phi=0-360 with  braodside=(0,0)
        element_pattern : Applies cosine envelope on top of the array factor
        '''
        
        assert num_elem[0] > 0 , ('array length can not be zero')
        assert num_elem[1] > 0 , ('array length can not be zero')
        self.num_elem = num_elem
        self.element_spacing = element_spacing
        self.row = np.linspace(0,self.num_elem[0]-1,self.num_elem[0]) * self.element_spacing[0]
        self.col = np.linspace(0,self.num_elem[1]-1,self.num_elem[1]) * self.element_spacing[1]
        self.scan_angle = scan_angle
        self.theta = theta
        self.phi = phi
        self.element_pattern = element_pattern
        self.window = window
        self.SLL = SLL
        
    @classmethod
    def from_element_position(cls,X,**kwargs):
        return cls(len(X),np.diff(sorted(X)),**kwargs)
        
    @property
    def calc_AF(self):
        

        array_length = max(self.row[-1] ,self.col[-1])
        if not any(self.theta):
            HPBW = 51 / array_length
            Nt = int(180 / (HPBW / 4))
            Nt = Nt + (Nt+1) % 2 # making Nt an odd number
            Nt = max(Nt,181) # 181 point is at least 1 degree theta resolution
            self.theta = np.linspace(0,180,Nt)
        if not any(self.phi):
            self.phi = np.linspace(0,360,361)
        
        self.row_ = np.reshape(self.row,(-1,1,1))   
        self.col_ = np.reshape(self.col,(-1,1,1))        


        self.Pcol = -2 * PI * self.col_ * np.sin(np.radians(self.scan_angle[0])) * np.cos(np.radians(self.scan_angle[1])) 
        self.Prow = -2 * PI * self.row_ * np.sin(np.radians(self.scan_angle[0])) * np.sin(np.radians(self.scan_angle[1])) 
        
        self.Icol = np.ones(self.col_.shape)
        self.Irow = np.ones(self.row_.shape)

        if self.window:
            self.I = get_window(self.window, self.num_elem)
        if self.SLL:
            if self.SLL < 50:
                self.I = taylor(self.num_elem, nbar=5, sll=self.SLL)
            else:
                self.I = chebwin(self.num_elem, self.SLL)
        
 
        self.AF = self.calc_AF_()          
       
        return self.AF  
        
    def calc_AF_(self):
        print(self.theta.shape)
        theta = self.theta.reshape(1,-1)
        phi = self.phi.reshape(-1,1)
        CPST = np.matmul(np.cos(phi * PI/180),np.sin(theta * PI/180))
        SPST = np.matmul(np.sin(phi * PI/180),np.sin(theta * PI/180))

        AFrow = np.sum(self.Irow * np.exp(1j * self.Prow + 1j * 2 * np.pi * np.tensordot(self.row,SPST,axes = 0)),axis=0)
        AFcol = np.sum(self.Icol * np.exp(1j * self.Pcol + 1j * 2 * np.pi * np.tensordot(self.col,CPST,axes = 0)),axis=0)
        AF = AFrow * AFcol
        T = np.tile(theta,(len(phi),1))
        cos_theta = np.cos(np.radians(T))
        cos_theta[T>89] = np.cos(np.radians(89.5))
        
        if self.element_pattern:
            AF = AF * cos_theta

        delta_theta = (self.theta[1] - self.theta[0]) * np.pi / 180
        delta_phi= (phi[1] - phi[0]) * np.pi / 180
            
        AF_int =  np.sum(np.abs(AF)**2 * np.sin(np.radians(T))) * delta_theta * delta_phi / 4 / PI # integral of AF^2
        AF = AF/ (AF_int ** 0.5)
        
        return AF

    def pattern_cut(self,cut_angle):
        G = db20(self.AF)
        theta_cut = np.hstack((-np.flip(self.theta[1:]), self.theta))
        idx_phi_cut1 = np.argmin(np.abs(self.phi - cut_angle))
        idx_phi_cut2 = np.argmin(np.abs(self.phi - (180 + cut_angle)))
        return theta_cut , np.hstack((np.flip(G[idx_phi_cut2,1:]), G[idx_phi_cut1,:]) )
        
    
    def calc_peak_sll_hpbw_calc(self):
        '''Function calculates the Peak value and angle, SLL, and HPBW of G in dB
        assuming a pattern with a single peak (no grating lobes)'''
        theta_deg,G = self.pattern_cut(self.scan_angle[1])
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
        self.pattern_params = pattern_params(float(f'{peak:1.1f}'), float(f'{theta_peak:1.1f}'), float(f'{SLL:1.1f}'), float(f'{HPBW:1.1f}'))
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
        
    def plot_pattern(self,cut_angle=None,**kwargs):
        if not cut_angle:
            cut_angle = self.scan_angle[1]
        theta_deg,G = self.pattern_cut(cut_angle)
        
        return self._plot(theta_deg,G,**kwargs)
        
    def plot_envelope(self,plot_all=True,**kwargs):
        if plot_all:    
            return self._plot(self.theta,self.envelopes,**kwargs)
        else:
            return self._plot(self.theta,self.envelopes[:,-1 ],**kwargs)

    def polar_pattern(self,cut_angle=None,**kwargs):
        if not cut_angle:
            cut_angle = self.scan_angle[1]
        theta_deg,G = self.pattern_cut(cut_angle)
        return self._polar(theta_deg,G,**kwargs)  
    
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

    def plot_pattern3D(self):
        pass
    
    
    def polar3D(self,**kwargs):
        G =  20 * np.log10(np.abs(self.AF))
        [T,P] = np.meshgrid(self.theta,self.phi)
        return self._polar3D(T,P,G,**kwargs)
    
        
    
    @staticmethod
    def _polar3D(T,P,G,g_range=30,fig=None,title=''):
        
        if not isinstance(fig, matplotlib.figure.Figure):
            fig, ax = plt.subplots(figsize=(8,6),subplot_kw={'projection':'3d'})
        else:
            ax = fig.add_subplot(projection='3d')
        
        peak = np.max(G)

        G = G - peak + g_range
        G[G < (peak - g_range)] = 0

        X = G * np.sin(np.radians(T)) * np.cos(np.radians(P))
        Y = G * np.sin(np.radians(T)) * np.sin(np.radians(P))
        Z = G * np.cos(np.radians(T))
        norm = plt.Normalize()

        rat = 1 * g_range
        ax.plot_surface(X,Y,Z,facecolors=plt.cm.jet(norm(G)),rstride=2,cstride=1,alpha=.5)
        ax.plot([-rat,rat],[0,0],'black')
        ax.plot([0,0],[-rat,rat],'black')
        ax.plot([0,0],[0,0],[-rat,rat],'black')
        ax.text(rat,0,0,'x,\nphi=0')
        ax.text(0,rat,0,'y,\nphi=90')
        ax.text(0,0,rat,'z')

        tt = np.linspace(0,360,361) * np.pi / 180
        ax.plot(rat * np.cos(tt),rat * np.sin(tt),'--',color=[0.5,0.5,0.5])
        ax.plot(np.zeros(tt.shape),rat * np.cos(tt),rat * np.sin(tt),'--',color=[0.5,0.5,0.5])
        ax.plot(rat * np.cos(tt),np.zeros(tt.shape),rat * np.sin(tt),'--',color=[0.5,0.5,0.5])

        ax.set_zlim([-rat,rat])
        ax.set_xlim([-rat,rat])
        ax.set_ylim([-rat,rat])

        plt.title(title)
        plt.axis('off')
        # ax.plot_wireframe(X,Y,Z,rstride=20,cstride=20)
        ax.view_init(elev=24, azim=25)
        fig.colorbar(cm.ScalarMappable(norm=plt.Normalize(vmin=peak-g_range,vmax=peak), cmap='jet'),shrink=0.4)

        return fig,ax;
    
    def pattern_contour(self,**kwargs):
        G =  20 * np.log10(np.abs(self.AF))
        [T,P] = np.meshgrid(self.theta,self.phi)
        return self._plot_contour(T,P,G,**kwargs);
        
    @staticmethod
    def _plot_contour(T,P,G,g_range=30,fig=None,tlim = None, plim = None,tlab='theta',plab='phi',title=''):
        
        if not isinstance(fig, matplotlib.figure.Figure):
            fig, ax = plt.subplots(figsize=(8,6))
        else:
            ax = fig.add_axes([0, 0, 1.6, 1.2], polar=True)
            
        peak = np.max(G)
        G[G < (peak - g_range)] = peak - g_range
        G1 = np.vstack((G[int(G.shape[0]/2)+1:,:],G[:int(G.shape[0]/2)+1,:]))
        plt.contourf(P-180,T,G1,cmap='hot')
        plt.colorbar()
        plt.clim([peak-g_range,peak])
        plt.xlabel(plab);
        plt.ylabel(tlab);
        plt.ylim(tlim);
        plt.xlim(plim);
        plt.title(title);
        
        return fig,ax;

    def polarsurf(self,g_range=30,fig=None,title='Polar Surf'):
        if not isinstance(fig, matplotlib.figure.Figure):
            fig, ax = plt.subplots(figsize=(8,6))
        else:
            ax = fig.add_axes([0, 0, 1.6, 1.2], polar=True)
        
        G =  20 * np.log10(np.abs(self.AF))
        [T,P] = np.meshgrid(self.theta,self.phi)
        peak = np.max(G)
        X = np.sin(np.radians(T)) * np.cos(np.radians(P))

        Y = np.sin(np.radians(T)) * np.sin(np.radians(P))
        G[G < (peak - g_range)] = peak - g_range

        for p in np.linspace(0,330,12) * np.pi / 180:
            plt.plot([0, np.cos(p)],[0, np.sin(p)],'--',color=[0.5,0.5,0.5],alpha=0.5)
            plt.text(1.07 * np.cos(p),1.07 * np.sin(p),f'{(p*180/np.pi):1.0f}')
        tt = np.linspace(0,360,361) * np.pi / 180

        for t in np.linspace(0,90,4) * np.pi / 180:
            plt.plot(np.sin(t) * np.cos(tt), np.sin(t) * np.sin(tt),'--',color=[0.5,0.5,0.5],alpha=0.5)
            if t <=np.pi/3:
                plt.text(np.sin(t),0.05,f'{(t*180/np.pi):1.0f}')

        plt.axis('equal')
        plt.axis('off')
        plt.contourf(X,Y,G,cmap='hot')
        plt.colorbar()
        plt.clim([peak-g_range,peak])
        plt.title(title)
        plt.text(np.cos(np.pi/20),np.sin(np.pi/20),' phi')
        plt.plot(np.cos(np.pi/18),np.sin(np.pi/18),'^',color='black')
        return fig,ax;
    def polarsphere(self,g_range=30,fig=None):
        
        if not isinstance(fig, matplotlib.figure.Figure):
            fig, ax = plt.subplots(figsize=(8,6),subplot_kw={'projection':'3d'})
        else:
            ax = fig.add_subplot(projection='3d')
        
        
        [T,P] = np.meshgrid(self.theta,self.phi)
        G =  20 * np.log10(np.abs(self.calc_AF))
        peak = np.max(G)
        r_range = 30
        G[G < (peak - g_range)] = peak - g_range
        norm = plt.Normalize()

        X = np.sin(np.radians(T)) * np.cos(np.radians(P))
        Y = np.sin(np.radians(T)) * np.sin(np.radians(P))
        Z = np.cos(np.radians(T))

        ax.plot_surface(X,Y,Z,facecolors=plt.cm.jet(norm(G)),rstride=2,cstride=1,alpha=.7)
        plt.axis('off')
        rat = 1.25
        # ax.view_init(elev=90, azim=-90)
        ax.plot([0,rat],[0,0],'black')
        ax.plot([0,0],[0,rat],'black')
        ax.plot([0,0],[0,0],[0,rat],'black')
        ax.text(rat,0,0,'x',color='red')
        ax.text(0,rat,0,'y',color='red')
        ax.text(0,0,rat,'z',color='red')
        ax.view_init(elev=24, azim=25)
        fig.colorbar(cm.ScalarMappable(norm=plt.Normalize(vmin=peak-r_range,vmax=peak), cmap='jet'),shrink=0.5)