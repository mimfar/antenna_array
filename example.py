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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from antenna_array import calc_AF, calc_AF_, plot_pattern, polar_pattern ,calc_peak_sll_hpbw, db20

import random
from scipy.signal.windows import get_window , taylor, chebwin

#%%
# antenna_array module
# you can import the basic functions for antenna array calculation 
# - **calc_AF**: calculate array factor (AF) for a regulare equispaced linear antenna array
# - **calc_AF_**: calculate array factor (AF) for a general linear antenna array where elements are not equi-spaced
# - **polar_pattern** and **plot_pattern**: draw antenna array pattern in cart or polar coordinates

 
# This script show how to use the methods for antenna_array module to calculate the pattern 
# and parameters for a uniform linear array of 8 elements with scan angle = 30

num_elem = 16 # number of elements
element_spacing = 0.5
scan_angle = 30
theta_deg,AF_linear = calc_AF(num_elem,element_spacing,scan_angle)
G = db20(AF_linear)
peak, theta_peak, SLL, HPBW = calc_peak_sll_hpbw(G, theta_deg) 
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'
      .format(peak, theta_peak, SLL, HPBW))
polar_pattern(theta_deg,G);

#%%
# An array of num_elem element with random element spacing choosen from element spacing list
num_elem = 16
dx = sorted(random.choices([0.5,0.75,1.25,2],k=int(num_elem/2),weights = [1, .75, 0.5, 0.5]))
element_spacing = np.hstack((np.flip(dx),dx[1:])) 
X = np.insert(np.cumsum(element_spacing),0,0)
X = X - np.mean(X)
P = -2 * np.pi * X * np.sin(np.radians(scan_angle))
I = np.ones(X.shape)
Nt = 181 # length of theta vectoe
theta_deg,dtheta_deg = np.linspace(-90,90,Nt,retstep = True)
theta_deg = theta_deg.reshape(Nt,1)
AF_linear = calc_AF_(X,I,P,theta_deg)
G = db20(AF_linear)
peak, theta_peak, SLL, HPBW = calc_peak_sll_hpbw(G, theta_deg) 
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.
      format(peak, theta_peak, SLL, HPBW))
fig, ax = plot_pattern(theta_deg,G,title='Linear Array with Nonuniform Spacing');
ax.set_xticks(np.linspace(-90,90,7));
## plotting the array geometry in the inset
ax_inset = fig.add_axes([0.2, 0.15, 0.6, 0.1])
ax_inset.plot(X,np.zeros(X.shape),'.b')
ax_inset.patch.set_alpha(0.85)
ax_inset.set_yticklabels('');
ax_inset.tick_params(axis="x",direction="in", pad=-10)
ax_inset.set_yticklabels('');
ax_inset.set_xlabel('x(wavelength)');
plt.show()
#%%
''' LinearArray Class

The class provides methods to analyze a linear antenna array. The class provides more felxibility to deal with both methods,
 data and visulaization of an antenna array
- **calc_AF**: calculate array factor (AF) for a regulare equispaced linear antenna array
- **calc_AF_**: calculate array factor (AF) for a general linear antenna array where elements are not equi-spaced
- **polar_pattern** and **plot_pattern**: draw antenna array pattern in cart or polar coordinates
- **calc_peak_sll_hpbw** calculated Peak *Gain*, Side Lobe Level (*SLL*) and Half Power BeamWidth (*HPBW*) of the array radiation pattern''' 
from linear_array import LinearArray

num_elem = 8 # number of array elements
element_spacing = 0.5 # in wavelength
la = LinearArray(num_elem,element_spacing,scan_angle=30,element_pattern=True) #define the linear array geometry and excitation
la.calc_AF # calculate Array Factor (Array radiation pattern)
fig,ax = la.plot_pattern(annotate=True,xlab=r'$\theta$',ylab='Gain(dB)')
params = la.calc_peak_sll_hpbw()
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(*params))
fig.set_size_inches((6,4)),ax.set_xticks(range(-90,120,30));
#%%
# calc_envelope method produces the scanned beam in the provided range where they can be plotted using
# polar_envelope or plot_envelope methods
la.calc_envelope(theta1=0,theta2=45,delta_theta=5)
fig,ax = la.polar_envelope();
ax.set_title('Scanned Patterns and the Envelope ')
#%% Linear Array Class Example nonuniform spacing sparse array with low side lobe level
# The nonunifrom spacing removes the grating lobe
num_elem = 32
dx = sorted(random.choices([0.5,0.75,1.25,2],k=int(num_elem/2),weights = [1, .75, 0.5, 0.5]))
element_spacing = np.hstack((np.flip(dx),dx[1:])) 
scan_angle = 30
la = LinearArray(num_elem,element_spacing,scan_angle=scan_angle)
la.calc_AF

fig = plt.figure(figsize=(9,6))
_,ax = la.plot_pattern(fig=fig,marker='-',xlim=(-90,90))
params = la.calc_peak_sll_hpbw()
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(*params))

element_spacing_uniform = np.mean(element_spacing)
la_uniform = LinearArray(num_elem,element_spacing_uniform,scan_angle=scan_angle)
la_uniform.calc_AF
la_uniform.plot_pattern(fig=fig,marker='--',xlim=(-90,90),ylim=(-20,25))
plt.legend(['Nonuniform Spacing','Uniform Spacing'],loc=3)
plt.title('Nonuniform Spacing removes the grating lobe')
params = la_uniform.calc_peak_sll_hpbw()
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(*params))
## plotting the array geometry in the inset
ax_inset = fig.add_axes([0.15, 0.85, 0.7, 0.1])
ax_inset.plot(la.X,np.ones(la.X.shape),'.b')
ax_inset.plot(la_uniform.X,-np.ones(la_uniform.X.shape),'xr')
ax_inset.set_ylim(-2,2)
ax_inset.tick_params(axis="x",direction="in", pad=-10)
ax_inset.set_yticklabels('');
ax_inset.set_xlabel('x(wavelength)');
#%% Linear array example construct from element position
X = 4 * np.random.randn(num_elem)
la = LinearArray.from_element_position(X)
la.calc_AF
fig = plt.figure(figsize=(8,6))
_,ax = la.plot_pattern(fig=fig,marker='-',xlim=(-90,90),ylim=(-20,20))
params = la.calc_peak_sll_hpbw()
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'.format(*params))
## plotting the array geometry in the inset
ax_inset = fig.add_axes([0.2, 0.85, 0.7, 0.1])
ax_inset.plot(la.X,np.zeros(la.X.shape),'.b')
ax_inset.patch.set_alpha(0.85)
ax_inset.set_yticklabels('');
ax_inset.tick_params(axis="x",direction="in", pad=-10)
ax_inset.set_yticklabels('');
ax_inset.set_xlabel('x(wavelength)');

#%%
'''# Amplitude Tapering
we can improve SLL (Side Lobe Level) by using amplitude tapering. We can use the antenna_array 
module **AF_calc_** method and directly pass the amplitude taper to the method.'''

# An array of num_elem element with amplitude tapering / windowing to reduce side lobe level

from antenna_array import calc_AF_, plot_pattern ,calc_peak_sll_hpbw, db20
import pandas as pd
from scipy.signal.windows import get_window , taylor, chebwin

num_elem = 16
element_spacing = 0.5
scan_angle = 0
X = np.linspace(0,num_elem-1,num_elem) * element_spacing
P = -2 * np.pi * X * np.sin(np.radians(scan_angle))

Nt = 721 # length of theta vectoe
theta_deg,dtheta_deg = np.linspace(-90,90,Nt,retstep = True)
theta_deg = theta_deg.reshape(Nt,1)
fig, ax = plt.subplots(2,1,figsize=(10,10))
# fig1, ax1 = plt.subplots(figsize=(9,4))

# based on window type
window_list = ['boxcar','hamming','blackman','blackmanharris']
df = pd.DataFrame(columns = ['Peak','SLL','HPBW'])
for window in window_list:
    I = get_window(window, num_elem)

    plt.sca(ax[0]) 
    plt.plot(I,'-o')

    AF_linear = calc_AF_(X,I,P,theta_deg,element_pattern=False)
    G = db20(AF_linear)
    plot_pattern(theta_deg,G-10*np.log10(num_elem),ylim=(-110,0),ax=ax[1],xlab=r'$\theta$',ylab='normalized Gain(dB)')

    peak, theta_peak, SLL, HPBW = calc_peak_sll_hpbw(G, theta_deg)
    df.loc[window] = [peak, SLL, HPBW]

plt.legend(window_list)
plt.sca(ax[1]) 
plt.legend(window_list)
ax[0].set_xlabel('element position')
ax[0].set_title('Amplitude Tapering')
ax[0].set_ylabel('Magnitude (Linear)')
ax[0].set_ylim(0,1.2)
ax[0].set_xlim(0,num_elem-1)
ax[0].grid()
print(df.round(1))

#%%
'''Apmlitude Tapering with SLL target
We can set the amplitude tapering to achieve a certain SLL value using some 
windowing functions that provide the taper profile based on the desired SLL. 
Below we have used **Taylor** and **Chebychef** window methods to realized SLL values from 20 to 100dB.'''
#%% Linear Array Class with amplitude tapering Example
# we can pass the window type or SLL target directly to our la object
# list of available winow 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html

from linear_array import LinearArray
num_elem = 16# number of elements
element_spacing = 0.5 
scan_angle = 0
theta = np.linspace(-90,90,721)
la_hann = LinearArray(num_elem,element_spacing,theta=theta,scan_angle=scan_angle,window='hann')
la_hann.calc_AF
_,ax = la_hann.plot_pattern(marker='-',xlim=(-90,90),ylim=(-60,15))
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'
      .format(*la_hann.calc_peak_sll_hpbw()))
la_SLL_55 = LinearArray(num_elem,element_spacing,theta=theta,scan_angle=scan_angle,SLL=55)
la_SLL_55.calc_AF
la_SLL_55.plot_pattern(ax=ax,marker='--',xlim=(-90,90),ylim=(-60,15),title='Array Gain')
print('Peak = {:1.1f}dBi, theta_peak = {:1.1f}deg, SLL = {:1.1f}dB, HPBW = {:1.1f}deg'
      .format(*la_SLL_55.calc_peak_sll_hpbw()))

ax.legend(['Hann','SLL=55dB target']);
ax.set_xticks(np.linspace(-90,90,7));
#%%
'''# Planar Array
The planar array class follows the same methods of the Linear Array class.

In particulare the class provides more felxibility to define regulre and non-regulare planar array shapes and visulaize 
the radiation patterns with a bundle of methods.
### Array input:
an _array_shape_ input variable determines the planar array shape <br>
Options:
-  array_shape = **['rect',(4,8),(0.75,0.5)]**: Rectanglure shape, with 4 row and 8 columns, spacing 0.75 and 0.5 wavelength, respectively.
- array_shape = **['tri',(4,8),(0.7,0.5)]**: Trianglure shape, with 4 row and 8 columns, spacing 0.7 and 0.5 wavelength, respectively.
- array_shape = **['circ',(0.5,1.1,1.655),(5,10,17)]**: circulare shape, radius 0.5 wl with 5 elements spaced equally on 
        the circle, radius 1.1 with 10 elements spaced equally around the circle and so on. 
- **calc_AF**: calculate 2D array factor (AF) for a planar array. 

'''

# Example of four planar array with same # of elements occupying the same area
from planar_array import PlanarArray

fig,ax = plt.subplots(2,2,figsize=(9,9),sharex=True);

num_elem = (4,8)# number of row and col
element_spacing = (0.75,0.5)

array_shape = ['rect',num_elem,element_spacing]
pa_rect = PlanarArray(array_shape)
pa_rect.plot_array(fig=fig,ax=ax[0,0]);


array_shape = ['tri',num_elem,element_spacing]
pa_tri = PlanarArray(array_shape)
pa_tri.plot_array(fig=fig,ax=ax[0,1]);


num_elem_circ = [5,10,17] 
radius = [0.5,1.1,1.65] 
array_shape = ['circ',num_elem_circ,radius]
pa_circ = PlanarArray(array_shape)
pa_circ.plot_array(fig=fig,ax=ax[1,0]);


X = pa_rect.X + 0.05 * np.random.randn(32)
Y = pa_rect.Y + 0.075 * np.random.randn(32)
array_shape = ['other',X,Y]
pa_rand = PlanarArray(array_shape)
pa_rand.plot_array(fig=fig,ax=ax[1,1]);
ax[0,1].set_aspect('equal')
ax[1,1].set_aspect('equal')
ax[1,0].set_aspect('equal')
ax[0,0].set_aspect('equal')

# Calculating Max Directivity
aperture_rect = ((num_elem[0]-1) * element_spacing[0]  + 0.5) * ((num_elem[1]-1) * element_spacing[1] + 0.5)
Directivity_rect = 10 * np.log10(4 * np.pi * aperture_rect)
aperture_circ = np.pi * (radius[-1]+0.25)**2 
Directivity_circ = 10 * np.log10(4 * np.pi * aperture_circ)
print(f'Directivity in dB rect:{Directivity_rect:1.1f}, circ:{Directivity_circ:1.1f}')

#%%

# 3D polar plot, note the shape of side lobes for the circular array
fig,ax = plt.subplots(2,2,figsize=(9,7),subplot_kw={'projection':'3d'});

for idx,pa in enumerate([pa_rect,pa_tri,pa_circ,pa_rand]):
    pa.calc_AF
    pa.polar3D(g_range=30,fig=fig,ax=ax[int(idx/2),idx % 2],title=pa.shape)

#%%
# scan performance of the arrays in azimuth and elevation, Note grating lobes showing up in elevation scan
fig,ax = plt.subplots(3,1,figsize=(8,9),sharex=True);
for idx,pa in enumerate([pa_rect,pa_tri,pa_circ,pa_rand]):
    pa.scan_angle = (0,0)
    pa.calc_AF
    theta_deg,G = pa.pattern_cut(pa.scan_angle[1])
    pa.plot_pattern(fig=fig,ax=ax[0],xlim=[-90,90],ylim=[-5,25],xlab='',ylab='dB',title='Broadside')
    plt.legend(['rect','tri','circ','rand']);
    
    pa.scan_angle = (30,0)
    pa.calc_AF
    theta_deg,G = pa.pattern_cut(pa.scan_angle[1])
    pa.plot_pattern(fig=fig,ax=ax[1],xlim=[-90,90],ylim=[-5,25],xlab='',ylab='dB',title='az scan = 30')
    plt.legend(['rect','tri','circ','rand']);

    pa.scan_angle = (30,90)
    pa.calc_AF
    theta_deg,G = pa.pattern_cut(pa.scan_angle[1])
    pa.plot_pattern(fig=fig,ax=ax[2],xlim=[-90,90],ylim=[-5,25],xlab=r'$\theta$',ylab='dB',title='el scan = 30')
    plt.legend(['rect','tri','circ','rand']);
    
#%%
# planar array, step by step visualization

from planar_array import PlanarArray
num_elem = (6,12)# number of row and col
element_spacing = (0.75,0.5) 
scan_angle = (30,0)
array_shape = ['tri',num_elem,element_spacing]
pa = PlanarArray(array_shape,scan_angle=scan_angle)
fig,ax = plt.subplots(1,2,figsize=(10,3))
pa.plot_array(fig=fig,ax=ax[0])
pa.calc_AF
theta_deg,G = pa.pattern_cut(scan_angle[1])
pa.plot_pattern(fig=fig,ax=ax[1],xlim=[-90,90],xlab='theta',ylab='dB',title='Radiation Pattern')
pa.calc_peak_sll_hpbw();
pa.pattern_params

#%% Contour plot
fig1,ax1 = pa.pattern_contour(g_range=30,tlim=[0,90],plim=[-180,180],title='Gain(dBi)');
ax1.set_xticks(np.linspace(-180,180,5));
ax1.set_yticks(np.linspace(0,90,7));
fig1.set_size_inches((9,5))
#%%
pa.polarsurf(g_range=30);
#%%
pa.polarsphere()

#%% hexagone shape array

Ls = 3 # radius of the Circumcircle
N = 7 # num of the elements along the radius  
X = np.linspace(0,Ls,N)
Y = np.zeros(X.shape)

dx = Ls / (N - 1)
X1 =  X.copy()
for n in range(N-1):
    X1 = X1[:-1] +   dx / 2
    X = np.hstack((X,X1))
    Y = np.hstack((Y, dx * np.sqrt(3) / 2 * (n+1) * np.ones(X1.shape)))


X_all = np.hstack((X[0],X[N:]))
Y_all = np.hstack((Y[0],Y[N:]))
for n in range(1,6): 
    Xr = X[N:] * np.cos(n * np.pi / 3) - Y[N:] * np.sin(n * np.pi / 3)
    Yr = X[N:] * np.sin(n * np.pi / 3) + Y[N:]* np.cos(n * np.pi / 3)
    # plt.plot(Xr,Yr,'o')
    X_all = np.hstack((X_all,Xr))
    Y_all = np.hstack((Y_all,Yr))
plt.figure()
plt.plot(X_all,Y_all,'xr');
plt.axis('equal');

#%%
array_shape_Hex = ['other',X_all,Y_all]
scan_angle = (0,0)
pa_Hex = PlanarArray(array_shape_Hex,scan_angle=scan_angle)
pa_Hex.calc_AF

fig,ax = pa_Hex.plot_pattern(xlim=[-90,90],xlab=r'$\theta$')
fig.set_size_inches((6,4))

fig1,ax1 = pa_Hex.polar3D(g_range=30,title='Gain(dBi)')

# pa_Hex.polarsurf(g_range=30);
pa_Hex.polarsurf(g_range=30);
pa_Hex.calc_peak_sll_hpbw()
pa_Hex.pattern_params


#%% circular array
num_elem = [7,8]# number of row and col
radius = [1,2] 
scan_angle = (0,0)
array_shape = ['circ',num_elem,radius]
pa = PlanarArray(array_shape,scan_angle=scan_angle)
pa.plot_array()
pa.calc_AF
theta_deg,G = pa.pattern_cut(scan_angle[1])
plt.figure()
pa.plot_pattern(xlim=[-90,90],xlab='theta')
pa.polar_pattern(rlim=[-10,20])
pa.calc_peak_sll_hpbw()
print(pa.pattern_params)
pa.polarsurf(g_range=20);

#%%
x = np.linspace(0,6,10)
y = np.linspace(0,6,10)
[X,Y] = np.meshgrid(x,y)
delta_X = np.random.normal(0,.2,X.shape)
delta_Y = np.random.normal(0,.2,Y.shape)
X1 = X + delta_X
Y1 = Y + delta_Y
X1 = np.reshape(X1,(1,-1))
Y1 = np.reshape(Y1,(1,-1))

X = np.reshape(X,(1,-1))
Y = np.reshape(Y,(1,-1))

array_shape_rand = ['other',X1,Y1]
scan_angle = (45,0)
pa_rand = PlanarArray(array_shape_rand,scan_angle=scan_angle)
fig,ax = pa_rand.plot_array()

array_shape_rect = ['other',X,Y]
pa_rect = PlanarArray(array_shape_rect,scan_angle=scan_angle)
pa_rect.plot_array(fig=fig,ax=ax,colormarker='xr')

pa_rand.calc_AF
# theta_deg,G = pa_rand.pattern_cut(scan_angle[1])
plt.figure()
fig,ax = pa_rand.plot_pattern(xlim=[-90,90],xlab='theta')
pa_rect.calc_AF
theta_deg,G = pa_rect.pattern_cut(scan_angle[1])
pa_rect.plot_pattern(xlim=[-90,90],xlab='theta',fig=fig)


fig, ax = pa_rand.polar_pattern(rlim=[-30,30])
pa_rect.polar_pattern(rlim=[-30,30],fig=fig)
pa_rand.calc_peak_sll_hpbw()
pa_rect.calc_peak_sll_hpbw()

# pa_rand.pattern_params
# # pa.polarsurf(g_range=30);
# # pa.calc_peak_sll_hpbw_calc()
# pa_rand.pattern_params
fig1,ax1 = pa_rand.polar3D(g_range=20,title='Gain(dBi)')


