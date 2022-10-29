#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 23:06:50 2022

@author: mimfar
"""
import unittest
from linear_array import LinearArray, db20, PI
import numpy as np 




import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
if current_path not in sys.path: sys.path.insert(0,current_path)


def AnalyticalArrayPattern(num_elem,element_spacing,scan_angle=0,theta=[],element_pattern=False):
    Psi = 2 * PI * element_spacing * (np.sin(np.radians(theta)) - np.sin(np.radians(scan_angle)))
    Psi[Psi == 0] = 1e-12
    return np.exp(1j * (num_elem - 1)) * np.sin(num_elem * Psi / 2) / np.sin(Psi / 2) / (num_elem) ** 0.5
    


class LinearArrayTest(unittest.TestCase):
    
    def test_calc_AF(self):
        num_elem = 10# number of elements
        element_spacing = 0.5 
        scan_angle = 30
        la = LinearArray(num_elem,element_spacing,scan_angle=scan_angle,element_pattern=False)
        AF = la.calc_AF.ravel()
        AF1 = AnalyticalArrayPattern(num_elem,element_spacing,scan_angle=scan_angle,theta=la.theta,element_pattern=False)
        self.assertEqual(db20(AF).all(),db20(AF1).all())
