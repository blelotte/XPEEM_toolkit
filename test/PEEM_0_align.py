# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:10:27 2023

@author: lelotte_b
"""

import XPEEM
import XPEEM_utils as utils
import os

def correct_distorsion():
    """
    Dataset B 
    
    Example 1: operando-XPEEM Co L-edge at OCP
    A simple example.
    _______________________________________________________________________________"""
    basedir=os.getcwd()
    sample = 'dsB'
    SampleDirectory = utils.path_join(basedir,sample,'Co_dsB',dt='d')
    
    # Calculation of transform
    # XPEEM.align_images(SampleDirectory, ['Co_OCP_i230818_057#001','Co_OCP_i230818_057#001'], sample, params="0;1000;1000;100;Coarse;0.01;1;0",rectify=False)
    # XPEEM.align_images(SampleDirectory, ['Co_OCP_i230818_057#001','Co_OCP_i230818_057#001'], sample, params="0;1000;1000;100;Coarse;0.01;1;0",rectify=True)
    
    """
    Dataset B 
    
    Example 2: operando-XPEEM O K-edge after 1st charge to 4.6V
    A very difficult example.
    - uses a transformation.
    - rectifies the transform afterward to decrease noise.
    _______________________________________________________________________________"""
    basedir=os.getcwd()
    sample = 'dsB'
    SampleDirectory = utils.path_join(basedir,sample,'O_dsB',dt='d')
    
    # Calculation of transform
    # XPEEM.align_images(SampleDirectory, ['4_O_Chrg_i230819_045','4_O_Chrg_i230819_045'], sample, params="0;1000;1000;1000;Coarse;0.01;0.1;2",rectify=False)
    
    # Smoothing in transform and linear alignement with SIFT.
    # XPEEM.align_images(SampleDirectory, ['4_O_Chrg_i230819_045','4_O_Chrg_i230819_045'], sample, params="0;1000;1000;1000;Coarse;0.01;0.1;2",rectify=True)


if __name__ == '__main__':
    """
    Distorsion correction
    _______________________________________________________________________________"""
    correct_distorsion()