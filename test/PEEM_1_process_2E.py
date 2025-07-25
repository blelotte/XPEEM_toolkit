# -*- coding: utf-8 -*-
"""
Script which runs 2 energies micrograph comparison 

It creates the "DIV" folders, after alignement and distorsion correction.


Created on Fri Sep  8 16:24:10 2023

@author: lelotte_b
"""

import os

import XPEEM
import XPEEM_utils as utils

def normalisation_2i_std():

    path=os.getcwd()
    
    
    """ E=851_3 normalised """
    directory=utils.path_join(path,'dsA/Ni_dsA/2_energies')
    file="i211025_0"
    IDs=['09','10','11_DIV']    
    
    # XPEEM.process_2E(directory,file,IDs,(10,2,245,318),save_stack=True,bounds=[0,10])

    
    """ E=853_1 normalised """
    directory=utils.path_join(path,'dsA/Ni_dsA/2_energies')
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211025_0"
    IDs=['12','13','14_DIV']    
    
    # XPEEM.process_2E(directory,file,IDs,(10,2,245,318),save_stack=True,bounds=[0,10])


def normalisation_2i_chemcomp():
    
    path=os.getcwd()
    
    """ Ni red """
    directory=utils.path_join(path,'dsA/Ni_dsA/2_energies')

    file="i211025_0"
    IDs=['11_DIV','14_DIV','Ni_red_DIV']
    
    XPEEM.process_2E(directory,file,IDs,(10,2,245,318),save_stack=True,suffix='filtered')    
    
    """ Ni ox """
    directory=utils.path_join(path,'dsA/Ni_dsA/2_energies')

    file="i211025_0"
    IDs=['14_DIV','11_DIV','Ni_ox_DIV']
    
    # XPEEM.process_2E(directory,file,IDs,(10,2,245,318),save_stack=True,suffix='filtered')    


if __name__ == '__main__':
    normalisation_2i_std()
    normalisation_2i_chemcomp()