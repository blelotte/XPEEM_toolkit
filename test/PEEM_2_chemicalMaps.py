# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 22:51:37 2023

@author: lelotte_b
"""

import sys
import os
import XPEEM
import XPEEM_utils as utils
import originpro as op

def origin_shutdown_exception_hook(exctype, value, traceback):
    '''Ensures Origin gets shut down if an uncaught exception'''
    op.exit()
    sys.__excepthook__(exctype, value, traceback)
if op and op.oext:
    sys.excepthook = origin_shutdown_exception_hook

#Commands
save_stack=True

# change the current working directory to the current files location
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print('Current working directory: ' + os.getcwd())

def test():

    XPEEM.calculate_chemicalMap('dsA')
    
    sample='dsA'
    shiftPlot=0
    ROI_list, label_list, mask_list_broad = XPEEM.load_masks(sample,loadAll_bool=True)
    
    """ Ni L-edge """
    directory=utils.path_join(os.getcwd(),sample,"Ni_dsA")
    
    filename=XPEEM.prepare_MLMap(directory,mask=mask_list_broad[ROI_list.index('NCM')],ROI='NCM',filename='test_density.tif')
    XPEEM.calculate_MLMap(directory, 3, "1,2", sample=sample,s=shiftPlot,Eselect='851.3,853.1',filename=filename)

    """ O L-edge """
    directory=utils.path_join(os.getcwd(),sample,"O_dsA")
    
    filename=XPEEM.prepare_MLMap(directory,mask=mask_list_broad[ROI_list.index('NCM')],ROI='NCM',filename='test_density.tif')
    XPEEM.calculate_MLMap(directory, 5, "0,1,2,3,4",filename=filename, sample=sample,s=shiftPlot,Eselect='529.9,533.1,528.2,536.3,540.3')

if __name__ == '__main__':
    """
    Export
    _______________________________________________________________________________"""
    test()

        
