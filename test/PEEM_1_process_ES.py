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

# Change the current working directory to the current files location
save_stack=True
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print('Current working directory: ' + os.getcwd())

def test():
    
    sample='dsA'
    shiftPlot=0
    ROI_list, label_list, mask_list_broad = XPEEM.load_masks(sample,loadAll_bool=True)
    
    """ Ni L-edge Uncycled """
    directory=utils.path_join(os.getcwd(),sample,"Ni_dsA")
    folder="1_Ni_undistrdd"
    
    # XPEEM.find_Eshift(directory,folder,mask_list_broad[ROI_list.index('NCM')],name='UncyNi')
    # XPEEM.process_Estack(directory,folder,test=True,save_stack=True,mantis_name='test')
    # XPEEM.export_Estack(directory,folder,segm=[mask_list_broad,label_list,ROI_list],Originplot=False,shift=shiftPlot,samplelabel=sample,mantisfolder='_test',Eref_peak=853.1)

    """ O L-edge Uncycled """
    directory=utils.path_join(os.getcwd(),sample,"O_dsA")
    folder="1_O_undistrdd"
    
    # XPEEM.process_Estack(directory,folder, test=True,save_stack=True,E_range=[520,550],mantis_name='test')
    # XPEEM.export_Estack(directory,folder,segm=[mask_list_broad,label_list,ROI_list],Originplot=False,shift=shiftPlot,samplelabel=sample,mantisfolder='_test',Eref_peak=532)

if __name__ == '__main__':
    test()
        
