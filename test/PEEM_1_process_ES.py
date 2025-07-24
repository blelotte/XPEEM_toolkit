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
    
    sample='dsA'
    shiftPlot=0
    ROI_list, label_list, mask_list_broad = XPEEM.load_masks(sample,loadAll_bool=True)
    
    """ Ni L-edge Uncycled """
    directory=utils.path_join(os.getcwd(),sample,"Ni_dsA")
    folder="1_Ni_undistrdd"
    
    # XPEEM.find_Eshift(directory,folder,mask_list_broad[ROI_list.index('NCM')],name='UncyNi')
    XPEEM.process_Estack(directory,folder,test=True,save_stack=True,mantis_name='test')
    # XPEEM.export_Estack(directory,folder,segm=[mask_list_broad,label_list,ROI_list],Originplot=False,shift=shiftPlot,samplelabel=sample,mantisfolder='_test',Eref_peak=853.1)
    # filename=XPEEM.prepare_Mantis(directory,mask=mask_list_broad[ROI_list.index('NCM')],ROI='NCM',filename='test_density.tif')
    # XPEEM.process_Mantis(directory, 3, "1,2", sample=sample,s=shiftPlot,Eselect='851.3,853.1',filename=filename)

    """ O L-edge Uncycled """
    directory=utils.path_join(os.getcwd(),sample,"O_dsA")
    folder="1_O_undistrdd"
    
    # XPEEM.process_Estack(directory,folder, test=True,save_stack=True,E_range=[520,550],mantis_name='test')
    # filename=XPEEM.prepare_Mantis(directory,mask=mask_list_broad[ROI_list.index('NCM')],ROI='NCM',filename='test_density.tif')
    # XPEEM.export_Estack(directory,folder,segm=[mask_list_broad,label_list,ROI_list],Originplot=False,shift=shiftPlot,samplelabel=sample,mantisfolder='_test',Eref_peak=532)
    # XPEEM.process_Mantis(directory, 5, "0,1,2,3,4",filename=filename, sample=sample,s=shiftPlot,Eselect='529.9,533.1,528.2,536.3,540.3')

if __name__ == '__main__':
    """
    Export
    _______________________________________________________________________________"""
    test()

    """
    Image comparison
    _______________________________________________________________________________"""
    # t0=time.time()
    # XPEEM.execute_image_comparisons(utils.path_join(os.getcwd(), 'comparisons_list.xlsx', dt='f'),'Uncy')
    # t1=time.time()
    # print(f'Runtime execute_image_comparisons: {t1-t0}s')
        
