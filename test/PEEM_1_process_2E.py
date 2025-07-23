# -*- coding: utf-8 -*-
"""
Script which runs all the 2 energies micrograph comparison 

It creates the "DIV" folders, after alignement and distorsion correction.


Created on Fri Sep  8 16:24:10 2023

@author: lelotte_b
"""
import numpy as np
from PIL import Image
import math
import ElectroChem as EC

import time
import matplotlib.pyplot as plt
import os
import pandas as pd



def normalisation_2i_std():
    """
    10 Cycles
    _______________________________________________________________________________"""
    path=r'D:\Documents\a PSI\Data\Data analysis\spyder\2107_Progress_work\0410_SIM beamline 1\PEEM_py\10_cycles'
    
    """ E=851_5 normalised """
    directory=path+r'\Ni_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_0"
    IDs=['57','58','59_DIV']    
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10],scale=[5000,600])
    
    """ E=853_1 normalised """
    directory=path+r'\Ni_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_0"
    IDs=['81','82','83_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10],scale=[5000,600])

    
    """ E=777 normalised """
    directory=path+r'\Co_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_0"
    IDs=['64','65','66_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10])
    
    """ E=777.8 normalised """
    directory=path+r'\Co_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_0"
    IDs=['67','68','69_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10])
    
    """ E=642_2 normalised """
    directory=path+r'\Mn_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_0"
    IDs=['72','73','74_DIV']    
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10])
    
    """ E=639_8 normalised """
    directory=path+r'\Mn_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_0"
    IDs=['75','76','77_DIV']    
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10])
    
    """ E=530 normalised """
    directory=path+r'\O_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_0"
    IDs=['88','89','90_DIV']    
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10],std_bkg=False)
    
    """ E=533.1 normalised """
    directory=path+r'\O_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_0"
    IDs=['91','92','93_DIV']    
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10],std_bkg=False)
    
    """ E=284 normalised """
    directory=path+r'\C_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_0"
    IDs=['95','96','97_DIV']    

    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10],std_bkg=False)
    
    """ E=288.2 normalised """
    directory=path+r'\C_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_"
    IDs=['098','099','100_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10],std_bkg=False)

    
    """ E=289_9 normalised """
    directory=path+r'\C_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_"
    IDs=['104','105','106_DIV']

    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,5],std_bkg=False)
    
    """ E=289_9 normalised """
    directory=path+r'\C_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_"
    IDs=['122','123','124_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,5],std_bkg=False)
    
    """ E=284_9 normalised """
    directory=path+r'\C_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_"
    IDs=['128','129','130_DIV']

    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,5],std_bkg=False)
    
    """ E=288_2 normalised """
    directory=path+r'\C_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    

    file="i211020_"
    IDs=['131','132','133_DIV']

    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,5],std_bkg=False)
    
    """ E=289_9 vs  normalised """
    directory=path+r'\C_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_"
    IDs=['132','123','PEC_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,5],std_bkg=False)

    
    """ E=138 normalised """
    directory=path+r'\P_S_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_"
    IDs=['111','112','113_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,std_bkg=False)
    
    
    """ E=165.2 normalised """
    directory=path+r'\P_S_10_cycles\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211020_"
    IDs=['115','116','117_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,std_bkg=False)
    
    """ E=138 normalised """
    directory=path+r'\P_S_10_cycles\2_energies\\'
    directory=directory.replace('\\','/')
    directory=directory.replace('//','/')
    
    file="i211020_"
    IDs=['118','119','120_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,std_bkg=False)
    
    """ E=138 normalised_2 """
    # directory=path+r'\P_S_10_cycles\2_energies\\'
    # directory=directory.replace('\\', '/')
    # directory=directory.replace('//', '/')
    
    file="i211020_"
    IDs=['119','112','120-2_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,std_bkg=False)
    
    """
    Uncycled
    _______________________________________________________________________________"""
    path=r'D:\Documents\a PSI\Data\Data analysis\spyder\2107_Progress_work\0410_SIM beamline 1\PEEM_py\Uncy'
    
    
    """ E=853_1 normalised """
    directory=path+r'\Ni_Uncy\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211025_0"
    IDs=['09','10','11_DIV']    
    
    EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10])

    
    """ E=853_1 normalised """
    directory=path+r'\Ni_Uncy\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211025_0"
    IDs=['12','13','14_DIV']    
    
    EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10])
    
    """ E=777 normalised """
    directory=EC.path_join(path,'Co_uncycled/2_energies')
    
    file="i211025_0"
    IDs=['16','17','18_DIV']        
        
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10])
    
    """ E=778_8 normalised """
    directory=EC.path_join(path,'Co_uncycled/2_energies')
    
    file="i211025_0"
    IDs=['22','23','24_DIV']        
        
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10])
    
    """ E=639_8 normalised """
    directory=EC.path_join(path,'Mn_uncycled/2_energies')
    
    file="i211025_0"
    IDs=['27','28','29_DIV']        
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,bounds=[0,10])
    
    """ E=642_2 normalised """
    directory=path+r'\Mn_Uncycled\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211025_0"
    IDs=['30','31','32_DIV']
        
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True)
    
    """ E=529_9 normalised """
    directory=EC.path_rectify(path+r'\O_Uncycled\2_energies\\','d',sort='abs_path')
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211025_0"
    IDs=['34','35','36_DIV']
      
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,std_bkg=False)
    
    """ E=539_5 normalised """
    directory=EC.path_rectify(path+r'\O_Uncycled\2_energies\\','d',sort='abs_path')
    
    file="i211025_0"
    IDs=['37','38','39_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,std_bkg=False)
    
    """ E=533.1 normalised """
    directory=EC.path_rectify(path+r'\O_Uncycled\2_energies\\','d',sort='abs_path')
    
    file="i211025_0"
    IDs=['40','41','42_DIV']
        
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,std_bkg=False)
    
    """ E=284 normalised """
    directory=EC.path_rectify(path+r'\C_Uncycled\2_energies\\','d',sort='abs_path')
    
    file="i211025_0"
    IDs=['47','48','49_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True, bounds=[0,20],std_bkg=False)
    
    """ E=288_2 normalised """
    directory=EC.path_rectify(path+r'\C_Uncycled\2_energies\\','d',sort='abs_path')
    
    file="i211025_0"
    IDs=['50','51','52_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True, bounds=[0,10],std_bkg=False)
        
    """ E=289_9 normalised """
    directory=path+r'\C_Uncycled\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211025_0"
    file="i211025_0"
    IDs=['53','54','55_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True, bounds=[0,20],std_bkg=False)

    """ E=135eV normalised """
    directory=path+r'\P_S_Uncycled\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211025_0"
    IDs=['62','61','63_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True, bounds=[0,10],std_bkg=False)

    
    """ E=137.6 normalised """
    directory=path+r'\P_S_Uncycled\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211025_0"
    IDs=['65','64','66_DIV']

    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True, bounds=[0,10],std_bkg=False)

    """ E=170.2 normalised """
    directory=path+r'\P_S_Uncycled\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211025_0"
    IDs=['68','67','69_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True, bounds=[0,10],std_bkg=False)
    
    """ E=168.7 normalised """
    directory=path+r'\P_S_Uncycled\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211025_0"
    IDs=['71','70','72_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True, bounds=[0,10],std_bkg=False)
    
    # """ E=227.1 normalised """
    directory=path+r'\Cl_Uncycled\2_energies\\'
    directory=directory.replace('\\', '/')
    directory=directory.replace('//', '/')
    
    file="i211025_0"
    IDs=['77','78','79_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,514,514),save_stack=True,bounds=[0,5],std_bkg=False)

def normalisation_2i_chemcomp():
    """
    10 Cycles
    _______________________________________________________________________________"""
    path=r'D:\Documents\a PSI\Data\Data analysis\spyder\2107_Progress_work\0410_SIM beamline 1\PEEM_py\10_cycles'
    
    """ Ni red """
    directory=EC.path_join(path,'Ni_10_cycles/2_energies/')
    
    
    file="i211020_0"
    IDs=['83_DIV','59_DIV','Ni_red_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered')

    """ Ni ox """
    directory=EC.path_join(path,'Ni_10_cycles/2_energies/')
    
    
    file="i211020_0"
    IDs=['59_DIV','83_DIV','Ni_ox_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered')
        
    """ Co red """
    directory=EC.path_join(path,'Co_10_cycles/2_energies')
    
    file="i211020_0"
    IDs=['66_DIV','69_DIV','Co_red_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered')
        
    """ Co ox """
    directory=EC.path_join(path,'Co_10_cycles/2_energies')
    
    file="i211020_0"
    IDs=['69_DIV','66_DIV','Co_ox_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered', bounds=[0,20])

    
    """ Mn red """
    directory=EC.path_join(path,'Mn_10_cycles/2_energies')
    
    file="i211020_0"
    IDs=['77_DIV','74_DIV','Mn_red_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered')

    """ Mn ox """
    directory=EC.path_join(path,'Mn_10_cycles/2_energies')
    
    file="i211020_0"
    IDs=['74_DIV','77_DIV','Mn_ox_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered')
    
    """ SO4 LayO """
    directory=EC.path_join(path,'O_10_cycles/2_energies')
    
    file="i211020_0"
    IDs=['93_DIV','90_DIV','SuLa_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered', bounds=[0,30])

    
    """ SO4 LayO (LaSu) """
    directory=EC.path_join(path,'O_10_cycles/2_energies')
    
    file="i211020_0"
    IDs=['90_DIV','93_DIV','LaSu_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered',bounds=[0,5])

    """ CO3 ratio """
    directory=EC.path_join(path,'C_10_cycles/2_energies')
    
    file="i211020_"
    IDs=['124_DIV','100_DIV','Ca1_DIV']
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered',bounds=[0,50])

    """ CO3 ratio """
    directory=EC.path_join(path,'C_10_cycles/2_energies')
    
    file="i211020_"
    IDs=['133_DIV','097_DIV','Ca3_DIV']
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered',bounds=[0,50])


    """ CO3 ratio """
    directory=EC.path_join(path,'C_10_cycles/2_energies')
    
    file="i211020_"
    IDs=['133_DIV','100_DIV','Ca2_DIV']
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered',bounds=[0,50])

    """ CO3 ratio
    Note: I've spend some times working on this one. 
    > It is better to show the "Phi" contrast, which is closest to the spectral trends, than the peak ratio. 
    The peak ratio induces plenty and does not correct for the intensity. 
    One reason is that the "reference" is measured on the post-edge. 
    Therefore the position of the background after normalisation is unknown. """
    directory=EC.path_join(path,'C_10_cycles/2_energies')
    
    file="i211020_"
    IDs=['106_DIV','097_DIV','CO3_DIV']
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered',bounds=[0,5])

    """
    Uncycled
    _______________________________________________________________________________"""
    path=r'D:\Documents\a PSI\Data\Data analysis\spyder\2107_Progress_work\0410_SIM beamline 1\PEEM_py\Uncy'
    
    
    """ Ni red """
    directory=EC.path_join(path,'Ni_Uncy/2_energies')

    file="i211025_0"
    IDs=['11_DIV','14_DIV','Ni_red_DIV']
    
    EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered')    
    
    """ Ni ox """
    directory=EC.path_join(path,'Ni_Uncycled/2_energies')

    file="i211025_0"
    IDs=['14_DIV','11_DIV','Ni_ox_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered')    

    
    """ Co red """
    directory=EC.path_join(path,'Co_Uncycled/2_energies')
    
    file="i211025_0"
    IDs=['18_DIV','24_DIV','Co_red_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered')    
        
    """ Co ox """
    directory=EC.path_join(path,'Co_Uncycled/2_energies')
    
    file="i211025_0"
    IDs=['24_DIV','18_DIV','Co_ox_DIV']

    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered',bounds=[0,30])    


    """ Mn red """
    directory=EC.path_join(path,'Mn_Uncycled/2_energies')
    
    file="i211025_0"
    IDs=['29_DIV','32_DIV','Mn_red_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered',bounds=[0,2])    
    
    """ Mn ox """
    directory=EC.path_join(path,'Mn_Uncycled/2_energies')
    
    file="i211025_0"
    IDs=['32_DIV','29_DIV','Mn_ox_DIV']
    
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered',bounds=[0,5])    
    
    """ SO4 LayO (SuLa) """
    directory=EC.path_join(path,'O_Uncycled/2_energies')
    
    file="i211025_0"
    IDs=['42_DIV','36_DIV','SuLa_DIV']

    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered',bounds=[0,10])    

    """ LayO SO4 (LaSu) """
    directory=EC.path_join(path,'O_Uncycled/2_energies')
    
    file="i211025_0"
    IDs=['36_DIV','42_DIV','LaSu_DIV']

    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered',bounds=[0,10])    
        
    """ CO3 ratio """
    directory=EC.path_join(path,'C_Uncycled/2_energies')
    
    file="i211025_0"
    IDs=['53','54','CO3_DIV']
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='undistrdd',bounds=[0,5],std_bkg=False)
    
    """ CO3 ratio """
    directory=EC.path_join(path,'C_Uncycled/2_energies')
    
    file="i211025_0"
    IDs=['55_DIV','49_DIV','CO3_DIV']
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='filtered',shift=True)
    
    """ IntP peak """
    directory=EC.path_join(path,'P_S_Uncycled/2_energies')

    file="i211025_0"
    IDs=['62','64','IntP_DIV']
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='undistrdd',bounds=[-5,50])

    
    """ seP """
    directory=EC.path_join(path,'P_S_Uncycled/2_energies')

    file="i211025_0"
    IDs=['65','61','seP_DIV']
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='undistrdd')

    
    """ E=137.6 normalised """
    directory=EC.path_join(path,'P_S_Uncycled/2_energies')

    file="i211025_0"
    IDs=['61','65','iP2_DIV']
    # EC.PEEM_2i_averaging(directory,file,IDs,(10,2,490,490),save_stack=True,suffix='undistrdd',bounds=[-5,0])
    

if __name__ == '__main__':
    normalisation_2i_std()
    normalisation_2i_chemcomp()