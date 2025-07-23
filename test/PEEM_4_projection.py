# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 09:45:33 2024

@author: lelotte_b
"""

import os
import sys
sys.path.append('D:/Documents/a PSI/Data/Data analysis/spyder/Modules')
import XPEEM
import XPEEM_utils as utils



# Use the function
crdir=os.getcwd()
# XPEEM.calculate_ppcaMap(sample='Uncy',n_PCA=3,n_NNMA=4,n_clusters=4,originplot=False,method_whitening='rank')
# XPEEM.calculate_ppcaMap(sample='Cy10',n_PCA=4,originplot=False,method_whitening='rank')

XPEEM.calculate_ppcaMap(source='nnmf',sample='dsA',n_PCA=4,originplot=False,method_whitening='mean')

# oplt.openOriginAfterExecution()