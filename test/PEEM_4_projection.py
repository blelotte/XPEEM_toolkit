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


crdir=os.getcwd()
XPEEM.calculate_ppcaMap(source='nnmf',sample='dsA',n_PCA=4,originplot=False,method_whitening='mean')