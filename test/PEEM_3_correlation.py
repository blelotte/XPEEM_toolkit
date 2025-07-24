# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 09:45:33 2024

@author: lelotte_b
"""

import os
import XPEEM

crdir=os.getcwd()
XPEEM.calculate_ppcaMap(source='nnmf',sample='dsA',n_PCA=4,originplot=False,method_whitening='mean')