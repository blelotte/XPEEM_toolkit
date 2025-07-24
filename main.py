# -*- coding: utf-8 -*-
"""
The main iterates through the example test script.

Created on Tue Sep 26 20:57:25 2023

@author: lelotte_b
"""

import os
import subprocess
import time

import XPEEM_utils as utils
import OriginPlots as oplt
import originpro as op

# Get the directory containing the scripts from the command-line argument
root=os.getcwd()
scripts = [
            'test/PEEM_0_align.py',
            'test/PEEM_1_process_2E.py',
            'test/PEEM_1_process_ES.py',
            'test/PEEM_4_projection.py'
            ]

# whether some of the script are exporting to origin.
OriginPlots=False 

if OriginPlots and oplt.is_program_running('Origin64.exe'):
    raise OSError('Please close Origin before starting this script.')
    # You can use os.system("taskkill /f /im Origin64.exe")
else:
    # Loop through the scripts
    for i,script in enumerate(scripts):
        # Get the full path of the script
        script_path = utils.path_join(root,script,dt='f')
        
        # run the script
        try:
            result = subprocess.run(['python', script_path], check=True)
            
            if OriginPlots :
                op.attach()
                time.sleep(10)
                op.exit()
                time.sleep(10)
            
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
            if OriginPlots :
                os.system("taskkill /f /im Origin64.exe")
                time.sleep(60)
        else:
            print('Finished running script '+script)
        