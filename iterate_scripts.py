# -*- coding: utf-8 -*-
"""
This script iterates through various local scripts defined elsewhere.

Typically I define a "local script" which calls the "module" near the experiment
files. The present script allows to iterate various "local scripts"

Created on Tue Sep 26 20:57:25 2023

@author: lelotte_b
"""

import os
import ElectroChem as EC
import subprocess
import time
import originpro as op

# Get the directory containing the scripts from the command-line argument
root='D:/Documents/a PSI/Data/Data analysis/spyder/2107_Progress_work'
# file1='2107 Comparison LPSCl-LPS/pristine_LPSCl.py'
# file2='2212_LNO_coating/LNO_coated_samples.py'
# file3='2112_fluorine_modified/fluorine_mod_cleaned.py'
# file4='2108_oxygen modified/Oxygen_modified_samples_cleaned.py'
# file5='0410_SIM beamline 1/PEEM_py/PEEM_3_processing_spectrum.py'
# file6='2403_oXAS/operandoXPEEM-Valerie/PEEM_processing_spectrum.py'

file7='0410_SIM beamline 1/PEEM_py/PEEM_3_processing_spectrum.py'
file8='2403_oXPEEM/oXPEEM-Pristine/PEEM_3_processing_spectrum.py'
file9='2403_oXPEEM/oXPEEM-LNO/PEEM_3_processing_spectrum.py'
# file10='2403_oXPEEM/oXPEEM-Pristine/PEEM_2_processing_twoEnergies.py'
# file11='2403_oXPEEM/oXPEEM-LNO/PEEM_2_processing_twoEnergies.py'


scripts = [file8,file9]

# root=os.getcwd()
# scripts=['tests.py','tests.py']

if not EC.is_program_running('Origin64.exe'):
    # Loop through the scripts
    for i,script in enumerate(scripts):
        # Get the full path of the script
        script_path = EC.path_join(root,script,dt='f')
        
        # run the script
        try:
            
            result = subprocess.run(['python', script_path], check=True)
            
            op.attach()
            time.sleep(10)
            op.exit()
            time.sleep(10)
            
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
            os.system("taskkill /f /im Origin64.exe")
            time.sleep(60)
        else:
            print('Finished running script '+script)
else: 
    raise OSError('Please close Origin before starting this script.')
    # You can use os.system("taskkill /f /im Origin64.exe")
        