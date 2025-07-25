# XPEEM_toolkit
Python scripts for processing and visualisation of XPEEM spectromicroscopy data.

## Description
This projects is an XPEEM toolkit for image alignment, E-stack correction, E-stack analysis, 2E analysis and multivariate analysis (given in the Associated content section). This is a first version that we will improve to make the code easier to use for scientists without coding experience.

The main functions of the code were detailed in the following manuscript, please cite it if reusing the functions in whole or in part:

Lelotte, B.; Siller, V.; Pelé, V.; Jordy, C.; Gubler, L.; El Kazzi, M.; Vaz, C. A. F.; Toolkit for Spectral and Multivariate Statistical Analysis of XPEEM Images for Studying Composite Energy Materials. Submitted for publication, 2025.

## Prerequisites
- Python 3.8+
- (Optional) [OriginPro](https://www.originlab.com/) for plotting  
- [ImageJ](https://imagej.nih.gov/ij/) for image alignment macros  

This script was developped on Windows 10 with Spyder 3.9 and interactive python.
We cannot guarantee complete functionality for users from different operating system or development environment. 

## How to use
### 1. Clone the repository

```bash
git clone https://github.com/blelotte/XPEEM_toolkit.git
cd XPEEM_toolkit
```

### 2. Install Python dependencies

Add the repo to your PYTHONPATH (so scripts can import xpeem_toolkit):
```
# On macOS/Linux
export PYTHONPATH="$PWD:$PYTHONPATH"

# On Windows PowerShell
setx PYTHONPATH "$PWD;$env:PYTHONPATH"
```

Install packages from our requirements.txt:
```
    conda create -n xpeem python=3.10        # if you use conda, else skip
    conda activate xpeem                     # or activate your venv
    pip install -r requirements.txt
```

### 3. Auxiliary tools & plugins

Some processing steps depend on external macros or forks of other projects. Place these folders on your PYTHONPATH or in the designated plugin folders:

#### Image alignment (ImageJ macros)
Copy the contents of imagej_macros/ into your ImageJ plugins directory:

```
<ImageJ install folder>/plugins/
```

#### 2E / E-stack processing (Savitzky–Golay filters)
Add savitzkygolay-master to your Python path:
```
export PYTHONPATH="/path/to/savitzkygolay-master:$PYTHONPATH"
```

#### PCA / ICA / NNMF for processed E-stack (using a modified version of mantis)
Add the modified mantis_xray included in this repo to your python path:
```
export PYTHONPATH="/path/to/mantis_xray:$PYTHONPATH"
```
#### OriginPro templates
If you use Origin to generate plots, copy the Origin_templates/ folder into your Origin user directory:
```
<My Documents>/OriginLab/User Files/
```
### 4. Running the toolkit

All processing steps are driven from the top‐level main.py. Once you’ve set up the environment and plugins:
```
python main.py
```
This will display usage instructions and available command-line options (e.g. input folder, output folder, processing modes).
