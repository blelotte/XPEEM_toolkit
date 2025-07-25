#
#   This file is part of Mantis, a Multivariate ANalysis Tool for Spectromicroscopy.
#
#   Copyright (C) 2013 Mirna Lerotic, 2nd Look
#   http://2ndlookconsulting.com
#   License: GNU GPL v3
#
#   Mantis is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   any later version.
#
#   Mantis is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details <http://www.gnu.org/licenses/>.


import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import colorbar
from PIL import Image
import time

#Internal imports
from . import data_struct
from . import data_stack
from . import analyze
from . import nnma
from . import henke
from . import tomo_reconstruction

from . import file_plugins
from .file_plugins import file_xrm
from .file_plugins import file_bim
from .file_plugins import file_dataexch_hdf5
from .file_plugins import file_ncb
from .file_plugins import file_json
from .file_plugins import file_tif
from .file_plugins import file_stk
from .file_plugins import file_csv

#XPEEM linked imports
import XPEEM
import OriginPlot as oplt
import XPEEM_utils as utils

PlotH = 4.0
PlotW = PlotH*1.61803

#----------------------------------------------------------------------
def save_keyeng(key_engs, odir, filename, stk, anlz, png, pdf, svg):

    SaveFileName = os.path.join(odir,'')

    matplotlib.rcParams['pdf.fonttype'] = 42

    fig = matplotlib.figure.Figure(figsize =(PlotW, PlotH))
    canvas = FigureCanvas(fig)
    fig.clf()
    fig.add_axes((0.15,0.15,0.75,0.75))
    axes = fig.gca()

    odtotal = stk.od3d.sum(axis=0)
    odtotal = odtotal.sum(axis=0)/(stk.n_rows*stk.n_cols)
    odtotal /= odtotal.max()/0.7

    specplot = axes.plot(stk.ev,odtotal)

    for i in range(len(key_engs)):
        axes.axvline(x=key_engs[i], color = 'g', alpha=0.5)

    axes.set_xlabel('Photon Energy [eV]')
    axes.set_ylabel('Optical Density')

    if png == 1:
        ext = 'png'
        fileName_img = SaveFileName+"keyengs."+ext
        fig.savefig(fileName_img, pad_inches = 0.0)
    if pdf == 1:
        ext = 'pdf'
        fileName_img = SaveFileName+"keyengs."+ext
        fig.savefig(fileName_img, pad_inches = 0.0)
    if svg == 1:
        ext = 'svg'
        fileName_img = SaveFileName+"keyengs."+ext
        fig.savefig(fileName_img, pad_inches = 0.0)


    #Save text file with list of energies
    textfilepath = SaveFileName+'_keyenergies.csv'
    f = open(textfilepath, 'w')
    print('*********************  Key Energies  ********************', file=f)
    for i in range(len(key_engs)):
        print('%.6f' %(key_engs[i]), file=f)

    f.close()

    return


#----------------------------------------------------------------------
def save_spa(wdir, odir, filename, stk, anlz, png, pdf, svg):

    SaveFileName = os.path.join(odir,'')

    matplotlib.rcParams['pdf.fonttype'] = 42

    colors=['#FF0000','#000000','#FFFFFF']
    spanclrmap=matplotlib.colors.LinearSegmentedColormap.from_list('spancm',colors)

    for i in range (anlz.n_target_spectra):

        #Save composition maps
        if anlz.pca_calculated == 0:
            tsmapimage = anlz.target_svd_maps[:,:,i]
        else:
            tsmapimage = anlz.target_pcafit_maps[:,:,i]

        fig = matplotlib.figure.Figure(figsize =(PlotH, PlotH))
        canvas = FigureCanvas(fig)
        fig.clf()
        axes = fig.gca()

        divider = make_axes_locatable(axes)
        ax_cb = divider.new_horizontal(size="3%", pad=0.03)

        fig.add_axes(ax_cb)
        axes.set_position([0.03,0.03,0.8,0.94])

        min_val = np.min(tsmapimage)
        max_val = np.max(tsmapimage)
        bound = np.max((np.abs(min_val), np.abs(max_val)))


        im = axes.imshow(tsmapimage, cmap=spanclrmap, vmin = -bound, vmax = bound)
        cbar = axes.figure.colorbar(im, orientation='vertical',cax=ax_cb)

        axes.axis("off")


        if png == 1:
            filepath = os.path.join(wdir, filename)
            
            # Lelotte_B edit
            mask=np.array(Image.open(filepath.replace('.tif','_mask.tif')))
            unique_values = np.unique(mask)
            assert np.array_equal(unique_values, [0, 255]) or np.array_equal(unique_values, [1]), "Broad mask contains values other than 0 and 1"
            mask=mask==255
            tsmapimage[~mask]=0
            
            fileName_img = SaveFileName+"TSmap_" +str(i+1)+".tif"
            img1 = Image.fromarray(tsmapimage)
            img1.save(fileName_img)

        if png == 1:
            ext = 'png'
            fileName_img = SaveFileName+"TSmap_" +str(i+1)+"."+ext
            fig.savefig(fileName_img, pad_inches = 0.0)
        if pdf == 1:
            ext = 'pdf'
            fileName_img = SaveFileName+"TSmap_" +str(i+1)+"."+ext
            fig.savefig(fileName_img, pad_inches = 0.0)
        if svg == 1:
            ext = 'svg'
            fileName_img = SaveFileName+"TSmap_" +str(i+1)+"."+ext
            fig.savefig(fileName_img, pad_inches = 0.0)

    #Save spectra
    for i in range (anlz.n_target_spectra):

        tspectrum = anlz.target_spectra[i, :]

        fig = matplotlib.figure.Figure(figsize =(PlotW, PlotH))
        canvas = FigureCanvas(fig)
        fig.clf()
        fig.add_axes((0.15,0.15,0.75,0.75))
        axes = fig.gca()

        line1 = axes.plot(stk.ev,tspectrum, color='black', label = 'Raw data')

        if anlz.pca_calculated == 1:
            tspectrumfit = anlz.target_pcafit_spectra[i, :]
            diff = np.abs(tspectrum-tspectrumfit)
            line2 = axes.plot(stk.ev,tspectrumfit, color='green', label = 'Fit')
            line3 = axes.plot(stk.ev,diff, color='grey', label = 'Abs(Raw-Fit)')

        fontP = matplotlib.font_manager.FontProperties()
        fontP.set_size('small')

        axes.legend(loc=4, prop = fontP)

        axes.set_xlabel('Photon Energy [eV]')
        axes.set_ylabel('Optical Density')

        if png == 1:
            ext = 'png'
            fileName_spec = SaveFileName+"Tspectrum_" +str(i+1)+"."+ext
            fig.savefig(fileName_spec)
        if pdf == 1:
            ext = 'pdf'
            fileName_spec = SaveFileName+"Tspectrum_" +str(i+1)+"."+ext
            fig.savefig(fileName_spec)
        if svg == 1:
            ext = 'svg'
            fileName_spec = SaveFileName+"Tspectrum_" +str(i+1)+"."+ext
            fig.savefig(fileName_spec)

        fileName_spec = SaveFileName+"Tspectrum_" +str(i+1)+".csv"
        cname = "Tspectrum_" +str(i+1)
        stk.write_csv(fileName_spec, stk.ev, tspectrum, cname = cname)

    return


#----------------------------------------------------------------------
def MakeColorTable():
    maxclcolors = 11
    colors_i = np.linspace(0, maxclcolors, maxclcolors+1)

    colors=['#0000FF','#FF0000','#DFE32D','#36F200','#B366FF',
            '#FF470A','#33FFFF','#006600','#CCCC99','#993300',
            '#000000']

    clusterclrmap1=matplotlib.colors.LinearSegmentedColormap.from_list('clustercm',colors)

    bnorm1 = matplotlib.colors.BoundaryNorm(colors_i, clusterclrmap1.N)

    colors_i = np.linspace(0,maxclcolors+2,maxclcolors+3)

    #use black color for clusters > maxclcolors, the other 2 colors are for background
    colors2=['#0000FF','#FF0000','#DFE32D','#36F200','#B366FF',
            '#FF470A','#33FFFF','#006600','#CCCC99','#993300',
            '#000000','#FFFFFF','#EEEEEE']

    clusterclrmap2=matplotlib.colors.LinearSegmentedColormap.from_list('clustercm2',colors2)

    bnorm2 = matplotlib.colors.BoundaryNorm(colors_i, clusterclrmap2.N)

    return clusterclrmap1, bnorm1, clusterclrmap2, bnorm2


#----------------------------------------------------------------------
#If png_pdg = 1 save png, if =2 save pdf, if =3 save svg
def SaveScatt(SaveFileName, stk, anlz, png_pdf = 1):

    colors=['#0000FF','#FF0000','#DFE32D','#36F200','#B366FF',
            '#FF470A','#33FFFF','#006600','#CCCC99','#993300',
            '#000000']

    od_reduced = anlz.pcaimages[:,:,0:anlz.numsigpca]
    od_reduced = np.reshape(od_reduced, (stk.n_cols*stk.n_rows,anlz.numsigpca), order='F')

    clindices = anlz.cluster_indices
    clindices = np.reshape(clindices, (stk.n_cols*stk.n_rows), order='F')

    if png_pdf == 1:
        ext = 'png'
    elif png_pdf == 2:
        ext = 'pdf'
    elif png_pdf == 3:
        ext = 'svg'

    nplots = 0
    for ip in range(anlz.numsigpca):
        for jp in range(anlz.numsigpca):
            if jp >= (ip+1):
                nplots = nplots+1
    nplotsrows = int(np.ceil(nplots/2))

    plotsize = 2.5

    if nplots > 1 :
        fig = matplotlib.figure.Figure(figsize =(6.0,plotsize*nplotsrows))
        fig.subplots_adjust(wspace = 0.4, hspace = 0.4)
    else:
        fig = matplotlib.figure.Figure(figsize =(3.0,2.5))
        fig.subplots_adjust(bottom = 0.2, left = 0.2)

    canvas = FigureCanvas(fig)
    #axes = fig.gca()
    matplotlib.rcParams['font.size'] = 6

    pplot = 1
    for ip in range(anlz.numsigpca):
        for jp in range(anlz.numsigpca):
            if jp >= (ip+1):


                x_comp = od_reduced[:,ip]
                y_comp = od_reduced[:,jp]
                if nplots > 1 :
                    axes = fig.add_subplot(nplotsrows,2, pplot)
                else:
                    axes = fig.add_subplot(1,1,1)

                pplot = pplot+1

                for i in range(anlz.nclusters):
                    thiscluster = np.where(clindices == i)
                    axes.plot(x_comp[thiscluster], y_comp[thiscluster],'.',color=colors[i],alpha=0.5)
                axes.set_xlabel('Component '+str(ip+1))
                axes.set_ylabel('Component '+str(jp+1))

    fileName_sct = SaveFileName+"CAscatterplots."+ext
    matplotlib.rcParams['pdf.fonttype'] = 42
    fig.savefig(fileName_sct)


#----------------------------------------------------------------------
def save_ca(wdir, odir, filename, stk, anlz, png, pdf, svg):

    clusterclrmap1, bnorm1, clusterclrmap2, bnorm2 = MakeColorTable()
    maxclcolors = 11
    colors=['#0000FF','#FF0000','#DFE32D','#36F200','#B366FF',
            '#FF470A','#33FFFF','#006600','#CCCC99','#993300',
            '#000000']

    SaveFileName = os.path.join(odir,'')

    matplotlib.rcParams['pdf.fonttype'] = 42

    #Save image with composite cluster indices
    fig = matplotlib.figure.Figure(figsize = (float(stk.n_rows)/30, float(stk.n_cols)/30))
    canvas = FigureCanvas(fig)
    fig.clf()
    fig.add_axes((0.0,0.0,1.0,1.0))
    axes = fig.gca()

    im = axes.imshow(anlz.cluster_indices, cmap=clusterclrmap1, norm=bnorm1)
    axes.axis("off")

    if png == 1:
        ext = 'png'
        fileName_caimg = SaveFileName+"CAcimg."+ext
        fig.savefig(fileName_caimg, dpi=300, pad_inches = 0.0)
    if pdf == 1:
        ext = 'pdf'
        fileName_caimg = SaveFileName+"CAcimg."+ext
        fig.savefig(fileName_caimg, dpi=300, pad_inches = 0.0)
    if svg == 1:
        ext = 'svg'
        fileName_caimg = SaveFileName+"CAcimg."+ext
        fig.savefig(fileName_caimg, dpi=300, pad_inches = 0.0)

    #Save individual cluster images
    for i in range (anlz.nclusters):

        indvclusterimage = np.zeros((anlz.stack.n_cols, anlz.stack.n_rows))+20.        

        filepath = os.path.join(wdir, filename)
        msk_path=filepath.replace('.tif','_mask.tif')
        if os.path.exists(msk_path):
            mask=np.array(Image.open(filepath.replace('.tif','_mask.tif')))
            unique_values = np.unique(mask)
            assert np.array_equal(unique_values, [0, 255]) or np.array_equal(unique_values, [255]), "Broad mask contains values other than 0 and 1"
            mask = mask==255
        else:
            mask=np.ones_like(indvclusterimage,dtype=bool)

        CApositions = (anlz.cluster_indices == i) & mask
        ind = np.where(CApositions) 
        colorcl = min(i,9)
        indvclusterimage[ind] = colorcl

        if png == 1:
            fileName_img = SaveFileName+"CAimg_" +str(i+1)+".tif"
            img1 = Image.fromarray(CApositions.astype(bool))
            img1.save(fileName_img)
        
        fig = matplotlib.figure.Figure(figsize =(float(stk.n_rows)/30, float(stk.n_cols)/30))
        canvas = FigureCanvas(fig)
        fig.add_axes((0.0,0.0,1.0,1.0))
        axes = fig.gca()
        im = axes.imshow(indvclusterimage, cmap=clusterclrmap2, norm=bnorm2)
        axes.axis("off")

        if png == 1:
            ext = 'png'
            fileName_img = SaveFileName+"CAimg_" +str(i+1)+"."+ext
            fig.savefig(fileName_img, dpi=300, pad_inches = 0.0)
        if pdf == 1:
            ext = 'pdf'
            fileName_img = SaveFileName+"CAimg_" +str(i+1)+"."+ext
            fig.savefig(fileName_img, dpi=300, pad_inches = 0.0)
        if svg == 1:
            ext = 'svg'
            fileName_img = SaveFileName+"CAimg_" +str(i+1)+"."+ext
            fig.savefig(fileName_img, dpi=300, pad_inches = 0.0)

    for i in range (anlz.nclusters):

        clusterspectrum = anlz.clusterspectra[i, ]
        fig = matplotlib.figure.Figure(figsize =(PlotW, PlotH))
        canvas = FigureCanvas(fig)
        fig.add_axes((0.15,0.15,0.75,0.75))
        axes = fig.gca()
        if i >= maxclcolors:
            clcolor = colors[maxclcolors-1]
        else:
            clcolor = colors[i]

        specplot = axes.plot(anlz.stack.ev,clusterspectrum, color = clcolor)

        axes.set_xlabel('Photon Energy [eV]')
        axes.set_ylabel('Optical Density')

        if png == 1:
            ext = 'png'
            fileName_spec = SaveFileName+"CAspectrum_" +str(i+1)+"."+ext
            fig.savefig(fileName_spec)
        if pdf == 1:
            ext = 'pdf'
            fileName_spec = SaveFileName+"CAspectrum_" +str(i+1)+"."+ext
            fig.savefig(fileName_spec)
        if svg == 1:
            ext = 'svg'
            fileName_spec = SaveFileName+"CAspectrum_" +str(i+1)+"."+ext
            fig.savefig(fileName_spec)

        #Save all spectra in one plot
        fig = matplotlib.figure.Figure(figsize =(PlotW, PlotH))
        canvas = FigureCanvas(fig)
        fig.add_axes((0.15,0.15,0.75,0.75))
        axes = fig.gca()

        for i in range (anlz.nclusters):

            clusterspectrum = anlz.clusterspectra[i-1, ]/np.amax(anlz.clusterspectra[i-1, ])

            if i >= maxclcolors:
                clcolor = colors[maxclcolors-1]
            else:
                clcolor = colors[i]

            specplot = axes.plot(anlz.stack.ev,clusterspectrum, color = clcolor)

            axes.set_xlabel('Photon Energy [eV]')
            axes.set_ylabel('Optical Density')
        if png == 1:
            ext = 'png'
            fileName_spec = SaveFileName+"CAspectra"+"."+ext
            fig.savefig(fileName_spec)
        if pdf == 1:
            ext = 'pdf'
            fileName_spec = SaveFileName+"CAspectra"+"."+ext
            fig.savefig(fileName_spec)
        if svg == 1:
            ext = 'svg'
            fileName_spec = SaveFileName+"CAspectra"+"."+ext
            fig.savefig(fileName_spec)

    for i in range (anlz.nclusters):
        clusterspectrum = anlz.clusterspectra[i, ]
        fileName_spec = SaveFileName+"CAspectrum_" +str(i+1)+".csv"
        cname = "CAspectrum_" +str(i+1)
        stk.write_csv(fileName_spec, anlz.stack.ev, clusterspectrum, cname=cname)

    if png:
        SaveScatt(SaveFileName, stk, anlz, png_pdf = 1)
    if pdf:
        SaveScatt(SaveFileName, stk, anlz, png_pdf = 2)
    if svg:
        SaveScatt(SaveFileName, stk, anlz, png_pdf = 3)

    return


#----------------------------------------------------------------------
def save_pca(odir, filename, stk, anlz, png, pdf, svg):

    SaveFileName = os.path.join(odir,'')

    matplotlib.rcParams['pdf.fonttype'] = 42

    #Save evals
    evalmax = np.min([stk.n_ev, 40])
    pcaevals = anlz.eigenvals[0:evalmax]
    fig = Figure(figsize =(PlotW, PlotH))
    canvas = FigureCanvas(fig)

    fig.clf()
    fig.add_axes((0.15,0.15,0.75,0.75))
    axes = fig.gca()

    evalsplot = axes.semilogy(np.arange(1,evalmax+1), pcaevals,'b.')

    axes.set_xlabel('Principal Component')
    axes.set_ylabel('Log(Eigenvalue)')

    if png == 1:
        ext = 'png'
        fileName_evals = SaveFileName+"PCAevals."+ext
        fig.savefig(fileName_evals, pad_inches = 0.0)
    if pdf == 1:
        ext = 'pdf'
        fileName_evals = SaveFileName+"PCAevals."+ext
        fig.savefig(fileName_evals, pad_inches = 0.0)
    if svg == 1:
        ext = 'svg'
        fileName_evals = SaveFileName+"PCAevals."+ext
        fig.savefig(fileName_evals, pad_inches = 0.0)

    for i in range(10):
        pcaimage = anlz.pcaimages[:,:,i]

        fileName_img = SaveFileName+"PCA_" +str(i+1)+".tif"
        img1 = Image.fromarray(pcaimage)
        img1.save(fileName_img)

        fig = Figure(figsize =(PlotH*1.15, PlotH))
        canvas = FigureCanvas(fig)
        axes = fig.gca()
        divider = make_axes_locatable(axes)
        ax_cb = divider.new_horizontal(size="3%", pad=0.03)
        fig.add_axes(ax_cb)
        axes.set_position([0.03,0.03,0.8,0.94])
        bound = anlz.pcaimagebounds[i]

        im = axes.imshow(pcaimage, cmap=matplotlib.cm.get_cmap("seismic_r"), vmin = -bound, vmax = bound)

        cbar = colorbar(im, orientation='vertical', cax=ax_cb)
        axes.axis("off")

        if png == 1:
            ext = 'png'
            fileName_img = SaveFileName+"PCA_" +str(i+1)+"."+ext
            fig.savefig(fileName_img, pad_inches = 0.0)
        if pdf == 1:
            ext = 'pdf'
            fileName_img = SaveFileName+"PCA_" +str(i+1)+"."+ext
            fig.savefig(fileName_img, pad_inches = 0.0)
        if svg == 1:
            ext = 'svg'
            fileName_img = SaveFileName+"PCA_" +str(i+1)+"."+ext
            fig.savefig(fileName_img, pad_inches = 0.0)

    for i in range(10):
        pcaspectrum = anlz.eigenvecs[:,i]
        fig = Figure(figsize =(PlotW, PlotH))
        canvas = FigureCanvas(fig)
        fig.add_axes((0.15,0.15,0.75,0.75))
        axes = fig.gca()
        specplot = axes.plot(stk.ev, pcaspectrum)
        axes.set_xlabel('Photon Energy [eV]')
        axes.set_ylabel('Optical Density')
        if png == 1:
            ext = 'png'
            fileName_spec = SaveFileName+"PCAspectrum_" +str(i+1)+"."+ext
            fig.savefig(fileName_spec)
        if pdf == 1:
            ext = 'pdf'
            fileName_spec = SaveFileName+"PCAspectrum_" +str(i+1)+"."+ext
            fig.savefig(fileName_spec)
        if svg == 1:
            ext = 'svg'
            fileName_spec = SaveFileName+"PCAspectrum_" +str(i+1)+"."+ext
            fig.savefig(fileName_spec)

    for i in range (10):
        pcaspectrum = anlz.eigenvecs[:,i]
        fileName_spec = SaveFileName+"PCAspectrum_" +str(i+1)+".csv"
        cname = "PCAspectrum_" +str(i+1)
        stk.write_csv(fileName_spec, stk.ev, pcaspectrum, cname=cname)

    return

def save_nnma(odir, filename, nComponents, stk, nnmat, png, pdf, svg, origin, sample, ROI, edge, shift_col):


    SaveFileName = os.path.join(odir,'')
        
    costTotal = nnmat.costFnArray[:,0]
    costSparse = nnmat.costFnArray[:,1]
    costClusterSim = nnmat.costFnArray[:,2]
    costSmooth = nnmat.costFnArray[:,3]

    ext = 'png'
    fileName_evals = SaveFileName+"CostFunction."+ext
    
    fig = matplotlib.figure.Figure(figsize =(PlotH*1.15, PlotH))
    fig.clf()
    fig.add_axes((0.20,0.15,0.75,0.75))
    axes = fig.gca()

    fileName_evals = SaveFileName+"CostFunction."+ext
    fig.savefig(fileName_evals)

    _line1 = axes.plot(costTotal, label='Total')        
    _line2 = axes.plot(costSparse, label='Sparseness')
    _line3 = axes.plot(costClusterSim, label='Similarity')
    _line4 = axes.plot(costSmooth, label='Smoothness')

    
    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('small')

    axes.legend(loc=1, prop = fontP)


    axes.set_xlabel('Iteration')
    axes.set_ylabel('Cost Function')

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    matplotlib.rcParams['pdf.fonttype'] = 42

    ext = 'png'

    colors=['#FF0000','#000000','#FFFFFF']
    spanclrmap=matplotlib.colors.LinearSegmentedColormap.from_list('spancm',colors)

    # if map_png:
    for i in range(nComponents):

        mapimage = nnmat.tRecon[i, :,:]

        fig = matplotlib.figure.Figure(figsize =(PlotH*1.15, PlotH))
        _canvas = FigureCanvas(fig)
        axes = fig.gca()

        divider = make_axes_locatable(axes)
        ax_cb = divider.new_horizontal(size="3%", pad=0.03)
        fig.add_axes(ax_cb)
        axes.set_position([0.03,0.03,0.8,0.94])

        min_val = np.min(mapimage)
        max_val = np.max(mapimage)
        bound = np.max((np.abs(min_val), np.abs(max_val)))

        im = axes.imshow(np.rot90(mapimage), cmap=spanclrmap, vmin = -bound, vmax = bound)
        _cbar = axes.figure.colorbar(im, orientation='vertical',cax=ax_cb)

        axes.axis("off")

        fileName_img = SaveFileName+"NNMA_" +str(i+1)+"."+ext
        fig.savefig(fileName_img, pad_inches = 0.0)
        

    # if spec_png:
    for i in range(nComponents):

        fig = matplotlib.figure.Figure(figsize =(PlotW, PlotH))
        _canvas = FigureCanvas(fig)
        fig.add_axes((0.15,0.15,0.75,0.75))
        axes = fig.gca()

        nspectrum = nnmat.muRecon[:,i]

        line1 = axes.plot(stk.ev,nspectrum)
        lcolor = line1[0].get_color()

        axes.set_xlabel('Photon Energy [eV]')
        axes.set_ylabel('Absorption coefficient [a.u.]')

        fileName_spec = SaveFileName+"NNMAspectrum_" +str(i+1)+"."+ext
        fig.savefig(fileName_spec)

    # if spec_csv:
    for i in range(nComponents):
        nspectrum = nnmat.muRecon[:,i]
        fileName_spec = SaveFileName+"NNMAspectrum_" +str(i+1)+".csv"
        cname = "NNMAspectrum_" +str(i+1)
        stk.write_csv(fileName_spec, stk.ev, nspectrum, cname = cname)
        
        if nnmat.initmatrices == 'FastICA' :
            fileName_ICA = SaveFileName+"ICAspectrum_" +str(i+1)+".csv"
            cname = "ICAspectrum_" +str(i+1)
            stk.write_csv(fileName_ICA, stk.ev, nnmat.muICA[:,i], cname = cname)
            print('Saved ICA')

        
        
    # if spec_origin:
    if origin == 1: 
        loc=os.getcwd()
        filename = 'OriginPlots.opju'
    
        nspectrum = nnmat.muRecon[:,i]
        com = [sample + " C. "+str(i+1) for i in range(nComponents)]
        oplt.AddSheetOrigin(loc, filename, stk.ev, nnmat.muRecon, comments=['NNMA']+com, foldername=f'Mantis/{edge}', bookname='G'+'NNMA'+edge, ShNam=edge+ROI+'--_NNMA', shiftCol=shift_col)


    # if map_tif:
    for i in range(nComponents):
        mapimage = nnmat.tRecon[i, :,:] 
        if isinstance(nnmat.msk,np.ndarray):
            mapimage*=nnmat.msk
        fileName_img = SaveFileName+"NNMA_" +str(i+1)+".tif"
        img1 = Image.fromarray(mapimage)
        img1.save(fileName_img)
        
    if nnmat.initmatrices == 'FastICA'  :
        if isinstance(nnmat.msk,np.ndarray):
            ICAmap=np.zeros_like(nnmat.Temp)
            ICAmap[:,nnmat.fltmsk]=nnmat.mixmatrix
        else:
            ICAmap=nnmat.mixmatrix
            
        for i in range(nComponents):
            fileName_img = SaveFileName+"ICA_" +str(i+1)+".tif"
            imgICA = Image.fromarray(ICAmap[i,:].reshape(nnmat.nCols,nnmat.nRows).T)
            imgICA.save(fileName_img)
            print('Saved ICA')


        if origin == 1: 
            loc=os.getcwd()
            filename = 'OriginPlots.opju'
            
            oplt.AddImageOrigin(mapimage, f'Mantis/{edge}', f'M{i}'+sample+edge+'--Map',edge+ROI+'NNMA')
            
    # if error_tif:
    SpectraRecon = np.zeros((nnmat.stack.n_ev,nnmat.stack.n_cols*nnmat.stack.n_rows))
    mapimage = nnmat.tRecon
    if isinstance(nnmat.msk,np.ndarray):
        mapimage*=nnmat.msk[np.newaxis,:,:]
    SpectraRecon[:,:] = nnmat.muRecon @ mapimage.reshape(nnmat.kNNMA,nnmat.stack.n_cols*nnmat.stack.n_rows)
        
    SpectraRecon=SpectraRecon.reshape(nnmat.stack.n_ev,nnmat.stack.n_cols,nnmat.stack.n_rows)

    rawstack=np.reshape(nnmat.stack.od, (nnmat.nCols, nnmat.nRows, nnmat.stack.n_ev), order='F')
    if isinstance(nnmat.msk,np.ndarray):
        rawstack*=nnmat.msk[:,:,np.newaxis]
    
    # Calculate and save residuals
    Residuals=(rawstack)-np.transpose(SpectraRecon,(1,2,0))
    msk_neg=np.nanmean(Residuals,axis=2)<0
    msk_pos=np.nanmean(Residuals,axis=2)>0
    MapResiduals=np.zeros((nnmat.nCols, nnmat.nRows))
    MapResiduals[msk_pos] = np.sqrt(np.nanmean(Residuals[msk_pos,:]**2,axis=1))
    MapResiduals[msk_neg] = np.sqrt(np.nanmean(Residuals[msk_neg,:]**2,axis=1))
    fileName_img = SaveFileName+"NNMA_sum_squarred_residuals_map.tif"
    img1 = Image.fromarray(MapResiduals)
    img1.save(fileName_img)
    
    # Calculate and save R squarred
    utils.scatter_mean_stack(nnmat.energies,rawstack,'Mean_error',None, norm=-1,mask_xy=nnmat.msk==1, plot=False)
    MapStd=np.zeros((nnmat.nCols, nnmat.nRows))
    # MapStd[nnmat.msk==1]=np.sqrt(np.nanmean((rawstack[nnmat.msk==1,:]-stack_mean[np.newaxis,:])**2,axis=1))
    MapStd=np.sqrt(np.nanmean((rawstack)**2,axis=2))
    MapRsquarred=np.where(MapStd>0,1-(MapResiduals/MapStd),0)
    fileName_img = SaveFileName+"NNMA_Rsquarred_map.tif"
    img1 = Image.fromarray(MapRsquarred)
    img1.save(fileName_img)

    ax,SpectraError,SpectraError_std,SpectraErrorLabel = utils.scatter_mean_stack(nnmat.energies,Residuals,'Mean_error',None, norm=-1)
    posErr=np.percentile(abs(MapResiduals),90)
    posmsk10=MapResiduals>posErr
    ax,posSpectraError,posSpectraError_std,posSpectraErrorLabel = utils.scatter_mean_stack(nnmat.energies,Residuals,'Mean_+_error_highest_10pt',None, norm=-1,mask_xy=posmsk10)
    negErr=np.percentile(abs(MapResiduals),10)
    negmsk10=MapResiduals<negErr
    ax,negSpectraError,negSpectraError_std,negSpectraErrorLabel = utils.scatter_mean_stack(nnmat.energies,Residuals,'Mean_-_error_highest_10pt',None, norm=-1,mask_xy=negmsk10)

    ErrorSpectra=np.vstack((nnmat.energies,posSpectraError,posSpectraError_std[0],posSpectraError_std[1],SpectraError,SpectraError_std[0],SpectraError_std[1],negSpectraError,posSpectraError_std[0],posSpectraError_std[1])).T
    np.savetxt(utils.path_join(SaveFileName,'NNMA_total_Error_spectra.csv',dt='f'), ErrorSpectra, delimiter=";", header=f'Energy;{posSpectraErrorLabel};{posSpectraErrorLabel} std below;{posSpectraErrorLabel} std above;{SpectraErrorLabel};{SpectraErrorLabel} std below;{SpectraErrorLabel} std above;{negSpectraErrorLabel};{negSpectraErrorLabel} std below;{negSpectraErrorLabel} std above')

def batch_mode():
    verbose  = 1
    settingsfile = 'Mantis_batch_settings.txt'

    version = '2.0.5'
    wdir = ''
    outdir = 'MantisResults'
    filename = ''
    save_hdf5 = 0
    align_stack = 0
    i0_file = ''
    i0_histogram = 0
    run_pca = 0
    n_spca = 4
    run_ca = 0
    nclusters = 5
    ca_thickness = 0
    run_sa = 0
    sa_spectra = []
    sa_use_clspectra = 0
    run_nnma = 0
    nnma_components= None
    run_keyengs = 0
    kengs_thresh = 0.10
    save_png = 1
    save_pdf = 0
    save_svg = 0
    save_origin = 0
    sample = 'XPEEM'
    ROI = 'All'
    edge = 'Ni La'
    shiftcol=0
    Eselect=[]

    try:
        f = open(settingsfile, 'rt')
        for line in f:
            if ':' in line :
                slist = line.split(':')
                tag = slist[0]
                value = ':'.join(slist[1:])

                if   tag == 'VERSION': version = float(value)
                elif tag == 'WORK_DIR' : wdir  =  value.strip()
                elif tag == 'OUTPUT_DIR_NAME' : outdir  =  value.strip()
                elif tag == 'FILENAME' : filename  =  value.strip()
                elif tag == 'ALIGN_STACK' : align_stack  =  value.strip()
                elif tag == 'I0_FILE' : i0_file  =  value.strip()
                elif tag == 'I0_HISTOGRAM' : i0_histogram = int(value)
                elif tag == 'SAVE_HDF5' : save_hdf5 = int(value)
                elif tag == 'RUN_PCA' : run_pca = int(value)
                elif tag == 'N_SPCA' : n_spca = int(value)
                elif tag == 'RUN_CLUSTER_ANALYSIS' : run_ca = int(value)
                elif tag == 'N_CLUSTERS' : nclusters = int(value)
                elif tag == 'THICKNESS_CORRECTION' : ca_thickness = int(value)
                elif tag == 'RUN_SPECTRAL_ANALYSIS' : run_sa = int(value)
                elif tag == 'SA_SPECTRUM' :
                    spname = value.strip()
                    if len(spname) > 0 :
                        sa_spectra.append(spname)
                elif tag == 'SA_USE_CA_SPECTRA' : sa_use_clspectra = int(value)
                elif tag == 'RUN_NNMA' : run_nnma = int(value)
                elif tag == 'NNMA_COMPONENTS' : 
                    s = value.strip()
                    nnma_components = [int(i) for i in s.split(',')]
                    assert len(nnma_components) <= nclusters
                elif tag == 'RUN_KEY_ENGS' : run_keyengs = int(value)
                elif tag == 'KE_THRESHOLD' : kengs_thresh = float(value)
                elif tag == 'SAVE_PNG' : save_png = int(value)
                elif tag == 'SAVE_PDF' : save_pdf = int(value)
                elif tag == 'SAVE_SVG' : save_svg = int(value)
                elif tag == 'SAVE_ORIGIN' : save_origin = int(value)
                elif tag == 'SAMPLE' : sample  =  value.strip()
                elif tag == 'ROI' : ROI  =  value.strip()
                elif tag == 'EDGE' : edge  =  value.strip()
                elif tag == 'SHIFT' : shiftcol = int(value)

        f.close()

    except:
        print('Error: Could not read in Mantis_batch_settings.txt.')
        return

    wdir = os.path.normpath(wdir)

    if verbose:
        print('Version: ', version)
        print('Working directory: ', wdir)

    if not os.path.exists(wdir):
        print('Error - Directory ', wdir, ' does not exist. Please specify working directory.')
        return

    outdir = os.path.join(wdir, outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        if not os.path.exists(outdir):
            print('Error: Did not find and could not create a new output directory.')
            return

    if save_png == 1:
        print("Save .png images")

    if save_pdf == 1:
        print("Save .pdf images")

    datastruct = data_struct.h5()
    stk = data_stack.data(datastruct)
    anlz = analyze.analyze(stk)

    print('Reading file:', filename)
    basename, extension = os.path.splitext(filename)
    filepath = os.path.join(wdir, filename)

    supported_filters = file_plugins.supported_filters
    filter_list = file_plugins.filter_list

    action = 'read'
    data_type = 'stack'

    print (filter_list[action][data_type])

    plugin = file_plugins.identify(filepath)

    file_plugins.load(filepath, stack_object=stk, plugin=plugin, selection=None, json=None)


    if align_stack:
        print('Aligning the stack')
        xshifts = np.zeros((stk.n_ev))
        yshifts = np.zeros((stk.n_ev))

        referenceimage = stk.absdata[:,:,0].copy()

        for i in range(stk.n_ev):

            img2 = stk.absdata[:,:,i]

            if i==0:
                xshift, yshift, ccorr = stk.register_images(referenceimage, img2,
                                                          have_ref_img_fft = False)
            else:
                xshift, yshift, ccorr = stk.register_images(referenceimage, img2,
                                                          have_ref_img_fft = True)

#             #Limit the shifts to MAXSHIFT chosen by the user
#             if (self.maxshift > 0):
#                 if (abs(xshift) > self.maxshift):
#                         xshift = npy.sign(xshift)*self.maxshift
#                 if (abs(yshift) > self.maxshift):
#                         yshift = npy.sign(yshift)*self.maxshift

            xshifts[i] = xshift
            yshifts[i] = yshift

        #Apply shifts
        for i in range(stk.n_ev):
            img = stk.absdata[:,:,i]
            if (abs(xshifts[i])>0.02) or (abs(yshifts[i])>0.02):
                shifted_img = stk.apply_image_registration(img, xshifts[i], yshifts[i])
                stk.absdata[:,:,i] = shifted_img

    if datastruct.spectromicroscopy.normalization.white_spectrum is not None:
        print("I0 loaded")
    else:
        print("Loading I0")
        if i0_histogram == 1:
            print('Getting I0 from the histogram')
            stk.calc_histogram()
            averagefluxmax = np.max(stk.histogram)
            histmin = 0.98*averagefluxmax
            histmax = averagefluxmax
            stk.i0_from_histogram(histmin, histmax)

        elif len(i0_file) > 0:
            print('Reading I0 from file:', i0_file)
            i0basename, i0extension = os.path.splitext(i0_file)
            i0filepath = os.path.join(wdir, i0_file)
            stk.read_stk_i0(i0filepath, i0extension)

        else:
            print("Please either set I0_HISTOGRAM to 1 to calculate I0 or specify I0 file.")
            return


    if datastruct.spectromicroscopy.normalization.white_spectrum is None:
        print('Error: I0 not loaded')
        return

    if save_hdf5 == 1:
        fnameh5 =  os.path.join(wdir,basename+'_MantisBatch.hdf5')
        stk.write_h5(fnameh5, data_struct)
        print('Saving data to HDF5 file:', fnameh5)

    stk.UsePreNormalizedData()

    pca_calculated = 0
    if run_pca == 1:
        print("Running PCA Analysis")
        startTime=time.time()
        anlz.calculate_pca()
        endTime = time.time()
        timePCA = endTime - startTime
        print(f'Time PCA: {timePCA}')

        print("Chosen number of significant components:", n_spca)
        print("Suggested number of significant components:", anlz.numsigpca)
        pca_calculated = 1
        anlz.numsigpca = n_spca
        save_pca(outdir, filename, stk, anlz, save_png, save_pdf, save_svg)

    ca_calculated = 0
    if run_ca == 1:
        if pca_calculated == 0:
            anlz.calculate_pca()
        print("Running Cluster Analysis")
        print("Number of clusters",  nclusters)
        if ca_thickness == 1:
            print("Thickness correction enabled")
        startTime = time.time()
        nclusters = anlz.calculate_clusters(nclusters, ca_thickness)
        endTime = time.time()
        timeCA = endTime - startTime
        print(f'Time Kmeans: {timeCA}')
        ca_calculated = 1
        save_ca(wdir,outdir, filename, stk, anlz, save_png, save_pdf, save_svg)

    if run_sa == 1:
        print("Running Spectral Analysis")
        if len(sa_spectra) > 0:
            print("Loading spectra:", sa_spectra)
            for i in range(len(sa_spectra)):
                sppath = os.path.join(wdir, sa_spectra[i])
                anlz.read_target_spectrum(filename=sppath)

        if sa_use_clspectra == 1:
            if ca_calculated == 1:
                print("Loading cluster spectra")
                startTime = time.time()
                anlz.add_cluster_target_spectra()
                endTime = time.time()
                timeSA = endTime - startTime
                print(f'Time CA: {timeSA}')
            else:
                print("Please set RUN_CLUSTER_ANALYSIS to 1 to calculate cluster spectra.")

        if anlz.n_target_spectra > 1:
            save_spa(wdir, outdir, filename, stk, anlz, save_png, save_pdf, save_svg)
            
    if run_nnma == 1 : 
        if ca_calculated == 1:
            
            # Lelotte_B edit
            mask=np.array(Image.open(filepath.replace('.tif','_mask.tif')))
            unique_values = np.unique(mask)
            assert np.array_equal(unique_values, [0, 255]) or np.array_equal(unique_values, [255]), "Broad mask contains values other than 0 and 1"
            fltmsk=(mask.T).flatten()==255
            
            n_components=len(nnma_components)
            initmat='Ratio'
            
            nnmat=nnma.nnma(stk)
            if initmat == 'Cluster':
                nnmat.setClusterSpectra(anlz.clusterspectra[nnma_components,:],np.transpose(anlz.target_svd_maps[:,:,nnma_components],(1,0,2))[mask.T==255,:])
            elif initmat == 'Ratio':
                print('initmat Ratio')
                ratioimage_tiff = Image.open(filepath.replace('.tif','_ratioimages.tif'))
                try:
                    ratioimages = []
                    n_components=0
                    while True:
                        # Convert the current frame to a numpy array and append to the list
                        frame_array = np.array(ratioimage_tiff)
                        ratioimages.append(frame_array)
                        n_components+=1
                        # Move to the next frame
                        ratioimage_tiff.seek(ratioimage_tiff.tell() + 1)
                except EOFError:
                    # End of file reached
                    pass
                ratioimages=np.transpose(np.array(ratioimages),axes=(2,1,0)).reshape(-1,n_components)
                ratioimages=ratioimages[fltmsk,:]

                ratiospectras = np.loadtxt(filepath.replace('.tif','_ratiospectra.csv'), delimiter=';')
                
                plt.figure(figsize=(10, 6))
                plt.plot(stk.ev,ratiospectras)
                plt.title('Init vectors Ratio')
                plt.show()
                
                nnmat.setRatioSpectra(ratiospectras,ratioimages)
                
                
            nnmat.setParameters(kNNMA = n_components,
                                            maxIters = 50, #Ni L-edge 200
                                            deltaErrorThreshold = 0.01,
                                            initMatrices = initmat,
                                            lambdaSparse = 0,
                                            lambdaClusterSim = 0.001, #Ni L-edge 0.001
                                            lambdaSmooth = 1,
                                            msk=mask
                                            )
            
            nnmat.calcNNMA(initmatrices = initmat)
            
            save_nnma(outdir, filename, n_components, stk, nnmat, save_png, save_pdf, save_svg, save_origin, sample, ROI, edge, shiftcol)
            
    if run_keyengs == 1:
        if pca_calculated == 0:
            anlz.calculate_pca()
        print("Finding key energies")
        print("Threshold for finding key energies:", kengs_thresh)
        key_engs= anlz.calc_key_engs(kengs_thresh)
        save_keyeng(key_engs, outdir, filename, stk, anlz, save_png, save_pdf, save_svg)

    if (save_hdf5 == 1) and (pca_calculated == 1) :
        fnameh5 = os.path.join(wdir,basename+'_MantisBatch.hdf5')
        stk.write_results_h5(fnameh5, data_struct, anlz)


    print("Finished doing Mantis analysis")
    return True


""" ------------------------------------------------------------------------------------------------"""
def main():

    verbose = True

    print('Running Mantis in batch mode.')
    
    return batch_mode()

if __name__ == '__main__':
    main()

