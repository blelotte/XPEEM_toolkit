# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 09:30:12 2025

@author: lelotte_b
"""

# Basics
import time
import re
import numpy as np
import subprocess
from typing import Dict, Optional
import ast
import sys
import pyautogui
import pandas as pd
from PIL import Image
import seaborn as sns 
sns.set_context('paper', font_scale=2.2) # to change the font settings in the graphs
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina') # High resolution plots

# Docstrings, test and warnings
np.seterr(all='warn') # to be able to handle warnings

# Folder/system management
import psutil
import os

import XPEEM_utils as utils

def is_program_running(program_name):
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == program_name:
            return True
    return False

# Handle image inputs and outputs
# Simple, Python-native API for image I/O and basic transforms 

#Handling origin
import originpro as op
def origin_shutdown_exception_hook(exctype, value, traceback):
    '''Ensures Origin gets shut down if an uncaught exception'''
    op.exit()
    sys.__excepthook__(exctype, value, traceback)
if op and op.oext:
    sys.excepthook = origin_shutdown_exception_hook

op.exit


def openFileOrigin(path: Optional[str] = "", filename: Optional[str] = "", newfolder: Optional[str] = "", bookname: Optional[str] = None, booklname: Optional[str] = None, typeBook: Optional[str] = 'w'):
    """
    Open an Origin file and return the workbook.

    Parameters
    ----------
    path : str, optional
        Path to the file.
    filename : str, optional
        Name of the file.
    newfolder : str, optional
        Name of the new folder to create.
    bookname : str, optional
        Name of the workbook.

    Returns
    -------
    book0 : Origin workbook
        The opened workbook.
    """
    op.set_show(False)
    # Create a blank project if a filename is provided
    if filename:
        fileopen = op.open(os.path.join(path, filename))
        if fileopen: 
            print(f'Origin: the project {filename} was open')
    
    # Create a new folder in the base folder if newfolder is provided
    if newfolder:
        op.pe.cd('/OriginPlots')
        pathFd = op.pe.search(newfolder, kind=1)
        if not pathFd:
            op.pe.mkdir(newfolder, chk=True)
            op.pe.cd(newfolder)
        else:
            op.pe.cd(pathFd)
    
    # Open a workbook if bookname is provided, or create a new one if it doesn't exist
    if bookname:
        openBook = op.project.find_book(type=typeBook, name=bookname)
        if openBook is None:
            book0 = op.new_book(type=typeBook, lname=bookname)
        else:
            book0 = openBook    
        return book0
    
    # Set long name    
    if booklname:
        book0.lname=booklname
    else:
        book0.lname=bookname
        
def openSheetOrigin(book0: object, ShNam: str, bookname: str, typeBook: str= 'w') -> object:
    """
    This function adds a new sheet to an Origin workbook or returns an existing sheet with the same name.

    Parameters:
    book0 (object): The Origin workbook where the sheet will be added.
    ShNam (str): The name of the sheet to be added.
    bookname (str): The name of the workbook.
    typeBook (str, optional): The type (w) or (m).

    Returns:
    object: The new or existing sheet.
    """
    default_sheet = book0[0]
    # By default, the first sheet in a new project is Sheet1.
    # The first matrix is MSheet1
    # If this sheet still exists, rename it to the target sheet.
    if default_sheet.name == 'Sheet1' or default_sheet.name == 'MSheet1' :
        localsheet = default_sheet
        localsheet.name = ShNam
        print(f'Origin: created sheet {ShNam}')
        
    # Otherwise, try to find a sheet having the same name.
    else:
        localsheet = op.find_sheet(typeBook, '[' + bookname + ']' + ShNam)
        # If the sheet is not found, create a new one.
        if localsheet == None:
            localsheet = book0.add_sheet(name=ShNam)
        else:            
            print(f'Origin: found sheet {ShNam}')
    return localsheet, localsheet.index()


def openGraphOrigin(name: str, template_file: Optional[str] = None, longname: str = '', replace: bool = False) -> op.graph:
    """
    Open an Origin graph. If the graph does not exist, create a new one based on a template.

    Parameters
    ----------
    name : str
        Name of the graph.
    template_file : str, optional
        Path to the template file. If not provided, a default template will be used.
    longname : str, optional
        Long name for the graph.
    replace : bool, optional
        Defines whether to delete the existing graph and replace it with a new one.

    Returns
    -------
    graph : Origin graph
        The opened or created graph.

    Raises
    ------
    ValueError
        If the name is not a string, or if the template file does not exist.
    """    
    if not isinstance(name,str): raise ValueError('The graph name should be a string.')
    
    # Try to find the graph and, 
    graph=op.find_graph(name=name)
    
    # if replace is set to True and the graph was found, erase it.
    if graph and replace :
        graph.destroy()
        graph = None
    
    
    # If didn't find the graph or the graph was deleted, create one based on a template.
    if not graph : 
        if not template_file :
            template_file=os.path.abspath(os.path.join(op.path('e'),'Line.otp'))
        else :
            template_file+='.otpu'
            template_path=utils.path_join(utils.path_join(op.path('u'),'User Files'),template_file,dt='f')
        if not os.path.exists(template_path) : raise ValueError('The template was not found:'+template_path)
        
        graph=op.new_graph(lname=name,template=template_path,hidden=False)
        if not replace :
            print(f'The graph "{name}" was created.')
        else:
            print(f'The graph "{name}" was updated.')
    elif graph : 
        graph.activate()
        print(f'The graph "{name}" was found.')
    
    
    graph.lname=longname
    return graph



def statistics_Origin(subtext,book_name,sheet_names=['CapaComp'],book_path='OriginPlots/Summary',range_col=None,fct_names=['mean','std']):
    """
    Perform statistical analysis on an Origin project.
   
    Parameters
    ----------
    subtext : list of str
        Subtext to match in the spreadsheet.
    book_name : str
        Name of the book in the Origin project.
    sheet_names : list of str, optional
        Names of the sheets to process. Default is ['CapaComp'].
    book_path : str, optional
        Path to the book in the Origin project. Default is 'OriginPlots/Summary'.
    range_col : list of int, optional
        Range of columns to process. Default is None, which means all columns are processed.
    fct_names : list of str, optional
        Names of the functions to apply. Default is ['mean', 'std'].
   
    Raises
    ------
    ValueError
        If the book does not exist in the Origin project.
    """
    assert isinstance(subtext,list), 'subtext should be a list.' 
    assert isinstance(sheet_names,list), 'sheet_names should be a list.' 
    assert 'Summary' in book_path, 'statistics_Origin cannot handle file which are not Summary.'

            
    # Open the Origin project
    op.set_show(False)
    print(utils.path_join(os.getcwd(),'OriginPlots.opju',dt='f'))
    is_open=op.open(utils.path_join(os.getcwd(),'OriginPlots.opju',dt='f'))
    print('Origin: opened = '+str(is_open))
    
    # Get the book with the specified short name
    bookname='GCPLSum'+book_name
    book=op.project.find_book(type='w',name=bookname)
    if not book:
        raise ValueError ('The book '+ bookname +' does not exist. \n Check if you encoded the short name correctly')
    
    # Process each spreadsheet that matches the subtext
    for sheet_nr,sheet_name in enumerate(sheet_names):
        sheet = op.find_sheet('w','['+bookname+']'+sheet_name)
        
        time.sleep(5)
        columns=np.array(sheet.to_list2()) 
        o=sheet.cols
        long_names=sheet.get_labels()
        units=sheet.get_labels(type_ = 'U' )
        comments=sheet.get_labels(type_ = 'C' )
        
        # Enumerature through the subtext to find column with same name.
        for i,sub in enumerate(subtext[sheet_nr]):
            n=len(subtext[sheet_nr])
            y_extract=[]
            original_units=False
            original_comment=''
            for j,col in enumerate(columns):
                # Select range (assuming one x and several y for each file in summary)
                if range_col : 
                    isin_range_col=utils.is_included(j, [range_col[0]*(n+1),(range_col[1]+n-1)*(n+1)]) 
                else:
                    isin_range_col=True
                # Extract columns
                if sub in long_names[j] and isin_range_col:
                    y_extract.append(np.where(col == '', '0',col))
                    original_comment+=comments[j]
                    if not original_units :
                        original_units=units[j]
            # Calculate the mean and standard deviation.
            y_extract=np.array(y_extract,dtype=float)
            mean = np.nanmean(y_extract.astype(float),axis=0)
            std_dev = np.nanstd(y_extract.astype(float),axis=0)
            
            # Add columns at the end.
            sheet.from_list(col = o+2*i, data = mean.tolist(), comments=fct_names[0], lname=sub,units=original_units,axis='Y')
            sheet.from_list(col = o+2*i+1, data = std_dev.tolist(), comments=fct_names[1], lname=sub,units=original_units,axis='E')
            
            
    op.lt_exec('win-s T')
    path_save=utils.path_join('OriginPlots.opju',dt='f',sort='abs_path')
    print('Origin: Saved = '+str(op.save(os.path.abspath(path_save))))
    print(path_save)
    op.exit()

        
def AddSheetOrigin(location,filename,xOr,yOr,comments,foldername='tech',bookname='exp',sheet=0,ShNam='Data',shiftCol=0,shiftRow=0,lname='',colNames=None, typedata='columns'):
    """
    Takes as argument the vectors and produce a nice originlab graph.

    Parameters
    ----------
    location : str
        Location where the file will be stored.
    filename : str
        Name of the file to be created. The columns are stored in the first dimension.
    xOr : np.ndarray
        The X axis of the graph.
    yOr : np.ndarray
        The Y axis of the graph.
    comments : list of str
        A list containing the comments for the origin file.
    templatefile : str, optional
        The name of the file to be used as a template for the creation of the worksheets. Default is 'templates.opju'.
    foldername : str, optional
        Designates the technique to be plotted (ex: GCPL). Default is 'tech'.
        Used to find the template for the graph, in combination with the sheetname.
    bookname : str, optional 
        Both the shortname and longname of the book to be open . Default is 'exp'.
        - Short name should be unique
        - the function oricompa is used to remove special characters from the strings to use them as short name in origin.
        - "book short name" is assembled as the 13 first letters of bookname after removal of special char.
        - "sheet name" has no limitation on length.
        - "graph short name" is defined as oricompa(bookname)[:n] + oricompa(ShNam)[:m+6-n] + str(sheetindex) with n and m being 6 and 5 at most, respectively.
    sheet : int, optional
        The sheet number. Default is 0.
    ShNam : str, optional
        The sheet name. Default is 'Data'. 
        Defines the name of the graph to be plotted as ShNam +  str(sheet)
        Also used to define the template for the grap.
    shiftCol : int, optional
        The shift column. Default is 0.
    lname : str, optional
        The long name. Default is ''.
    typedata : str,  
        "columns" for a set of columns, metrics if xOr and yOr is 1 row to be added.

    Returns
    -------
    None.
    """
    sname=oricompa(bookname, splitminus=False)[:13]
    book0=openFileOrigin(path=location,filename=filename,newfolder=foldername,bookname=sname, booklname=lname)
    
    localsheet, _ = openSheetOrigin(book0,oricompa(ShNam, splitminus=False),sname,typeBook='w')
    
    # Check the dimensions and put the largest dimension as vertical.
    # If the typedata is row, it will do the reverse
    def find_dims(Z,td='columns'):
        if isinstance(Z,list):
            dim_x=len(Z) 
            dim_y=None #a list can store arrays of various length.
        elif isinstance(Z,np.ndarray):
            try:
                x1=np.shape(Z)[0]
                x2=np.shape(Z)[1]
                # Columns are the smallest dimension. 
                if x1>x2 : 
                    (dim_x,dim_y)=(x2,x1)
                    if td == 'columns':
                        Z=Z.T # first and longest dimensions=columns
                elif x1<x2:  
                    (dim_x,dim_y)=(x1,x2)
                    if td == 'row':
                        Z=Z.T # first and longest dimensions=columns
                elif x1==x2==1 :
                    (dim_x,dim_y)=(x1,x2)
                    Z=Z.T
                else:
                    raise ValueError('One dimension of the ndarray should be bigger.')      
            except IndexError: #if 1 column of 0 dimension np.shape raises an IndexError.
                Z=np.array([Z])    
                dim_x=1
                dim_y=np.shape(Z)[1]
        else: raise ValueError('xOr is neither a ndarray nor a list.')        
        return Z, dim_x, dim_y

    if typedata == 'row':
        assert isinstance(xOr,np.ndarray) and isinstance(yOr,np.ndarray)
        (xOr,colsX,n_x)=find_dims(xOr)
        (yOr,n_y,colsY)=find_dims(yOr,td='row')
    elif typedata == 'columns' : 
        (xOr,colsX,n_x)=find_dims(xOr)
        (yOr,colsY,n_y)=find_dims(yOr)
    else :
        raise ValueError 


    # Create column on the sheet.
    localsheet.cols=colsX+colsY+(colsX+colsY)*shiftCol    

    param = get_plot_param_from_excel(foldername, sheet, ShNam)
    if param is not None:
        makegraph = param.get('makegraph', False)
        Exportsheet = param.get('Exportsheet', False)
    
    if Exportsheet or makegraph:
        # Caution: for the lists encoded in excel, you have to add the strip and split to make them list in python.
        # Remove the square brackets and split the string on comma
        # Also convert the '' to ""
        graphTitles = eval(param['graphTitles'])
        assert isinstance(graphTitles, list), f"Expected a list, but got {type(graphTitles).__name__}"
        if not colNames : colNames = graphTitles
        colUnit = eval(param['colUnit'])
        assert isinstance(colUnit, list), f"Expected a list, but got {type(colUnit).__name__}"
        alternate = param.get('alternate',False)
        nr_yerr =  round(param.get('nr_yerr', 0))
        nr_yy = round(param.get('nr_yy', 0))

    if makegraph:
        plotcolors=param.get('colors', None)
        templategraph = param.get('templategraph', None)
        reconstructlegend = param.get('ReconstructLegend',False)
        addtoexisting = param.get('addtoexisting', False)
        xlimits = param.get('xlimits', None)
        ylimits = param.get('ylimits', None)
        y2limits = param.get('y2limits', None)
        typePlot= param.get('typePlot', "?")
        Spaceplot = param.get('Spaceplot', False)
        colorincarg= param.get('colorincarg', -1)
        
    # Comment is legend
    # colName is axis title  
    
    # Graph
    if makegraph:
        if not templategraph: templategraph=utils.path_join('u_templates',dt='f')
        else: templategraph=utils.path_join('u_templates',templategraph,dt='f')

        # addtoexisting define what to do when the graph already exists
        # add means that the new graph will be added to the existing
        if addtoexisting and shiftCol>0 :
            add=True
        else :
            add=False
            
        # Find/create graph
        lnam=oricompa(bookname,nospecialchars=False)+' '+oricompa(ShNam,nospecialchars=False)
        sheetindex = localsheet.index()
        # Set the shortname with an acceptable length (<13 characters) based on the bookname and sheetname
        n=min(len(oricompa(bookname)),5) # 5 Characters max from the bookname
        m=min(len(oricompa(ShNam)),6) # 6 Characters max from the bookname
        snam=oricompa(bookname)[:n] + oricompa(ShNam)[:m+6-n] + str(sheetindex)

        graph=openGraphOrigin(snam,templategraph,longname=lnam, replace=not add)
        graph.name=snam
        if graph.name != snam :
            raise ValueError(f'The graph {snam} short name could not be set properly. The graph name is {graph.name}')
        
        # Select graph layer
        gl1=graph[0]
        if nr_yy>0:
            gl2=graph[1]
    
    """
    >>> Put the data in the spreadsheet and, if makegraph, also plot in a graph.
    >> Case 1: several XY data columns to plot separately (i.e. cyclic voltammetry data, XYXYXYXY)
    ___________________________________________________________________________"""
    if colsX==colsY and alternate:
        for i in range(0,colsX):
            if Exportsheet :
                # The number of column to skip between each plot
                Spaceplot=round(Spaceplot)
                assert isinstance(Spaceplot,int) and Spaceplot > 0
                # 2 cases for the comment: automatic numbering (i.e. cycles) or 2 columns (i.e. overpotential).
                comix = comments[i] if (Spaceplot > 0) else comments[0]
                comiy = comments[i] if (Spaceplot > 0) else comments[1]
                # Export
                localsheet.from_list(col = 2*i+2*shiftCol, data = xOr[i].tolist(), comments=comix, lname=colNames[0],units=colUnit[0],axis='X',start=shiftRow)
                localsheet.from_list(col = 2*i+1+2*shiftCol, data = yOr[i].tolist(), comments=comiy, lname=colNames[1],units=colUnit[1],axis='Y',start=shiftRow)
                # Plot, if the column correspond to Spaceplot
                if makegraph and i%Spaceplot == 0:
                    # X-axis for all column but the YY ones.
                    plot = gl1.add_plot(localsheet,coly=2*i+2*shiftCol+1,type=typePlot)

        if makegraph:
            # Rescale, group, add colors and set increment
            gl1.rescale()
            gl1.group()
            if plotcolors :
                if not plotcolors == 'template':
                    plot.colormap=plotcolors
            else:
                plot.colormap='Magma_rev'
            if colorincarg < 0 :
                plot.colorinc= 2 if colsY > 2 else 1
            else:
                plot.colorinc= round(colorincarg)
        """
    >> Case 2: 1 set of data column to plot separately (i.e. capacity after cycling)
    ___________________________________________________________________________"""
    else:
        for i in range(0,colsX):
            if Exportsheet:
                xi=xOr[i]
                if isinstance(xi,np.ndarray):
                    xi=xi.tolist()
                localsheet.from_list(col = i+(colsX+colsY)*shiftCol, data = xi, comments=comments[i], lname=colNames[i],units=colUnit[i],axis='X',start=shiftRow)
        for j in range(0,colsY):   
            if Exportsheet :
                if(j<colsY-nr_yerr-nr_yy or j>=colsY-nr_yy):
                    axtype='Y'
                else:
                    axtype='E'
                yj=yOr[j]
                if isinstance(yj,np.ndarray):
                    yj=yj.tolist()
                localsheet.from_list(col = j+colsX+(colsX+colsY)*shiftCol, data = yj, comments=comments[j+colsX], lname=colNames[j+colsX],units=colUnit[j+colsX],axis=axtype,start=shiftRow)
            if makegraph:
                # In case second axis
                if j<colsY-nr_yy: cl = gl1
                else: cl=gl2
                # In case error column
                if not (j<colsY-nr_yerr-nr_yy or j>=colsY-nr_yy):
                    cl.remove_plot(cl.plot_list()[-1])
                    plot = cl.add_plot(localsheet,coly=j+colsX+(colsX+colsY)*shiftCol-1, colyerr=j+colsX+(colsX+colsY)*shiftCol,type=typePlot)
                    # errplot=cl.plot_list()[-1]
                    # errplot.type=typePlot
                else:
                    plot = cl.add_plot(localsheet,coly=j+colsX+(colsX+colsY)*shiftCol,type=typePlot)
        # Rescale, group, add colors and set increment list for the plot.
        if makegraph:
            # Rescale and group
            gl1.rescale()
            gl1.group()
            if nr_yy>0:
                gl2.rescale()
                gl2.group()
                
            # Set color and color increment
            pl1=graph[0].plot_list()
            for plot in pl1:
                if plotcolors:
                    plot.colormap=plotcolors                
                else:
                    plot.colormap='Candy'            # set colormap to candy
                if not pd.isnull(colorincarg) :
                    # set color increment to "by one"
                    plot.colorinc = 1
                else:
                    plot.colorinc= colorincarg
            if nr_yy>0:
                pl2=graph[1].plot_list()
                for plot in pl2:
                    plot.colormap='Candy'            # set colormap to candy
                    if colorincarg < 0 :
                        # set color increment to "by one"
                        plot.colorinc = 1
                    else:
                        plot.colorinc= round(colorincarg)

    
    if makegraph:
        # Set the limits to the one defined in Origin_parameters_plot
        # X
        if not pd.isnull(xlimits): 
            xlimits=ast.literal_eval(xlimits)
            gl1.set_xlim(begin=round(xlimits[0],1),end=round(xlimits[1],1),step=round(xlimits[2],1))
        # Y
        if not pd.isnull(ylimits): 
            ylimits=ast.literal_eval(ylimits)
            gl1.set_ylim(begin=round(ylimits[0],1),end=round(ylimits[1],1),step=round(ylimits[2],1))        
        # YY
        if nr_yy > 0 and not pd.isnull(y2limits): 
            y2limits=ast.literal_eval(y2limits)
            gl2.set_ylim(begin=round(y2limits[0],1),end=round(y2limits[1]),step=round(y2limits[2],1))        

        # Set the axis titles
        axX=gl1.axis('x')
        axX.title = graphTitles[0] + ' ' + colUnit[0]
        axY=gl1.axis('y')
        axY.title = graphTitles[1] + ' ' + colUnit[1]
        if nr_yy > 0 :
            axY2=gl2.axis('y')
            axY2.title = graphTitles[colsY-nr_yy] + ' ' + colUnit[colsY-nr_yy]
        
        # Reconstruct legend
        if reconstructlegend :
            graph[0].lt_exec('legend -r')
        graph.set_int('aa', 1)
    
    # Adding a line
    # https://my.originlab.com/forum/topic.asp?TOPIC_ID=48006
        
    return closeFileOrigin(book0, location, filename)
    
def test_AddSheetOrigin(test="hist"):
    # A sheet with a large column of points (dim=shortest).
    if test == "std":
        """
        >> Test plot with add to existsing 
        ________________________________________________________________________"""
        # Define the parameters for the test
        location = os.getcwd()
        filename = 'OriginPlots.opju'
        xOr = np.random.rand(10)
        yOr = np.random.rand(10)
        comments = ['test_comments']*2
    
        # Call the function with the test parameters
        result = AddSheetOrigin(location, filename, xOr, yOr, comments, foldername='test', bookname='test_bookname', sheet=0, ShNam='test1_sh--name', shiftCol=0, lname='test_lna--me')
    
        # Define the parameters for the test
        location = os.getcwd()
        filename = 'OriginPlots.opju'
        xOr = np.random.rand(10)
        yOr = np.random.rand(10)
        comments = ['test_comments']*2
    
        # Call the function with the test parameters
        # result = AddSheetOrigin(location, filename, [xOr], [yOr], comments, foldername='test', bookname='test_bookname', sheet=0, ShNam='test1_sh--name', shiftCol=1, lname='test_lna--me')
        # result = AddSheetOrigin(location, filename, [xOr], [yOr], comments, foldername='test', bookname='test_bookname', sheet=0, ShNam='test1_sh--name', shiftCol=1, lname='test_lna--me')
    
        """
        >> Test plot without add to existsing 
        ________________________________________________________________________"""
        # Define the parameters for the test
        location = os.getcwd()
        filename = 'OriginPlots.opju'
        xOr = np.random.rand(10)
        yOr = np.random.rand(10)
        comments = ['test_comments']*2
    
        # Call the function with the test parameters
        # result = AddSheetOrigin(location, filename, [xOr], [yOr], comments, foldername='test', bookname='test_bookname', sheet=0, ShNam='test2_sh--name', shiftCol=0, lname='test_lna--me')
    
        # Define the parameters for the test
        location = os.getcwd()
        filename = 'OriginPlots.opju'
        xOr = np.random.rand(10, 1)
        yOr = np.random.rand(10, 1)
        comments = ['test_comments']*2
    
        # Call the function with the test parameters
        # result = AddSheetOrigin(location, filename, [xOr], [yOr], comments, foldername='test', bookname='test_bookname', sheet=0, ShNam='test2_sh--name', shiftCol=1, lname='test_lna--me')
        # result = AddSheetOrigin(location, filename, [xOr], [yOr], comments, foldername='test', bookname='test_bookname', sheet=0, ShNam='test2_sh--name', shiftCol=1, lname='test_lna--me')
    
    
    
        # Check the result
        assert result, "Failed: AddSheetOrigin did not write the code."

        print('All standard tests passed.')
    # A sheet with a certain number of points.
    elif test == "hist":
        # >>> Test an histgramm
        # Define the parameters for the test
        location = os.getcwd()
        filename = 'OriginPlots.opju'
        xOr = np.random.rand(10)
        yOr = np.random.rand(10)
        comments = ['test_comments']*2
    
        # Call the function with the test parameters
        # result = AddSheetOrigin(location, filename, xOr, [yOr], comments, foldername='test', bookname='test_bookname', sheet=0, ShNam='test3_sh--name', shiftCol=0, lname='test3_lna--me')

        
        # Define the parameters for the test
        location = os.getcwd()
        filename = 'OriginPlots.opju'
        xOr = np.array([[1]],dtype=float)
        xOr2 = np.array([[2]],dtype=float)
        yOr = np.array([[1,0.2]],dtype=float)
        yOr2 = np.array([[1.2,0.1]],dtype=float)
        comments = ['test_comments']*3
    
        # Call the function with the test parameters
        result = AddSheetOrigin(location, filename, xOr, yOr, comments, foldername='test', bookname='test_bookname', sheet=0, ShNam='test3', shiftCol=0, lname='test_hist--me', typedata='row')
        result = AddSheetOrigin(location, filename, xOr2, yOr2, comments, foldername='test', bookname='test_bookname', sheet=0, ShNam='test3', shiftCol=0, shiftRow=1, lname='test_hist--me', typedata='row')
    
        
    openOriginAfterExecution()
    

def AddDFOrigin(location: str, filename: str, df: pd.DataFrame, foldername: str = 'Summary', bookname: str = 'List', ShNam: str = 'Data', lname: str = '') -> None:
    """
    Adds a panda DataFrame to a specified sheet in an Origin file.

    Parameters:
    location (str): The path to the Origin file.
    filename (str): The name of the Origin file.
    df (pd.DataFrame): The DataFrame to be added.
    foldername (str, optional): The name of the new folder. Defaults to 'Summary'.
    bookname (str, optional): The name of the book. Defaults to 'List'.
    ShNam (str, optional): The name of the sheet. Defaults to 'Data'.
    lname (str, optional): The long name of the book. Defaults to ''.

    Returns:
    None
    """
    sname=oricompa(bookname,splitminus=False)[:13]
    book0=openFileOrigin(path=location,filename=filename,newfolder=foldername,bookname=sname, booklname=lname)
    
    localsheet, _ = openSheetOrigin(book0,ShNam,bookname,typeBook='w')
    
    localsheet.from_df( df, head = 'C')
    
    closeFileOrigin(book0,location,filename)
    
    print(f'The dataframe was sucessfully saved to the sheet {ShNam}')
    
def test_AddDFOrigin():
    location = os.getcwd()  # Set this to your desired location
    filename = 'OriginPlots.opju'  # Set this to your desired filename
    foldername = 'test_df'
    bookname = 'BkA'
    ShNam = 'ShE'
    lname = 'Longname'

    # Test with an empty dataframe
    df = pd.DataFrame()
    AddDFOrigin(location, filename, df, foldername=foldername, bookname=bookname, ShNam=ShNam+str(1), lname=lname)
    

    # Test with a dataframe with no labels
    df = pd.DataFrame([[1, 2], [3, 4]])
    AddDFOrigin(location, filename, df, foldername=foldername, bookname=bookname, ShNam=ShNam+str(2), lname=lname)
    

    # Test with a dataframe with NaN values
    df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
    AddDFOrigin(location, filename, df, foldername=foldername, bookname=bookname, ShNam=ShNam+str(3), lname=lname)
    AddDFOrigin(location, filename, df, foldername=foldername, bookname=bookname, ShNam=ShNam+str(3), lname=lname)
        
    op.exit()
    
    openOriginAfterExecution()

def closeFileOrigin(book0, loc, filename):
    folder_path=loc
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # Hide workbooks and tile graphs
    book0.show=False
    folder = op.pe.active_folder()
    graphs = folder.pages('g')
    for i, graph in enumerate(graphs):
        graph.show = False
    op.lt_exec('win-s T')
    issaved=op.save(os.path.abspath(filename))
    print('Origin: Saved = '+ str(issaved))
    op.exit()
        
    return issaved

def openOriginAfterExecution():
    # Close python origin
    op.exit()
    
    # Specify the directory where your Origin files are located
    directory = os.getcwd()
    
    # Get a list of all files in the directory
    files = os.listdir(directory)
    
    # Filter the list to include only .opju files
    files = [f for f in files if f.endswith('.opju')]
    
    # Sort the list by modification time so the last file is at the end
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    
    # Get the path to the last file
    last_file = os.path.join(directory, files[-1])
    
    subprocess.Popen(['C:\Program Files\OriginLab\Origin 2024\Origin64.exe', last_file])
    
    print('1')
    t0=time.time()
    t1=t0
    isfound=False
    
    # Get a list of all windows with "Origin" in their title
    while (not t1 - t0 > 15 and not isfound ) :
        origin_windows = pyautogui.getWindowsWithTitle("Origin 2024")
        t1=time.time()
        if len(origin_windows)>0 :
            isfound = True
            
            time.sleep(1)

            print('Opening origin, please check the execution')
        
            # If there is at least one such window, bring it to the foreground
            origin_windows[0].activate()
            # Maximize the window
            origin_windows[0].maximize()

    if not isfound :
        print('Origin did not open')

def AddImageOrigin(image: np.ndarray, foldername: str, bookname: str, sheetname: str, booklname: Optional[str]=None, test: bool=False, log=False):
    """
    Adds an image to an Origin project.

    Args:
        image (np.ndarray): The image to add. Should be a square 2D or 3D array.
        foldername (str): The name of the folder in the Origin project.
        bookname (str): The name of the workbook in the Origin project.
        sheetname (str): The name of the sheet in the workbook.
        booklname (Optional[str], optional): The long name of the workbook. Defaults to None.
        test (bool, optional): If True, returns the image data from the Origin matrix for testing. Defaults to False.
        log (bool, optional): whether this image should be a logarithmic plot or 

    Raises:
        ValueError: If the image is not 2D or 3D, or if it's a 3D image with more than one channel.
        AssertionError: If the image is not square.

    Returns:
        np.ndarray or None: If test is True, returns a tuple of the image data from the Origin matrix and the normalized image. Otherwise, closes the Origin file and returns None.
    """
    
    # Open Origin project
    compabknam=oricompa(bookname)
    sname=oricompa(compabknam)[:min(13,len(compabknam))]
    if booklname : booklname=oricompa(booklname,nospecialchars=False)
    loc=os.getcwd()
    file='OriginPlots.opju'
    matrix=openFileOrigin(path=loc,filename=file,newfolder=oricompa(foldername,nospecialchars=False),bookname=sname, booklname=booklname, typeBook='m')
    
    msheet,index=openSheetOrigin(matrix,sheetname,sname,'m')

    param = get_plot_param_from_excel(foldername, None, sheetname, typeBook='m')
    
    plotimage = param.get('plotimage', None)
    if not plotimage : return None
    
    templategraph = param.get('templategraph',None)
    addtoexisting = param.get('addtoexisting',False)
    pixel_size = param.get('pixel_size',1)

    # >> SET-UP DIMENSIONS
    # > Assert the image shape, add a dimension if there is only 1.
    if len(image.shape) < 2 :
        raise ValueError('The image should have more than one dimension')
    elif len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)
    elif len(image.shape) > 2 and image.shape[0]>1:
        raise ValueError('AddImageOrigin is not configurated for image stacks')
    n=image.shape[1]
    m=image.shape[2]
    assert n == m, 'Please pass a square image as argument'
    
    # >> MASK
    # > Remove extreme values and segmented zones
    [image],msksegm=utils.image_well_defined(image,axis=0)
    if np.sum(msksegm) == 0:
        raise ValueError('AddImageOrigin; the image as only non valid value.')
    
    # > Flatten the mask
    fmsk = msksegm==1
    fi = image
    fimsk = fi[fmsk]
    
    # >> NORMALISATION
    # > Calculate the min, max, median and assert result.
    med2Im = np.median(fimsk)
    minIm,maxIm = np.percentile(fimsk,[2,98])
    if minIm == maxIm :
        print('AddImageOrigin: caution, the minimum and maximum have the same value.')
    assert minIm <= maxIm and minIm <= med2Im and med2Im <= maxIm
    if 5*med2Im < maxIm: maxIm = 5*med2Im
    if 0 < minIm: minIm = 0
        
    # Export the image data to an Origin matrix
    if not log :    
        normIm_with_nan = np.where(msksegm,image,np.nan)
        msheet.from_np(normIm_with_nan)
    else:
        normIm=np.log10(utils.image_positivehist(image,normalize='absolute', ext_vals=[minIm,med2Im], mask=msksegm)+minIm)
        normIm_with_nan = np.where(msksegm,normIm,np.nan)
        msheet.from_np(normIm_with_nan)
    
    # >> PLOT IMAGE IN ORIGIN
    # > Set the pixel size 97nm/pixel
    px_size = pixel_size
    msheet.xymap = 0, n*px_size, 0, m*px_size
    
    # > Find/create graph
    if len(str(index)) > 1 :
        raise ValueError('Cannot have more than 9 sheet and plot.')
    lnam=bookname+' '+sheetname+str(index)
    # > Set the shortname with an acceptable length (<13 characters)
    n=min(len(oricompa(bookname)),6)
    m=min(len(oricompa(sheetname)),6)
    snam=oricompa(bookname)[:n]+oricompa(sheetname)[:m+5-n]+str(index)
    graph=openGraphOrigin(snam,utils.path_join('u_templates',templategraph, dt='f'),longname=lnam)
    graph.name=snam

    # > Delete previous execution
    # > "addtoexisting" define what to do when the graph already exists
    if addtoexisting :
        add=True
    else: 
        add=False
    if not add:
        pl=graph[0].plot_list()
        for plot in pl:
            graph[0].remove(plot)
    
    # > Add layer
    lay1 = graph[0]
    plot=lay1.add_mplot(msheet, 0, 1, 2, type = 220)
    lay1.rescale()
    
    # > Adjust scale
    nrsteps=15
    nrdecimals=2
    if not log:
        # new_minIm,new_maxIm= 0, (maxIm-minIm)/(med2Im-minIm)
        new_minIm,new_maxIm= minIm, maxIm
    elif log:
        new_minIm,new_maxIm= 0, np.log10((maxIm-minIm)/(med2Im-minIm)+minIm)
    levels=np.round(np.linspace(new_minIm, new_maxIm,nrsteps),decimals=nrdecimals)
    z = plot.zlevels
    z['minors'] = nrsteps
    z['levels'] = levels.tolist()
    plot.zlevels = z
    
    # >> PLOT HISTOGRAMM IN ORIGIN
    # > Find/create graph
    if len(str(index+1)) > 1 :
        raise ValueError('Cannot have more than 9 sheet and plot.')
    snamhist='H'+oricompa(bookname)[1:n]+oricompa(sheetname)[:m+5-n]+str(index)
    templatehist=templategraph.replace('Image','Hist')
    graph2=openGraphOrigin(snamhist,utils.path_join('u_templates',templatehist, dt='f'),longname=snamhist,replace=True)
    graph2.name=snamhist

    lay2 = graph2[0]
    plot=lay2.add_mplot(msheet, 0, type = 219)
    lay2.rescale()

    # If test is set to True, extract back the matrix from to origin to test that it was imported correctly.
    if test:
        return msheet.to_np3d(), normIm
    else:
        return closeFileOrigin(matrix, loc, file)

def test_AddImageOrigin(testall=True,returnmatrix=False):
    if testall:
        try:
            # Case 1: Empty image sets
            im = np.array([])
            result = AddImageOrigin(im, 'test/testsubfolder', 'book', 'testshname1' ,test=returnmatrix)
        except ValueError:
            pass
    
    
        # Case 2: Single-pixel images (9 pixels here)
        im = np.random.rand(8, 8)
        result = AddImageOrigin(im, 'test/testsubfolder', 'book', 'testshname2' ,test=returnmatrix)
        if returnmatrix:
            assert np.all(result[0] == result[1]), "Failed on case 2: Single-pixel images"
    
        # Case 3: Images with all pixels the same
        im = np.full((10, 10), 5)
        result = AddImageOrigin(im, 'test/testsubfolder', 'book', 'testshname3' ,test=returnmatrix)
        if returnmatrix:
            assert np.all(result[0] == result[1]), "Failed on case 3: Images with all pixels the same"
    
        # Case 4: Images with extreme pixel values
        im = np.random.rand(10, 10) * 1e6
        result = AddImageOrigin(im, 'test/testsubfolder', 'book', 'testshname4' ,test=returnmatrix)
        if returnmatrix:
            assert np.all(result[0] == result[1]), "Failed on case 4: Images with extreme pixel values"
    
        try:
            # Case 5: Images with NaN or Inf values
            im = np.full((10, 10), np.nan)
            result = AddImageOrigin(im, 'test/testsubfolder', 'book', 'testshname5' ,test=returnmatrix)
        except ValueError:
            pass

        
    # Case 6, a test image
    path= r'D:\Documents\a PSI\Data\Data analysis\spyder\2107_Progress_work\0410_SIM beamline 1\PEEM_py\Uncycled\Ni_uncycled\2_energies'
    testim2 = np.array(Image.open(utils.path_join(path,'i211025_009_851_3_undistrdd/i211025_009#001.tif',dt='f')))
    testim1 = np.array(Image.open(utils.path_join(path,'i211025_012_853_1_undistrdd/i211025_012#001.tif',dt='f')))

    testim1,testim2=utils.images_outliers2nan(testim1,testim2)
    
    result = AddImageOrigin(testim1, 'test/testsubfolder', 'book', 'testshname6' ,test=returnmatrix, log= True)
    if returnmatrix:
        assert np.all(result[0] == result[1]), "Failed on case 6: test image not properly returned."
        
    if not returnmatrix:
        openOriginAfterExecution()

def get_plot_param_from_excel(foldername: str, sheet: int, ShNam: str, typeBook: str ='w') -> Optional[Dict[str, str]]:
    """
    This function reads data from an Excel file and filters it based on the provided foldername, sheet, and ShNam.

    Parameters:
    foldername (str): The name of the folder.
    sheet (str): The name of the sheet.
    ShNam (str): The name of the ShNam.

    Returns:
    dict: A dictionary containing the first matching row of data. If no matching data is found, a NotImplementedError is raised.
    """
    if typeBook == 'w' :
        namesheet = 'Param_workbook'
    elif typeBook == 'm' :
        namesheet = 'Param_matrix'
    # Read the data from the Excel file
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_excel(utils.path_join(current_script_directory,'Origin_parameters_plot.xlsx',dt='f'),sheet_name=namesheet)

    # Remove rows that contain '--' in any column
    df = df[~df.apply(lambda row: row.astype(str).str.contains('--').any(), axis=1)]

    # Filter the data based on the foldername, sheet, and ShNam
    matcharg = []
    values = [foldername, sheet, ShNam]
    columns = ['foldername', 'sheet', 'sheetname']
    matcharg = []
    for arg,col in zip(values,columns):
        colmatch=[]
        for _, row in df.iterrows():
            valcol=row[col]
            if pd.isnull(valcol):
                condition = True
            elif isinstance(valcol,int) or isinstance(valcol,float) :
                condition = (valcol == arg)
            elif isinstance(valcol,str) : 
                if valcol == 'testshname*':
                    pass
                condition = valcol == arg or '*' in valcol and valcol.replace('*', '') in arg
            else:
                raise ValueError('The origin_parameters_plot.xlsx should only contain int, float, None or string in the column Foldername, sheet and ShNam')
            colmatch.append(condition)  
        matcharg.append(colmatch)
    param = df[np.array(np.all(matcharg,axis=0))]
    
    # If there is no matching data, return None
    if param.empty:
        raise NotImplementedError(f'Error: you tried exporting to origin an experiment that is not configurated: \n Folder: {foldername} \n Sheet: {sheet} \n Sheet Name: {ShNam}')   

    # Otherwise, return the first matching row as a dictionary
    return param.iloc[0].to_dict()

def test_get_plot_param_from_excel(args):
    # Call the function with the test parameters
    result = get_plot_param_from_excel(args[0], args[1], args[2])
    
    # Check that the result is a dictionary
    assert isinstance(result, dict), "Result should be a dictionary"
    
    varname=['Sample','sheet','sheetname']
    # Check that the input arguments are present in the dictionary
    for labelassert, arg in zip(varname, args):
        if pd.isna(result[labelassert]):
            assert pd.isnull(result[labelassert]), f"Expected {labelassert} to be 'nan' for input 'None'"
        elif labelassert == 'Sample':
            assert result[labelassert].strip('*') in result[labelassert], f"Expected {labelassert} to be equal to input"
        elif labelassert == 'sheet':
            assert result[labelassert] == arg , f"Expected {labelassert} to be equal to input"
        elif labelassert == 'ShNam':
            assert result[labelassert].strip('*') in result[labelassert], f"Expected '*' to be in {labelassert} for input {arg}"
        print('Returns '+ str(labelassert) + ' :' + str(result[labelassert]) + ' for input ' + str(arg))
    assert ~(pd.isnull(pd.Series(result)[varname])).all(), f"{varname} are all set to None"    
    print('test_get_plot_param_from_excel ran without problem')

def oricompa(text: str,splitminus:bool=True,nospecialchars:bool=True) -> str:
    """
    This function takes a string as input and removes underscores, spaces, parentheses, and hyphens to make the
    text compatible with the short names in origin.

    Parameters:
    text (str): The input string.
    splitminus (bool): if set to True, the input text will by cut a the first double minus, i.e. "--".

    Returns:
    str: The processed string with certain characters removed.
    """
    assert isinstance(text,str)
    
    # Split in the iphen.
    if text and splitminus :
        text = text.split('--', 1)[0]
        
    # Remove problematic special characters 
    if nospecialchars :
        text=text.replace('_','')
        text=text.replace(' ','')
        text=text.replace('(','')
        text=text.replace(')','')
        text=text.replace('-','')
    else:
        text=text.replace('--','')
    
    # Remove leading zeros to the text
    # > Split the text into parts consisting of digits and non-digits
    parts = re.split('(\D+)', text)
    # > Remove leading zeros from parts that are digits, unless the part is '0'
    parts = [part.lstrip('0') if part.isdigit() and part != '0' else part for part in parts]
    # > Combine the parts back into a single string
    text = ''.join(parts)

    return text

if __name__ == '__main__':
    # Interfaces with origin/excel.
    test_get_plot_param_from_excel(['PEEM_2i_sample_details', None, 'C65Regr']) #
    # test_get_plot_param_from_excel(['GCPL',0,'Data'])
    # test_AddSheetOrigin()
    # test_AddDFOrigin()
    
    # openOriginAfterExecution()
    
    # DEV - with export to origin
    # test_AddImageOrigin(testall=False)
    # Test - all conditions without export to origin
    # test_AddImageOrigin(testall=True, returnmatrix=False)
    
    print(os.getcwd())