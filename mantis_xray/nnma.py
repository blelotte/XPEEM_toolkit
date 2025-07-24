import numpy as np
import scipy as sp
from scipy.integrate import trapz
import sys, time

# Added
from scipy.ndimage import gaussian_filter
# from sklearn.decomposition import NMF
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math

from . import analyze

class nnma():

    def __init__(self, stkdata):

        self.stack = stkdata
    
        # Set default values for various NNMA parameters
        self.kNNMA = 5            # use no. of significant PCA components, otherwise default = 5              
        self.maxIters = 10         # default no. of iterations
        self.deltaErrorThreshold = 1e-3        # default threshold for change in reconstruction error
        self.initMatrices = 'Random'  # default initialization for matrices
    
        self.lambdaSparse = 0.        # sparseness regularization param
        self.lambdaClusterSim = 0.    # cluster spectra similarity regularization param
        self.lambdaSmooth = 0.       # smoothness regularization param


    # Fill initial matrices depending on user-specified initialization method
    def setParameters(self, kNNMA = 5, 
                            maxIters = 10,
                            deltaErrorThreshold = 1e-3,
                            initMatrices = 'Random',
                            lambdaSparse = 0.,
                            lambdaClusterSim = 0.,
                            lambdaSmooth = 0.,
                            msk=None):
        
        # Set the values for various NNMA parameters
        self.kNNMA = kNNMA                  
        self.maxIters = maxIters         
        self.deltaErrorThreshold = deltaErrorThreshold        
        self.initMatrices = initMatrices  
    
        self.lambdaSparse = lambdaSparse       
        self.lambdaClusterSim = lambdaClusterSim    
        self.lambdaSmooth = lambdaSmooth     
        
        self.msk = msk

    def setRatioSpectra(self, ratiospectra, ratioimage):
        
        self.ratiospectra = ratiospectra.copy() 
        self.ratioimage = ratioimage.T.copy()
        
        print('Initialized ratio')
        

    def setClusterSpectra(self, clusterspectra, TSmap):
        
        self.clusterspectra = clusterspectra.T.copy() 
        self.TSmap = TSmap.copy().T
        
        
    def setStandardsSpectra(self, standspectra):
        
        self.standspectra = standspectra

#---------------------------------------------------------------------------------------
# Define some helper functions for NNMA
#---------------------------------------------------------------------------------------

    # Fill initial matrices depending on user-specified initialization method
    def fillInitMatrices(self):
        print ('Init method:', self.initMatrices)
        if self.initMatrices == 'Random':
            muInit = np.random.rand(self.nEnergies, self.kNNMA)
            tInit = np.random.rand(self.kNNMA, self.nPixels)
        elif self.initMatrices == 'Cluster':
            muInit = self.clusterspectra
            tInit = self.mixmatrix
                        
            self.plot_t(tInit,'Cluster spectra after Rescaling ')

            # tInit = np.random.rand(self.kNNMA, self.nPixels)  # use SVD on mu to find t instead?
        elif self.initMatrices == 'Ratio':
            muInit = self.ratiospectra
            tInit = self.mixmatrix
            
            self.plot_t(tInit,'Ratio after Rescaling ')
        elif self.initMatrices == 'FastICA':
            muInit = self.muCluster
            tInit = self.mixmatrix
                        
            self.plot_t(tInit,'Mixing matrix after Rescaling ')
                
        elif self.initMatrices == "Standards":
            muInit = self.standspectra
            tInit = np.random.rand(self.kNNMA, self.nPixels)  # use SVD on mu to find t instead?
        return muInit, tInit

    def plot_t(self,t,title):
        # Apply mask
        if isinstance(self.msk,np.ndarray):
            self.Temp[:,self.fltmsk]=t
        else:
            self.Temp=t

        for i in range(0,self.kNNMA):
            plt.figure(figsize=(10, 6))
            plt.imshow(self.Temp[i,:].reshape(self.nCols,self.nRows).T)
            plt.title(title+str(i))
            plt.show()

    # Update t
    def tUpdate(self, mu, t):
        tUpdateFactor = np.dot(mu.T, self.OD) / (np.dot(mu.T, np.dot(mu, t)) + self.lambdaSparse + 1e-9 )
        tUpdated = t * tUpdateFactor
        return tUpdated

    # Update mu
    def muUpdate(self, mu, tUpdated):
        dCostSmooth_dMu = self.calcDCostSmooth_dMu(mu)
        muDiff = mu - self.muCluster
        muUpdateFactor = np.dot(self.OD, tUpdated.T) / ( np.dot(mu, np.dot(tUpdated, tUpdated.T))
                         + self.lambdaSmooth*dCostSmooth_dMu + self.lambdaClusterSim*2*muDiff + 1e-9 )
        muUpdated = mu * muUpdateFactor + 1e-9
        return muUpdated
  
    # Calculate sparseness contribution (JSparse) to cost function
    def calcCostSparse(self, t):
        costSparse = np.sum(np.sum(np.abs(t)))
        return costSparse

    # Calculate cluster spectra similarity contribution (JClusterSim) to cost function
    def calcCostClusterSim(self, mu):
        muDiff = mu - self.muCluster
        costClusterSim = (np.linalg.norm(muDiff))**2 
        return costClusterSim
    
    # Calculate cluster map similarity contribution (JClusterSim) to cost function
    def calcCostMapSim(self, t):
        tDiff = t - self.mixmatrix
        costMapSim = (np.linalg.norm(tDiff))**2 
        return costMapSim

    # Calculate smoothness contribution (JSmooth) to cost function
    def calcCostSmooth(self, t):
        if isinstance(self.msk,np.ndarray) :
            self.Temp[:,self.fltmsk]=t
            tempT=self.Temp.reshape((-1,self.nCols,self.nRows))
            tsmooth=gaussian_filter(tempT,[0,1,1])
            return (np.linalg.norm(tempT-tsmooth))**2
        else : 
            return 0        
  
    # Calculate dJSmooth/dMu needed in mu update algorithm
    def calcDCostSmooth_dMu(self, mu):
        return 0
 
    # Calculate integral of each column in mu for normalization
    def calcMuColNorm(self, mu, muRefNorm, bkg_subs=True):
        n_calc=muRefNorm.shape[0]
        muNorm = np.ones(n_calc)
        for k in range(n_calc):
            if bkg_subs:
                muBkgSub = mu[:, k]-np.nanmean(mu[0:5, k],axis=0)
            else:
                muBkgSub = mu[:, k]
                
            # If energy is oversampled at the edge -> increase the weight of the edge on determining the sign
            sign = math.copysign(1, trapz(muBkgSub))
            muBkgSub *= sign
            
            # If there is a sharp feature (peak) on the negative side, reverse the spectra
            # The number 2 is somewhat arbitrary
            if 2*abs(np.nanmax(muBkgSub)) < abs(np.nanmin(muBkgSub)) :
                muBkgSub*=-1
            
            # Integrate the spectra on positive indexes
            posIndmu = muBkgSub > 0
            muNorm[k] = trapz(muBkgSub[posIndmu], x=self.energies[posIndmu])
            mu[:, k] =  (muBkgSub / muNorm[k]) * muRefNorm[k]
        return mu, muNorm/muRefNorm
    

    # Calculate current total cost function, fill cost function array
    def calcCostFn(self, mu, t, count):
        D = np.dot(mu, t)
        costDataMatch = 0.5 * (np.linalg.norm(self.OD - D))**2
        costSparse = self.calcCostSparse(t)
        costClusterSim = self.calcCostClusterSim(mu)
        if isinstance(self.mixmatrix,np.ndarray):
            calcCostMapSim = self.calcCostMapSim(t)
        else:
            calcCostMapSim = 0 
        costSmooth = self.calcCostSmooth(t)
        costTotal = ( costDataMatch + self.lambdaSparse*costSparse 
		  + self.lambdaClusterSim*costClusterSim + self.lambdaClusterSim*calcCostMapSim + self.lambdaSmooth*costSmooth )
        if count > 0:
            deltaError = self.costFnArray[count, 1] - self.costFnArray[count-1, 1]
        elif count == 0:
            deltaError = -1e-13
        self.costFnArray[count, :] = np.array([costTotal, deltaError, 
	          costSparse, costClusterSim, costSmooth])
        return costTotal, deltaError


#---------------------------------------------------------------------------------------
# Calculate NNMA
#---------------------------------------------------------------------------------------
    def calcNNMA(self, initmatrices = 'Random'):
      
        print ('calculating nnma')
        
        self.initMatrices = initmatrices

        self.OD = self.stack.od.copy() / 255
        self.energies = self.stack.ev
        self.nEnergies = self.stack.n_ev
        self.nCols = self.stack.n_cols
        self.nRows = self.stack.n_rows
        self.Temp=np.zeros((self.kNNMA,self.nCols*self.nRows))

        # Apply mask to OD
        if isinstance(self.msk,np.ndarray):
            self.nPixels = int(np.sum (self.msk))
            self.fltmsk=(self.msk.T).flatten()==1
            self.OD=self.OD[self.fltmsk,:]
        else:
            self.nPixels = int(self.nCols * self.nRows)
        
        # Transpose optical density matrix since NNMA needs dim NxP
        self.OD = self.OD.T.astype(np.float32)
        
        # Zero out negative values in OD matrix
        negInd = np.where(self.OD < 0.)
        if negInd: self.OD[negInd] = 0.

        self.muRecon = np.zeros((self.nEnergies, self.kNNMA))
        self.tRecon = np.zeros((self.kNNMA, self.nPixels))
        self.DRecon = np.zeros((self.nEnergies, self.nPixels))
        self.costFnArray = np.zeros((self.maxIters+1, 5))
        self.costFnArray[0, 0] = 1e99
        self.costTotal = 0    # stores current value of total cost function
        self.deltaError = 0    # stores current value of cost function change

        # If doing cluster spectra similarity regularization, import cluster spectra
        self.initmatrices=initmatrices
        if initmatrices == 'Cluster':
            self.muCluster = self.clusterspectra.astype(np.float32)
            self.muCluster,muNorm=self.calcMuColNorm(self.muCluster,np.ones(self.kNNMA), bkg_subs=False)
            self.mixmatrix = self.TSmap.astype(np.float32)
            print(self.mixmatrix.shape)

            for i in range(self.kNNMA):
                self.mixmatrix[i,:] = self.mixmatrix[i,:] - np.nanmin(self.mixmatrix[i,:])
                self.mixmatrix[i,:] = self.mixmatrix[i,:] / np.nanmax(self.mixmatrix[i,:])
            
            self.median_TSmap=np.ones(self.kNNMA)
            for i in range(0,self.kNNMA):
                TSmapi=self.mixmatrix[i,:]
                print(TSmapi)
                self.median_TSmap[i]=np.nanmedian(TSmapi[(TSmapi > 0) & ~np.isnan(TSmapi)])
            
            self.mixmatrix= self.mixmatrix/self.median_TSmap[:,np.newaxis]
            randommatrix = np.random.rand(self.kNNMA, self.nPixels) / 10
            self.mixmatrix=np.where((self.mixmatrix <= 0) | np.isnan(self.mixmatrix),randommatrix,self.mixmatrix)
            self.mixmatrix=np.where((self.mixmatrix > (self.median_TSmap*5)[:,np.newaxis]),(self.median_TSmap*5)[:,np.newaxis],self.mixmatrix)

            # Inline plot of the input ratio.
            plt.figure(figsize=(10, 6))
            plt.plot(self.energies,self.muCluster)
            plt.title('Init vectors')
            plt.show()

        elif initmatrices == 'Ratio':
            self.muCluster = self.ratiospectra
            self.muCluster,muNorm=self.calcMuColNorm(self.muCluster,np.ones(self.kNNMA), bkg_subs=False)
            
            self.median_ratio=np.ones(self.kNNMA)
            for i in range(0,self.kNNMA):
                ratioimagei=self.ratioimage[i,:]
                self.median_ratio[i]=np.nanmedian(ratioimagei[(ratioimagei > 0) & ~np.isnan(ratioimagei)])
            
            self.mixmatrix= self.ratioimage/self.median_ratio[:,np.newaxis]
            randommatrix = np.random.rand(self.kNNMA, self.nPixels) / 10
            self.mixmatrix=np.where((self.mixmatrix <= 0) | np.isnan(self.mixmatrix),randommatrix,self.mixmatrix)
            self.mixmatrix=np.where((self.mixmatrix > (self.median_ratio*5)[:,np.newaxis]),(self.median_ratio*5)[:,np.newaxis],self.mixmatrix)

            
            # Inline plot of the input ratio.
            plt.figure(figsize=(10, 6))
            plt.plot(self.energies,self.muCluster)
            plt.title('Init vectors')
            plt.show()
        elif initmatrices == 'FastICA':
            ICAstartTime=time.time()
            self.ica = FastICA(n_components=self.kNNMA)
            self.muICA=self.ica.fit_transform(self.OD)
            self.muICA,muNorm=self.calcMuColNorm(self.muICA,np.ones(self.kNNMA))
            self.muCluster=np.abs(self.muICA)
            
            # Inline plot of the ICA results.
            plt.figure(figsize=(10, 6))
            plt.plot(self.energies,self.muCluster)
            plt.title('Init vectors')
            plt.show()
            
            ICA_matrix = np.linalg.pinv(self.ica.components_).T 

            self.mixmatrix = ICA_matrix * muNorm[:,np.newaxis]
            for i in range(self.kNNMA):
                self.mixmatrix[i,:] = self.mixmatrix[i,:] - np.nanmin(self.mixmatrix[i,:])
                self.mixmatrix[i,:] = self.mixmatrix[i,:] / np.nanmax(self.mixmatrix[i,:])
            randommatrix = np.random.rand(self.kNNMA, self.nPixels) / 10
            self.mixmatrix=np.where(self.mixmatrix<=0,randommatrix,self.mixmatrix)
            ICAendTime = time.time()
            timeICA = ICAendTime - ICAstartTime
            print(f'Time ICA: {timeICA}')
            
        else:
            self.muCluster = 0.
        
        self.timeTaken = 0.    # stores time taken to complete NNMA analysis
        
        # Initialize matrices
        muInit, tInit = self.fillInitMatrices()
        self.muinit = muInit.copy()
        muCurrent = muInit
        tCurrent = tInit
        count = 0
        
        costCurrent, deltaErrorCurrent = self.calcCostFn(muCurrent, tCurrent, count)

        # Start NNMA        
        startTime = time.time()
        
        while ((count < self.maxIters) and (self.deltaError < self.deltaErrorThreshold)):
            # Store values from previous iterations before updating
            self.tRecon = tCurrent
            self.muRecon = muCurrent
            self.costTotal = costCurrent
            self.deltaError = deltaErrorCurrent

            # Now do NNMA update
            tUpdated = self.tUpdate(muCurrent, tCurrent)
            muUpdated = self.muUpdate(muCurrent, tUpdated)
            
            tUpdated = self.tUpdate(muCurrent, tCurrent)
            muUpdated = self.muUpdate(muCurrent, tUpdated)
                        
            muUpdated,Int_i=self.calcMuColNorm(muUpdated,np.ones(self.kNNMA), bkg_subs=False)
            tUpdated=tUpdated*Int_i[:,np.newaxis]
            
            # Zero out any negative values in t and mu
            negIndT = np.where(tUpdated < 0)
            tUpdated[negIndT] = 0
            highIndT = np.where(tUpdated >= 1.1)
            tUpdated[highIndT] = 1.1
            tUpdated[self.kNNMA-1,:]=np.nanmin(np.array([1-np.nansum(tUpdated[:self.kNNMA-1,:],axis=0),tUpdated[self.kNNMA-1,:]]),axis=0)
            negIndMu = np.where(muUpdated < 0)
            muUpdated[negIndMu] = 0
            
            tCurrent = tUpdated
            muCurrent = muUpdated
            
            # Calculate cost function
            costCurrent, deltaErrorCurrent = self.calcCostFn(muCurrent, tCurrent, count)

            count = count + 1
            print ('Iteration number {0}/{1}'.format(count,self.maxIters) +f' Error {np.round(costCurrent,decimals=2)}')
            
        
        # Normalize the integral so that the weights range from [0,1].
        self.muRecon=self.muRecon
        self.muRecon,IntFinal=self.calcMuColNorm(self.muRecon,np.ones(self.kNNMA), bkg_subs=False)
        self.tRecon=self.tRecon*IntFinal[:,np.newaxis]
        
        
        mean_stack=np.nanmean(self.OD,axis=1)*255
        _,IntStack=self.calcMuColNorm(mean_stack.reshape(-1,1),np.ones(1), bkg_subs=False)
        self.muRecon *= IntStack
        self.tRecon *= 255/IntStack
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.energies,self.muRecon)
        plt.title('Processed vectors')
        plt.show()
        
        self.plot_t(self.tRecon,'Thickness map reconstructed')
        
        endTime = time.time()
        self.timeTaken = endTime - startTime
        print(self.timeTaken)
        if count < self.maxIters:
            self.maxIters = count
            
        if isinstance(self.msk,np.ndarray):
            temp_tRecon=np.zeros((self.kNNMA, self.nCols*self.nRows))
            temp_tRecon[:,self.fltmsk]=self.tRecon
            self.tRecon=np.reshape(temp_tRecon,(self.kNNMA, self.nCols, self.nRows), order='F')
        else:
            self.tRecon = np.reshape(self.tRecon, (self.kNNMA, self.nCols, self.nRows), order='F')
    
        
        return 1
