#
# ppcTools.py
#
# part of mcmcTOFfitting
# this is a set of utilities that assist in the sampling of a posterior distr.
# this aides in producing a 'posterior predictive check' or ppc
#
# armed with a posterior we get from MCMC runs, we want to produce some fake
# data using the posterior PDF
# we can then compare this with the input data and see how things look

from __future__ import print_function
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from scipy.integrate import (ode, odeint)
from constants.constants import (masses, distances, physics, tofWindows)
from utilities.utilities import (beamTimingShape, ddnXSinterpolator,
                                 getDDneutronEnergy, readChainFromFile,
                                 getTOF, zeroDegreeTimingSpread)
from utilities.ionStopping import ionStopping
from scipy.stats import (lognorm, skewnorm, norm)





class ppcTools:
    """Assist in the sampling of posterior distributions from MCMC TOF fits
    """
    def __init__(self, chainFilename, nSamplesFromTOF, nBins_eD = 100, nBins_x = 20, nRuns = 4):
        """Create a PPC tools object - reads in chain from file"""
        self.chain, self.probs, self.nParams, self.nWalkers, self.nSteps = readChainFromFile(chainFilename)
        
        self.nRuns = nRuns
        
        self.eD_bins = nBins_eD
        self.eD_minRange = 200.0
        self.eD_maxRange = 1200.0
        self.eD_range = (self.eD_minRange, self.eD_maxRange)
        self.eD_binSize = (self.eD_maxRange - self.eD_minRange)/self.eD_bins
        self.eD_binCenters = np.linspace(self.eD_minRange + self.eD_binSize/2,
                                    self.eD_maxRange - self.eD_binSize/2,
                                    self.eD_bins)
        self.eD_binMax = self.eD_bins - 1
        
        self.x_bins = nBins_x
        self.x_minRange = 0.0
        self.x_maxRange = distances.tunlSSA_CsI.cellLength
        self.x_range = (self.x_minRange,self.x_maxRange)
        self.x_binSize = (self.x_maxRange - self.x_minRange)/self.x_bins
        self.x_binCenters = np.linspace(self.x_minRange + self.x_binSize/2,
                                   self.x_maxRange - self.x_binSize/2,
                                   self.x_bins)
        
        # parameters for making the fake data...
        self.nEvPerLoop = nSamplesFromTOF
        self.nSamplesFromTOF = nSamplesFromTOF
        self.data_x = np.repeat(self.x_binCenters, self.nEvPerLoop)
        
        
        self.ddnXSinstance = ddnXSinterpolator()
        self.beamTiming = beamTimingShape()
        self.zeroDegTimeSpreader = zeroDegreeTimingSpread()
        
        # stopping power model and parameters
        stoppingMedia_Z = 1
        stoppingMedia_A = 2
        stoppingMedia_rho = 8.565e-5 # from red notebook, p 157
        incidentIon_charge = 1
        stoppingMedia_meanExcitation = 19.2*1e-3
        dgas_materialDef = [stoppingMedia_Z, stoppingMedia_A, stoppingMedia_rho, stoppingMedia_meanExcitation]
        #stoppingModel = ionStopping.simpleBethe( stoppingModelParams )
        self.stoppingModel = ionStopping.simpleBethe([incidentIon_charge])
        self.stoppingModel.addMaterial(dgas_materialDef)
        
            
        self.eN_binCenters = getDDneutronEnergy( self.eD_binCenters )
        
        
        tofWindowSettings = tofWindows()
        tof_nBins = tofWindowSettings.nBins
        self.tof_minRange = [tofWindowSettings.minRange['mid'], 
                        tofWindowSettings.minRange['close'], 
                        tofWindowSettings.minRange['close'],
                        tofWindowSettings.minRange['far'],
                        tofWindowSettings.minRange['production'] ]
        self.tof_maxRange = [tofWindowSettings.maxRange['mid'], 
                        tofWindowSettings.maxRange['close'], 
                        tofWindowSettings.maxRange['close'],
                        tofWindowSettings.maxRange['far'],
                        tofWindowSettings.maxRange['production'] ]
        self.tof_range = []
        for minR,maxR in zip(self.tof_minRange, self.tof_maxRange):
            self.tof_range.append((minR,maxR))
        self.tofRunBins = [tof_nBins['mid'], tof_nBins['close'], 
                   tof_nBins['close'], tof_nBins['far'], tof_nBins['production']]
                   
        self.standoffs = [distances.tunlSSA_CsI.standoffMid, 
             distances.tunlSSA_CsI.standoffClose,
             distances.tunlSSA_CsI.standoffClose,
             distances.tunlSSA_CsI.standoffFar,
             distances.tunlSSA_CsI.standoff_TUNLruns]
        
        self.tofData = None
        self.neutronSpectra = None
        
        
        self.paramNames = ['$E_0$', '$f_1$', '$f_2$', '$f_3$', '$N_1$',
                           '$N_2$', '$N_3$', '$N_4$', '$N_5$']
        
        
    def generateModelData(self, params, standoffDistance, range_tof, nBins_tof, ddnXSfxn,
                      dedxfxn, beamTimer, nSamples, getPDF=False):
        """
        Generate model data with cross-section weighting applied
        ddnXSfxn is an instance of the ddnXSinterpolator class -
        dedxfxn is a function used to calculate dEdx -
        probably more efficient to these in rather than reinitializing
        one each time
        This is edited to accommodate multiple standoffs being passed
        """
        beamE, eLoss, scale, s, scaleFactor = params
        e0mean = 900.0
        dataHist = np.zeros((self.x_bins, self.eD_bins))
        
        dedxForODE = lambda x, y: dedxfxn(energy=y,x=x)
        
        nLoops = int(np.ceil(nSamples / self.nEvPerLoop))
        for loopNum in range(0, nLoops):
            eZeros = np.repeat(beamE, self.nEvPerLoop)
            eZeros -= lognorm.rvs(s=s, loc=eLoss, scale=scale, size=self.nEvPerLoop)
            checkForBadEs = True
            while checkForBadEs:
                badIdxs = np.where(eZeros <= 0.0)[0]
                nBads = badIdxs.shape[0]
                if nBads == 0:
                    checkForBadEs = False
                replacements = np.repeat(beamE, nBads) - lognorm.rvs(s=s, loc=eLoss, scale=scale, size=nBads)
                eZeros[badIdxs] = replacements
        
            
            odesolver = ode( dedxForODE ).set_integrator('dopri5').set_initial_value(eZeros)
            eD_atEachX = np.zeros(self.eD_bins)
            for idx, xEvalPoint in enumerate(self.x_binCenters):
                sol = odesolver.integrate( xEvalPoint )
                data_weights = ddnXSfxn.evaluate(sol)
                hist, edEdges = np.histogram( sol, bins=self.eD_bins,
                                             range=(self.eD_minRange,
                                                    self.eD_maxRange),
                                             weights=data_weights)
                dataHist[idx,:] += hist
                hist, edEdges = np.histogram(sol, bins=self.eD_bins,
                                             range=(self.eD_minRange,
                                                    self.eD_maxRange),
                                            density=False)
                eD_atEachX = np.vstack((eD_atEachX, hist))

        dataHist /= np.sum(dataHist*self.eD_binSize*self.x_binSize)
        e0mean = np.mean(eZeros)
        drawHist2d = (np.rint(dataHist * nSamples)).astype(int)
        tofs = []
        tofWeights = []
        eN_list = []
        eN_atEachX = np.zeros(self.eD_bins)
        for index, weight in np.ndenumerate( drawHist2d ):
            cellLocation = self.x_binCenters[index[0]]
            effectiveDenergy = (e0mean + self.eD_binCenters[index[1]])/2
            tof_d = getTOF( masses.deuteron, effectiveDenergy, cellLocation )
            neutronDistance = (distances.tunlSSA_CsI.cellLength - cellLocation +
                               standoffDistance )
            tof_n = getTOF(masses.neutron, self.eN_binCenters[index[1]],
                           neutronDistance)
            zeroD_times, zeroD_weights = self.zeroDegTimeSpreader.getTimesAndWeights( self.eN_binCenters[index[1]] )
            tofs.append( tof_d + tof_n + zeroD_times )
            tofWeights.append(weight * zeroD_weights)
            eN_list.append(weight)
            if index[1] == self.eD_binMax:
                eN_arr = np.array(eN_list)
                eN_atEachX = np.vstack((eN_atEachX, eN_arr))
                eN_list = []
                

        tofData, tofBinEdges = np.histogram( tofs, bins=nBins_tof, range=range_tof,
                                        weights=tofWeights, density=getPDF)
        return (scaleFactor * self.beamTiming.applySpreading(tofData), 
                eN_atEachX,
                eD_atEachX)
        
        
        
    def generateModelData_original(self, params, standoffDistance, range_tof, nBins_tof, ddnXSfxn,
                      dedxfxn, beamTimer, getPDF=False):
        """
        Generate model data with cross-section weighting applied
        ddnXSfxn is an instance of the ddnXSinterpolator class -
        dedxfxn is a function used to calculate dEdx -
        probably more efficient to these in rather than reinitializing
        one each time
        This is edited to accommodate multiple standoffs being passed 
        """
        e0, sigma0, skew0, scaleFactor = params
        dataHist = np.zeros((self.x_bins, self.eD_bins))
        nLoops = int(np.ceil(self.nSamplesFromTOF / self.nEvPerLoop))
        for loopNum in range(0, nLoops):
            #eZeros = np.random.normal( params[0], params[0]*params[1], nEvPerLoop )
            # TODO: get the number of samples right - doesnt presently divide across multiple loops
            try:
                eZeros = skewnorm.rvs(a=skew0, loc=e0, scale=e0*sigma0, size=self.nSamplesFromTOF)
            except ValueError:
                print('value error raised in skewnorm.rvs! params {0}, {1}, {2}, nsamples {3}'.format(e0, sigma0, skew0, self.nSamplesFromTOF))
                eZeros = norm.rvs(loc=e0, scale=e0*sigma0, size=self.nSamplesFromTOF)
            data_eD_matrix = odeint( dedxfxn, eZeros, self.x_binCenters )
            #data_eD = data_eD_matrix.flatten('K') # this is how i have been doing it..
            data_eD = data_eD_matrix.flatten()
            data_weights = self.ddnXSinstance.evaluate(data_eD)
    #    print('length of data_x {} length of data_eD {} length of weights {}'.format(
    #          len(data_x), len(data_eD), len(data_weights)))
            dataHist2d, xedges, yedges = np.histogram2d( self.data_x, data_eD,
                                                    [self.x_bins, self.eD_bins],
                                                    [[self.x_minRange,self.x_maxRange],[self.eD_minRange,self.eD_maxRange]],
                                                    weights=data_weights)
            dataHist += dataHist2d # element-wise, in-place addition
            
            # manually manage some memory 
            del dataHist2d
            del xedges
            del yedges
            del eZeros
            del data_eD_matrix
            del data_eD
            del data_weights
                
    #    print('linalg norm value {}'.format(np.linalg.norm(dataHist)))
    #    dataHist = dataHist / np.linalg.norm(dataHist)
    #    print('sum of data hist {}'.format(np.sum(dataHist*eD_binSize*x_binSize)))
        dataHist /= np.sum(dataHist*self.eD_binSize*self.x_binSize)
    #    plot.matshow(dataHist)
    #    plot.show()
        drawHist2d = (np.rint(dataHist * self.nSamplesFromTOF)).astype(int)
        tofs = []
        tofWeights = []
        eN_list = []
        eN_atEachX = np.zeros(self.eD_bins)
        eD_list = []
        eD_atEachX = np.zeros(self.eD_bins)
        for index, weight in np.ndenumerate( drawHist2d ):
            cellLocation = self.x_binCenters[index[0]]
            effectiveDenergy = (e0 + self.eD_binCenters[index[1]])/2
            tof_d = getTOF( masses.deuteron, effectiveDenergy, cellLocation )
            neutronDistance = (distances.tunlSSA_CsI.cellLength - cellLocation +
                               distances.tunlSSA_CsI.zeroDegLength/2 +
                               standoffDistance )
            tof_n = getTOF(masses.neutron, self.eN_binCenters[index[1]], neutronDistance)
            tofs.append( tof_d + tof_n )
            tofWeights.append(weight)
            eN_list.append(weight)
            eD_list.append(weight)
            if index[1] == self.eD_binMax:
                eN_arr = np.array(eN_list)
                #print('stack of EN values has shape {0}'.format(eN_atEachX.shape))
                #print('new EN array to append has shape {0}'.format(eN_arr.shape))
                eN_atEachX = np.vstack((eN_atEachX, eN_arr))
                eN_list = []
                eD_atEachX = np.vstack((eD_atEachX, np.array(eD_list)))
                eD_list=[]
            # TODO: this next line is the original way of doing this in a modern 
            # numpy distribution. should really check for version <1.6.1
            # and if lower than that, use the normed arg, otherwise use density
    #    tofData, tofBinEdges = np.histogram( tofs, bins=nBins_tof, range=range_tof,
    #                                        weights=tofWeights, density=getPDF)
        tofData, tofBinEdges = np.histogram( tofs, bins=nBins_tof, range=range_tof,
                                            weights=tofWeights, normed=getPDF)
        return (scaleFactor * self.beamTiming.applySpreading(tofData), 
                eN_atEachX,
                eD_atEachX)
        
    def generatePPC(self, nChainEntries=500):
        """Sample from the posterior and produce data in the observable space
        
        nChainEntries specifies the number of times to sample the posterior
        nSamplesFromTOF is the number of deuteron tracks to produce for a given set of parameters
        """
        generatedData = []
        generatedNeutronSpectra=[]
        generatedDeuteronSpectra=[]
        totalChainSamples = len(self.chain[:-20,:,0].flatten())
        
        # TODO: this next line could mean we repeat the same sample, i think
        samplesToGet = np.random.randint(0, totalChainSamples, size=nChainEntries)
        for sampleToGet in samplesToGet:
            modelParams = []
            for nParam in range(self.nParams):
                modelParams.append(self.chain[:,:,nParam].flatten()[sampleToGet])
                
                
            e0, loc, scale, s = modelParams[:4]
            scaleFactorEntries = modelParams[4:4+self.nRuns]
            returnedData = [self.generateModelData([e0, loc, scale, s, scaleFactor],
                                       standoff, tofrange, tofbins,
                                       self.ddnXSinstance, self.stoppingModel.dEdx,
                                       self.beamTiming, self.nSamplesFromTOF, True) for
                                       scaleFactor, standoff, tofrange, tofbins
                                       in zip(scaleFactorEntries, 
                                              self.standoffs[:self.nRuns],
                                              self.tof_range[:self.nRuns],
                                              self.tofRunBins[:self.nRuns])]
            # returned data is an array of .. a tuple (modelData, neutronSpectrum, deuteronSpectrum)
            modelData = []
            modelNeutronSpectrum = []
            modelDeuteronSpectrum=[]
            for retDat in returnedData:
                modelData.append(retDat[0])
                modelNeutronSpectrum.append(retDat[1])
                modelDeuteronSpectrum.append(retDat[2])
            generatedData.append(modelData)
            generatedNeutronSpectra.append(modelNeutronSpectrum)
            generatedDeuteronSpectra.append(modelDeuteronSpectrum)
            
        self.tofData = generatedData
        self.neutronSpectra= generatedNeutronSpectra
        self.deuteronSpectra = generatedDeuteronSpectra
        return (generatedData, 
                generatedNeutronSpectra, 
                generatedDeuteronSpectra)
        
        
        
    def sampleInitialEnergyDist(self, nSamples = 100, returnNormed=False):
        """Generate a series of samples from the chain in the form of initial deuteron energy distributions"""
        dZeroSamples = np.zeros(self.eD_bins)
        totalChainSamples = len(self.chain[:-20,:,0].flatten())
        samplesToGet = np.random.randint(0, totalChainSamples, size=nSamples)
        for sampleToGet in samplesToGet:
            # TODO: if number of parameters associated with energy distribution change, this will need to be updated as the  num is presently hardcoded
            modelParams = []
            for nParam in range(4):
                modelParams.append(self.chain[:,:,nParam].flatten()[sampleToGet])
        
            e0, loc, scale, s = modelParams[:4]
            edistrib = e0 - lognorm.rvs(s=s, loc=loc, scale=scale, size=self.nEvPerLoop)
            eHist, bins = np.histogram(edistrib, bins=self.eD_bins,
                                       range=(self.eD_minRange, 
                                              self.eD_maxRange),
                                       density=returnNormed)
            dZeroSamples = np.vstack((dZeroSamples, eHist))
        if returnNormed:
            return dZeroSamples[1:]*self.eD_binSize
        return dZeroSamples[1:]
            
            
    def makeSDEF_sia_cumulative(self, distNumber = 100):
        """Produce an MCNP SDEF card for the neutron distribution, marginalized over the cell length
        
        This SDEF card will adhere to an SI A standard"""
        # if we havent generated data yet, do it
        if self.tofData == None or self.neutronSpectra == None:
            self.generatePPC(self, 500)
            
        # first we collapse the neutron spectrum, collected by default atall X values
        neutronSpectrumCollection = np.zeros(self.eD_bins)
        for sampledParamSet in self.neutronSpectra:
            samplesAlongLength = sampledParamSet[0]
            summedAlongLength = np.sum(samplesAlongLength, axis=0)
            neutronSpectrumCollection = np.vstack((neutronSpectrumCollection, summedAlongLength))
        self.neutronSpectrum = np.sum(neutronSpectrumCollection, axis=0)
        
        
        siStrings = ['si{} a'.format(distNumber)]
        spStrings = ['sp{}'.format(distNumber)]
        for eN, counts in zip(self.eN_binCenters, self.neutronSpectrum):
            siStrings.append(' {:.3f}'.format(eN/1000))
            spStrings.append(' {:.0f}'.format(counts))
        siString = ''.join(siStrings)
        spString = ''.join(spStrings)
        self.sdef_sia_cumulative = {'si': siString, 'sp': spString}
        return self.sdef_sia_cumulative             
        
        
    def makeCornerPlot(self, paramIndexLow = 0, paramIndexHigh = None, plotFilename = 'ppcCornerOut.png'):
        """Produce a corner plot of the parameters from the chain
        
        paramIndexLow and paramIndexHigh define the upper and lower boundaries of the parameter indices to plot"""
        if paramIndexHigh == None:
            paramIndexHigh = self.nParams
        samples = self.chain[:-20,:,paramIndexLow:paramIndexHigh].reshape((-1, paramIndexHigh - paramIndexLow))
        import corner as corn
        cornerFig = corn.corner(samples, labels=self.paramNames[paramIndexLow:paramIndexHigh], 
                                quantiles=[0.15, 0.5, 0.84], show_titles=True,
                                title_kwargs={'fontsize': 12})
        cornerFig.savefig(plotFilename, dpi=300)