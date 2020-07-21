#
# ppcTools_oneBD.py
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
#from ionStopping import ionStopping
from constants.constants import experimentConsts
from initialization import initialize_oneBD
from scipy.stats import (lognorm, skewnorm, norm)



nRuns = 3

standoff = {0: distances.tunlSSA_CsI_oneBD.standoffClose, 
            1: distances.tunlSSA_CsI_oneBD.standoffMid,
            2: distances.tunlSSA_CsI_oneBD.standoffFar}
standoffs = [distances.tunlSSA_CsI_oneBD.standoffClose, 
            distances.tunlSSA_CsI_oneBD.standoffMid,
            distances.tunlSSA_CsI_oneBD.standoffFar]
standoffName = {0: 'close', 1:'mid', 2:'far'}



tofWindowSettings = tofWindows.csi_oneBD()
##############
# vars for binning of TOF 
# this range covers each of the 4 multi-standoff runs

# tof_nBins = tofWindowSettings.nBins[standoffName[runNumber]]
# tof_minRange = tofWindowSettings.minRange[standoffName[runNumber]]
# tof_maxRange = tofWindowSettings.maxRange[standoffName[runNumber]]
# tof_range = (tof_minRange,tof_maxRange)

tof_nBins = tofWindowSettings.nBins
tof_minRange = [tofWindowSettings.minRange['close'], 
                tofWindowSettings.minRange['mid'], 
                tofWindowSettings.minRange['far']]
tof_maxRange = [tofWindowSettings.maxRange['close'], 
                tofWindowSettings.maxRange['mid'], 
                tofWindowSettings.maxRange['far']]
tof_range = []
for i in range(nRuns):
    tof_range.append((tof_minRange[i],tof_maxRange[i]))
tofRunBins = [tof_nBins['close'], 
                tof_nBins['mid'], 
                tof_nBins['far']]


################################################
# binning set up

eD_bins, eD_range, eD_binSize, eD_binCenters = initialize_oneBD.setupDeuteronBinning()
x_bins, x_range, x_binSize, x_binCenters = initialize_oneBD.setupXbinning()

eD_minRange, eD_maxRange = eD_range
x_minRange, x_maxRange = x_range

eN_binCenters = getDDneutronEnergy( eD_binCenters )

################################################

# parameters for making the fake data...
nSamples = 50000
nEvPerLoop = 10000
data_x = np.repeat(x_binCenters,nEvPerLoop)

ddnXSinstance = ddnXSinterpolator()

# TODO: make better implementation of 0deg transit time
zeroDegSpread_binCenters = np.linspace(0, 24, 7, True)
zeroDegSpread_vals = np.exp(-zeroDegSpread_binCenters/4.) /np.sum(np.exp(-zeroDegSpread_binCenters/4.))

ddnXSinstance = ddnXSinterpolator()
# 
# SET UP THE BEAM TIMING SPREAD
# 
# for one-BD data, binning is 4ns
# spread, based on TF1 fits to gamma peak, is ~4 ns
# to incorporate potential binning errors, maybe 4+4 in quadrature? (this is ~5.65)
# this is perhaps not where binning errors should be accommodated
# anyway, first arg sets timing spread sigma
#
# UPDATE: July 14 2020, improved timing in TOF spectra from CFD-like approach
# this means the sigma in the gaussians fit to gamma peak in data is smaller
# 2.7(ish) is the largest observed
beamTiming = beamTimingShape.gaussianTiming(2.7, 4)
zeroDegTimeSpreader = zeroDegreeTimingSpread()

# stopping power model and parameters
stoppingMedia_Z = 1
stoppingMedia_A = 2
#stoppingMedia_rho = 8.565e-5 # from red notebook, p 157
stoppingMedia_rho = 4*8.565e-5 # assume density scales like pressure!
# earlier runs were at 0.5 atm, this run was at 2 atm
incidentIon_charge = 1

# NOTE: this value (19.2) is from PDG
# http://pdg.lbl.gov/2016/AtomicNuclearProperties/HTML/deuterium_gas.html
# it is in ELECTRONVOLTS
# alt ref: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19840013417.pdf
# where this value is 18.2 for H_2 gas
# anyway, the programming in ionStopping assumes keV
# thus the factor of 1e-3
stoppingMedia_meanExcitation = 19.2 * 1e-3 # 

stoppingModelParams = [stoppingMedia_Z, stoppingMedia_A, stoppingMedia_rho,
                       incidentIon_charge, stoppingMedia_meanExcitation]
stoppingModel = ionStopping.simpleBethe( stoppingModelParams )

    

eD_stoppingApprox_binning = (100,2400,100)

stoppingApprox = ionStopping.betheApprox(stoppingModel, eD_stoppingApprox_binning, x_binCenters)

dataHist = np.zeros((x_bins, eD_bins))
# introduce a ~10% attentuation of beam intensity across cell
cellAttenuationWeights = initialize_oneBD.getCellAttenuationCoeffs(x_binCenters)

class ppcTools_oneBD:
    """Assist in the sampling of posterior distributions from MCMC TOF fits
    """
    def __init__(self, chainFilename, nSamplesFromTOF=nSamples):
        """Create a PPC tools object - reads in chain from file"""
        self.chain, self.probs, self.nParams, self.nWalkers, self.nSteps = readChainFromFile(chainFilename)
        
        
        self.eD_binMax = eD_bins - 1
        
        
        
        self.data_x = np.repeat(x_binCenters, nEvPerLoop)
        
        
        
        
        
        
            
        self.eN_binCenters = getDDneutronEnergy(eD_binCenters )
        
        
        
        self.tofData = None
        self.neutronSpectra = None
        
        # flag to use "old" energy parameterization
        # old parameterization uses both a beam energy and an energy loss parameter
        # these are totally degenerate, so it was abandoned in favor of a single (loss) parameter w.r.t. a reference energy
        self.oldEnergyParams = False

        

        if self.nParams == 10:
            if self.oldEnergyParams == True:
                self.paramNames = [r'$E_0$', r'$f_1$', r'$f_2$', r'$f_3$']
                self.nParamsOfInterest = 4
        if self.nParams == 9:
            self.paramNames = [r'$\Delta E_0$', r'Scale', r's']
            self.nParamsOfInterest = 3

        

        [self.paramNames.append(r'$N_{}$'.format(runId)) for runId in range(nRuns)]
        [self.paramNames.append(r'$BG_{}$'.format(runId)) for runId in range(nRuns)]
        
        
    def generateModelData(self, params, standoffDistance, range_tof, nBins_tof, ddnXSfxn,
                      stoppingApprox, beamTimer, nSamples, getPDF=False, storeDTOF=False):
        """
        Generate model data with cross-section weighting applied
        ddnXSfxn is an instance of the ddnXSinterpolator class -
        dedxfxn is a function used to calculate dEdx -
        probably more efficient to these in rather than reinitializing
        one each time
        This is edited to accommodate multiple standoffs being passed
        
        storeDTOF is an option added to add capabilities in MCNP SDEF generation
        """
        if self.oldEnergyParams == True:
            beamE, eLoss, scale, s, scaleFactor, bgLevel = params
        else:
            eLoss, scale, s, scaleFactor, bgLevel = params
            beamE = experimentConsts.csi_oneBD.beamReferenceEnergy
        e0mean = 1500.0
    

        nLoops = int(np.ceil(nSamples / nEvPerLoop))
        solutions = np.zeros((nEvPerLoop,x_bins))

        for loopNum in range(0, nLoops):
            # maybe pull out definition of eZeros? 
            # can really just be run .. once.. 
            eZeros = np.repeat(beamE, nEvPerLoop)
            eZeros -= lognorm.rvs(s=s, loc=eLoss, scale=scale, size=nEvPerLoop)
            
        
            
            
            eD_atEachX = np.zeros((x_bins,eD_bins))
            for idx,eZero in enumerate(eZeros):
                sol = stoppingApprox.evalStopped(eZero, x_binCenters)
                solutions[idx,:] = sol
            for idx,(xStepSol, attenuationFactor) in enumerate(zip(solutions.T, cellAttenuationWeights)):
                # weight is based on DDN cross section and any attenuation due to location in cell
                data_weights = ddnXSfxn.evaluate(xStepSol) * attenuationFactor
                hist, edEdges = np.histogram( xStepSol, bins=eD_bins, range=(eD_minRange, eD_maxRange), weights=data_weights)
                dataHist[idx,:] = hist
                noWeight_hist, edEdges = np.histogram( xStepSol, bins=eD_bins, range=(eD_minRange, eD_maxRange))
                eD_atEachX[idx,:] = noWeight_hist
            

        #dataHist /= np.sum(dataHist*self.eD_binSize*self.x_binSize)
        e0mean = np.mean(eZeros)
        drawHist2d = (np.rint(dataHist * nSamples)).astype(int)
        tofs = []
        tofWeights = []
        eN_list = []
        eN_atEachX = np.zeros(eD_bins)
#        dtofs = []
        for index, weight in np.ndenumerate( drawHist2d ):
            cellLocation = x_binCenters[index[0]]
            effectiveDenergy = (e0mean + eD_binCenters[index[1]])/2
            tof_d = getTOF( masses.deuteron, effectiveDenergy, cellLocation )
            neutronDistance = (distances.tunlSSA_CsI.cellLength - cellLocation +
                               standoffDistance )
            tof_n = getTOF(masses.neutron, self.eN_binCenters[index[1]],
                           neutronDistance)
            zeroD_times, zeroD_weights = zeroDegTimeSpreader.getTimesAndWeights( self.eN_binCenters[index[1]] )
            tofs.append( tof_d + tof_n + zeroD_times )
            tofWeights.append(weight * zeroD_weights)
            eN_list.append(weight)
#            if storeDTOF:
#                dtofs.append(tof_d)
            if index[1] == self.eD_binMax:
                eN_arr = np.array(eN_list)
                eN_atEachX = np.vstack((eN_atEachX, eN_arr))
                eN_list = []
                

        tofData, tofBinEdges = np.histogram( tofs, bins=nBins_tof, range=range_tof,
                                        weights=tofWeights, density=getPDF)
        # spread with expo modeling transit time across 0 degree
        tofData = np.convolve(tofData, zeroDegSpread_vals, 'full')[:-len(zeroDegSpread_binCenters)+1]
        return (scaleFactor * beamTimer.applySpreading(tofData) + np.random.poisson(bgLevel, nBins_tof), 
                eN_atEachX,
                eD_atEachX)
        
        
        

        
    def generatePPC(self, nSamples = 50, nChainEntries=50, lnprobcut=0.):
        """Sample from the posterior and produce data in the observable space
        
        nChainEntries specifies the number of steps in the chain(s) to consider when drawing samples
        nSamples is the number of parameter samples to include in the PPC
        """
        generatedData = []
        generatedNeutronSpectra=[]
        generatedDeuteronSpectra=[]
        totalChainSamples = len(self.chain[-nChainEntries:,:,0].flatten())
        if lnprobcut != 0.:
            flatProbCutIndices = np.where(self.probs[-nChainEntries:,:].flatten() > lnprobcut)
            totalChainSamples = len(flatProbCutIndices)

        
        # TODO: this next line could mean we repeat the same sample, i think
        samplesToGet = np.random.randint(0, totalChainSamples, size=nSamples)
        if lnprobcut != 0.:
            # if we're getting samples subject to lnprob cut, get those indices
            samplesToGet = flatProbCutIndices[samplesToGet]
        for sampleToGet in samplesToGet:
            modelParams = []
            for nParam in range(self.nParams):
                modelParams.append(self.chain[-nChainEntries:,:,nParam].flatten()[sampleToGet])

            scaleFactorEntries = modelParams[self.nParamsOfInterest:self.nParamsOfInterest+nRuns]
            bgEntries = modelParams[-nRuns:]

            if self.oldEnergyParams == True and self.nParams == 10:    
                e0, loc, scale, s = modelParams[:4]
            
                returnedData = [self.generateModelData([e0, loc, scale, s, scaleFactor, bgScale],
                                        standoff, tofrange, tofbins,
                                        ddnXSinstance, stoppingApprox,
                                        beamTiming, nSamples, True) for
                                        scaleFactor, bgScale, standoff, tofrange, tofbins
                                        in zip(scaleFactorEntries, bgEntries, 
                                                standoffs[:nRuns],
                                                tof_range[:nRuns],
                                                tofRunBins[:nRuns])]
            if self.nParams == 9:    
                deltaE, scale, s = modelParams[:3]
            
                returnedData = [self.generateModelData([deltaE, scale, s, scaleFactor, bgScale],
                                        standoff, tofrange, tofbins,
                                        ddnXSinstance, stoppingApprox,
                                        beamTiming, nSamples, True) for
                                        scaleFactor, bgScale, standoff, tofrange, tofbins
                                        in zip(scaleFactorEntries, bgEntries, 
                                                standoffs[:nRuns],
                                                tof_range[:nRuns],
                                                tofRunBins[:nRuns])]
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
        dZeroSamples = np.zeros(eD_bins)
        totalChainSamples = len(self.chain[-50:,:,0].flatten())
        samplesToGet = np.random.randint(0, totalChainSamples, size=nSamples)
        for sampleToGet in samplesToGet:
            # TODO: if number of parameters associated with energy distribution change, this will need to be updated as the  num is presently hardcoded
            modelParams = []
            for nParam in range(4):
                modelParams.append(self.chain[-50:,:,nParam].flatten()[sampleToGet])
        
            e0, loc, scale, s = modelParams[:4]
            edistrib = e0 - lognorm.rvs(s=s, loc=loc, scale=scale, size=nEvPerLoop)
            eHist, bins = np.histogram(edistrib, bins=eD_bins,
                                       range=(eD_minRange, 
                                              eD_maxRange),
                                       density=returnNormed)
            dZeroSamples = np.vstack((dZeroSamples, eHist))
        if returnNormed:
            return dZeroSamples[1:]*eD_binSize
        return dZeroSamples[1:]
            
    
    
    def getDTOFdistribution(self):
        '''Produce a distribution of deuteron time-of-flight through 
        gas cell from PPC'''
        totalChainSamples = len(self.chain[-50:,:,0].flatten())
        
        dtofHist = np.zeros((x_bins, 100))
        
        dedxForODE = lambda x, y: self.stoppingModel.dEdx(energy=y,x=x)
        
        samplesToGet = np.random.randint( 0, totalChainSamples, 1 )
        for sampleToGet in samplesToGet:
            modelParams = []
            for nParam in range(self.nParams):
                modelParams.append(self.chain[-50:,:,nParam].flatten()[sampleToGet])
                
            e0, loc, scale, s = modelParams[:4]
            
            eZeros = np.repeat( e0, 1000 )
            eZeros -= lognorm.rvs(s=s, loc=loc, scale=scale, size=1000)
            print(eZeros)
            odesolver = ode( dedxForODE ).set_integrator('dopri5')
            odesolver.set_initial_value(eZeros)
            eD_atEachX = np.zeros(eD_bins)
            res = None
            evPoints = []
            for idx, xEvalPoint in enumerate(x_binCenters):
                sol = odesolver.integrate(xEvalPoint)
                print('x loc: {}\t{}'.format(xEvalPoint,sol[0]))
                if idx == 0:
                    res = sol
                else:
                    res = np.vstack((res, sol))
                evPoints.append(xEvalPoint)
        odesolver.set_initial_value([900.0,850.0])
        testSols = [odesolver.integrate([1.0,1.0]), odesolver.integrate([2.0,2.0])]
        print('test solutions...\ntest 1: {}\ntest 2: {}'.format(testSols[0], testSols[1]))
        return res, evPoints, sol
            
            
    def makeSDEF_sia_cumulative(self, distNumber = 100):
        """Produce an MCNP SDEF card for the neutron distribution, marginalized over the cell length
        
        This SDEF card will adhere to an SI A standard"""
        # if we havent generated data yet, do it
        if self.tofData == None or self.neutronSpectra == None:
            self.generatePPC(self, 500)
            
        # first we collapse the neutron spectrum, collected by default atall X values
        neutronSpectrumCollection = np.zeros(eD_bins)
        for sampledParamSet in self.neutronSpectra:
            samplesAlongLength = sampledParamSet[0]
            summedAlongLength = np.sum(samplesAlongLength, axis=0)
            neutronSpectrumCollection = np.vstack((neutronSpectrumCollection, summedAlongLength))
        self.neutronSpectrum = np.sum(neutronSpectrumCollection, axis=0)
        
        
        siStrings = ['si{} a'.format(distNumber)]
        spStrings = ['sp{}'.format(distNumber)]
        for eN, counts in zip(self.eN_binCenters, self.neutronSpectrum):
            siStrings.append(' {:.3f}'.format(eN/1000))
            spStrings.append(' {:.3e}'.format(counts))
        siString = ''.join(siStrings)
        spString = ''.join(spStrings)
        self.sdef_sia_cumulative = {'si': siString, 'sp': spString}
        return self.sdef_sia_cumulative, self.eN_binCenters, self.neutronSpectrum            
        
        
    def makeCornerPlot(self, paramIndexLow = 0, paramIndexHigh = None, nStepsToInclude = 50, plotFilename = 'ppcCornerOut.png'):
        """Produce a corner plot of the parameters from the chain
        
        paramIndexLow and paramIndexHigh define the upper and lower boundaries of the parameter indices to plot"""
        if paramIndexHigh == None:
            paramIndexHigh = self.nParams
        samples = self.chain[-nStepsToInclude:,:,paramIndexLow:paramIndexHigh].reshape((-1, paramIndexHigh - paramIndexLow))
        import corner as corn
        cornerFig = corn.corner(samples, labels=self.paramNames[paramIndexLow:paramIndexHigh], 
                                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                title_kwargs={'fontsize': 12})
        cornerFig.savefig(plotFilename)