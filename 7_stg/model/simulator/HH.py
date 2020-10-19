#!/usr/bin/env python
import sys
import tqdm
import numpy

DEBUG = True

if 'debug' in sys.argv:
    DEBUG = True

if DEBUG:
    import pdb

# Use when need to suppress windows and fancy progress bars
if 'nodisp' in sys.argv:
    import matplotlib as mpl
    mpl.use('Agg')
    tqdm.tqdm = lambda x: x

import numpy as np

exp = np.exp
log = np.log

CYTHON = True # sets whether cython will be used.

if CYTHON:
    import pyximport

    # setup_args added by Michael Deistler, needed on my OS
    pyximport.install(setup_args={"include_dirs":numpy.get_include()}, reload_support=True)
    from cHH import HH as cHH

def mhtt(V, num, den):
    return 1.0 / (1 + exp((V + num) / den))

def mhtt2(V, num1, den1, num2, den2):
    return 1.0 / (exp((V + num1) / den1) + exp((V + num2) / den2))

def getINatauh(V):
    return (1.34 / (1 + exp((V + 62.9) / -10))) * (1.5 + 1 / (1 + exp((V + 34.9) / 3.6)))

def getIHtaum(V):
    return 2.0 / (exp(-14.59 - 0.086 * V) + exp(-1.87 + 0.0701 * V))

# Numerical data for the channel dynamics, taken from Prinz 2003, p. 3999
# These value are the constants needed to calculate m_inf, h_inf, tau_m, tau_n given the voltage V
# These are NO conductances
INadata     = [3, 25.5 , -5.29, 48.9 , 5.18 , 2.64 , -2.52 , 120  , -25   , None, None , None , None  , None , None  , None , None]
ICaTdata    = [3, 27.1 , -7.2 , 32.1 , 5.5  , 43.4 , -42.6 , 68.1 , -20.5 , None, None , 210  , -179.6, 55   , -16.9 , None , None]
ICaSdata    = [3, 33   , -8.1 , 60   , 6.2  , 2.8  , 14    , 27   , 10    , 70  , -13  , 120  , 300   , 55   , 9     , 65   , -16 ]
IAdata      = [3, 27.2 , -8.7 , 56.9 , 4.9  , 23.2 , -20.8 , 32.9 , -15.2 , None, None , 77.2 , -58.4 , 38.9 , -26.5 , None , None]
IKCadata    = [4, 28.3 , -12.6, None , None , 180.6, -150.2, 46   , -22.7 , None, None , None , None  , None , None  , None , None]
IKddata     = [4, 12.3 , -11.8, None , None , 14.4 , -12.8 , 28.3 , -19.2 , None, None , None , None  , None , None  , None , None]
IHdata      = [1, 75   , 5.5  , None , None , None , None  , None , None  , None, None , None , None  , None , None  , None ]
IProcdata   = [1, 55   , 5.0  , None , None , 6    , 0     , 0    , 1     , None, None , None , None  , None , None  , None ]
IProcdata   = [1, 12   ,-3.05 , None , None , 0.5  , None  , None , None  , None, None , None , None  , None , None  , None]

# Reversal voltages and dissipation time constants for the synapses, taken from Prinz 2004, p. 1351
Esglut = -70            # mV    
kminusglut = 40         # ms

Eschol = -80            # mV
kminuschol = 100        # ms

# Conductances of the different neuron types from Prinz' 2004 paper
# The format of each entry is:
# [ gNa     gCaT    gCaS    gA      gKCa    gKd     gH      gleak ]
# The conductances are given in mS
abmodels = [ [ 400, 2.5, 6, 50, 10, 100, 0.01, 0.00 ], \
             [ 100, 2.5, 6, 50, 5 , 100, 0.01, 0.00 ], \
             [ 200, 2.5, 4, 50, 5 , 50 , 0.01, 0.00 ], \
             [ 200, 5.0, 4, 40, 5 , 125, 0.01, 0.00 ], \
             [ 300, 2.5, 2, 10, 5 , 125, 0.01, 0.00 ] ]

lpmodels = [ [ 100, 0.0, 8 , 40, 5, 75 , 0.05, 0.02 ], \
             [ 100, 0.0, 6 , 30, 5, 50 , 0.05, 0.02 ], \
             [ 100, 0.0, 10, 50, 5, 100, 0.00, 0.03 ], \
             [ 100, 0.0, 4 , 20, 0, 25 , 0.05, 0.03 ], \
             [ 100, 0.0, 6 , 30, 0, 50 , 0.03, 0.02 ] ]

pymodels = [ [ 100, 2.5 , 2, 50, 0, 125, 0.05, 0.01 ], \
             [ 200, 7.5 , 0, 50, 0, 75 , 0.05, 0.00 ], \
             [ 200, 10.0, 0, 50, 0, 100, 0.03, 0.00 ], \
             [ 400, 2.5 , 2, 50, 0, 75 , 0.05, 0.00 ], \
             [ 500, 2.5 , 2, 40, 0, 125, 0.01, 0.03 ], \
             [ 500, 2.5 , 2, 40, 0, 125, 0.00, 0.02 ] ]



#########################################################################################################
# MD: I stopped maintaining this function HH.py. It thus lacks new features such as neuromodulators and #
#     and updated implementation of temperature.                                                        #
#########################################################################################################

class HH:
    def __init__(self, seed=None):
        """ Initializes class 

            seed       : seed to initialise random number generator (can be None)
        """

        # note: make sure to generate all randomness through self.rng (!)
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

    def sim_time(self, 
                 dt,
                 t, 
                 Ix,   
                 modelx,
                 conns,
                 g_q10_conns_gbar,
                 g_q10_conns_tau,
                 temp, 
                 init = None, 
                 verbose=True):
        """Simulates the model for a specified time duration.

           dt       : timestep (mS)
           t        : array of time values - should be np.arange(0,tmax,dt)
           Ix       : external input currents for each neuron --> often noise
           modelx   : model to use for each neuron
           conns    : list of connections in the form [ #out, #in, strength, Es, kminus ]
                      Units:
                          strength:     mS  <--- g_s
                          Es:           mV  <--- reversal potential of the synapse
                          kminus:       ms  <--- rate constant for transmitter-receptor dissociation rate

           init     : initial values for voltage, Ca concentration and state variables
        """

        modelx = np.asarray(modelx)
        conns = np.asarray(conns)

        n = len(modelx)
        m = len(conns)

        g_q10 = 1.3
        g_temp_factor = g_q10 ** ((temp - 283) / 10)

        # Parameters
        gNax    = modelx.T[0] * g_temp_factor # mS
        gCaTx   = modelx.T[1] * g_temp_factor # mS
        gCaSx   = modelx.T[2] * g_temp_factor # mS
        gAx     = modelx.T[3] * g_temp_factor # mS
        gKCax   = modelx.T[4] * g_temp_factor # mS
        gKdx    = modelx.T[5] * g_temp_factor # mS
        gHx     = modelx.T[6] * g_temp_factor # mS
        gleakx  = modelx.T[7] * g_temp_factor # mS
        gProcx  = modelx.T[8] * 1.0           # mS, proctolin is modelled to not underly temperature
 
        # Constants
        C = 0.6283e-3       # muF

        ENa = 50            # mV
        EK = -80            # mV
        EH = -20            # mV
        Eleak = -50         # mV
        Eproc = 0.0         # mV

        Catau = 200         # ms
        f = 14961           # muM/muA
        CaExt = 3000        # muM
        Ca0 = 0.05          # muM

        R = 8.31451e3                   # mJ / (mol * K)
        F = 96485.3415;                 # C / mol
        z = 2                           # Ca is divalent
        RToverzF = R * temp / (z * F)      # mJ / (mol * K) * K / (C / mol) = mV

        # I think that Vth is the max potential for the synapse
        Vth = -35           # mV
        Delta = 5           # mV

        nsteps = len(t)

        tau_q10 = 1.6
        tau_temp_factor = tau_q10 ** ((temp - 283) / 10)

        ####################################

        # Neuron state variables
        Vx   = np.empty_like(Ix)
        Cax  = np.empty_like(Ix)

        mNax    = np.empty_like(Ix)
        mCaTx   = np.empty_like(Ix)
        mCaSx   = np.empty_like(Ix)
        mAx     = np.empty_like(Ix)
        mKCax   = np.empty_like(Ix)
        mKdx    = np.empty_like(Ix)
        mHx     = np.empty_like(Ix)
        mProcx  = np.empty_like(Ix)
        
        hNax    = np.empty_like(Ix)
        hCaTx   = np.empty_like(Ix)
        hCaSx   = np.empty_like(Ix)
        hAx     = np.empty_like(Ix)

        # Synapse state variables
        sx      = np.zeros((m, nsteps)) # sx is s, the conductance of the synapse --> overall strength is g_s*s

        ICax      = np.zeros((m, nsteps))

        ### Always the case if SNPE calls this, init data is passed via constructor (not supported yet)
        if init == None:        # default: simulation from initial point
            for j in range(n):
                Vx[j, 0] = Eleak
                Cax[j, 0] = Ca0

                hNax[j, 0] = 1
                hCaTx[j, 0] = 1
                hCaSx[j, 0] = 1
                hAx[j, 0] = 1

                mNax[j, 0] = 0
                mCaTx[j, 0] = 0
                mCaSx[j, 0] = 0
                mAx[j, 0] = 0
                mKCax[j, 0] = 0
                mKdx[j, 0] = 0
                mHx[j, 0] = 0
                mProcx[j, 0] = 0
                hNax[j, 0] = 0
                hCaTx[j, 0] = 0
                hCaSx[j, 0] = 0
                hAx[j, 0] = 0

                # ICa is calculated later

        else:                  # simulation from given points
            for i in range(n):
                data = init[i]
                Vx[i, 0] = data[0]
                Cax[i, 0] = data[1]

                mNax[i, 0] = data[2]
                mCaTx[i, 0] = data[3]
                mCaSx[i, 0] = data[4]
                mAx[i, 0] = data[5]
                mKCax[i, 0] = data[6]
                mKdx[i, 0] = data[7]
                mHx[i, 0] = data[8]
                mProcx[i, 0] = data[9]

                hNax[i, 0] = data[9]
                hCaTx[i, 0] = data[10]
                hCaSx[i, 0] = data[11]
                hAx[i, 0] = data[12]

            for i in range(m):
                sx[i, 0] = init[n][i]

        # Currents (only for current timestep)
        cNax = np.zeros(n)           # mS
        cCaTx = np.zeros(n)          # mS
        cCaSx = np.zeros(n)          # mS
        cAx = np.zeros(n)            # mS
        cKCax = np.zeros(n)          # mS
        cKdx = np.zeros(n)           # mS
        cHx = np.zeros(n)            # mS
        cleakx = np.zeros(n)         # mS
        cProcx = np.zeros(n)         # mS

        csx = np.empty(n)                   # mS
        Icsx = np.empty(n)                  # muA
        ECax = np.empty(n)                  # mV

        #################################################################################
        # The rest of the function body should be identical in HH.py and cHH.pyx
        # Debug variables
        logs = { 'I' : Ix, 'Is' : np.empty_like(mNax), 'INa' : np.empty_like(mNax), 'ICaT' : np.empty_like(mCaTx), \
                 'ICaS' : np.empty_like(mCaSx), 'IA' : np.empty_like(mAx), \
                 'IKCa' : np.empty_like(mKCax), 'IKd' : np.empty_like(mKdx), 'IH' : np.empty_like(mHx), 'Ileak' : np.empty_like(mHx),
                  'mNa' : mNax, 'mCaT' : mCaTx, 'mCaS' : mCaSx, 'mA' : mAx, 'mKCa' : mKCax, 'mKd' : mKdx, 'mH' : mHx,
                  'hNa' : hNax, 'hCaT' : hCaTx, 'hCaS' : hCaSx, 'hA' : hAx, 's' : sx }

        # Calculate Ca current at time 0
        for j in range(n):
            # equilibrium potential for calcium
            ECax[j] = -RToverzF * log(Cax[j, 0]/CaExt)             # mV * 1 = mV
            # calcium current based on the CaT (fast) and CaS (slow)
            ICax[j, 0] = (gCaTx[j] * (mCaTx[j, 0] ** ICaTdata[0]) * hCaTx[j, 0] + \
                          gCaSx[j] * (mCaSx[j, 0] ** ICaSdata[0]) * hCaSx[j, 0]) * (Vx[j, 0] - ECax[j])     # mS??? * mV = muA

        # just for print statement
        if verbose:
            iterlist = tqdm.tqdm(range(1, nsteps))
        else:
            iterlist = range(1,nsteps)

        # looping over timesteps
        for i in iterlist:

            # Calculate synaptic currents
            for k in range(n):
                csx[k] = Icsx[k] = 0

            # updating synpatic states. Synapse model is described in Prinz et al. 2004
            for k in range(m): # m are the connections
                npost = int(conns[k,0]) # 0 are postsynaptic neurons
                # sx is the synaptic state
                csx[npost] += -conns[k,2] * sx[k, i-1] # 2 are strenghts of the synapse  # positive currents inhibit spiking in our model
                Icsx[npost] += -conns[k,2] * sx[k, i-1] * conns[k,3] # 3 is the reversal potential # mS * 1 * mV = muA

            # Update V and [Ca] for all neurons
            for j in range(n):
                # Exponential Euler
                # these values are the 'effective conductances', i.e., they are g*m^3*h
                cNax[j] = gNax[j] * (mNax[j, i - 1] ** INadata[0]) * hNax[j, i - 1]         # mS
                cCaTx[j] = gCaTx[j] * (mCaTx[j, i - 1] ** ICaTdata[0]) * hCaTx[j, i - 1]    # mS
                cCaSx[j] = gCaSx[j] * (mCaSx[j, i - 1] ** ICaSdata[0]) * hCaSx[j, i - 1]    # mS
                cAx[j] = gAx[j] * (mAx[j, i - 1] ** IAdata[0]) * hAx[j, i - 1]              # mS
                cKCax[j] = gKCax[j] * (mKCax[j, i - 1] ** IKCadata[0])                      # mS
                cKdx[j] = gKdx[j] * (mKdx[j, i - 1] ** IKddata[0])                          # mS
                cHx[j] = gHx[j] * (mHx[j, i - 1] ** IHdata[0])                              # mS
                cleakx[j] = gleakx[j]                                                       # mS

                # Calculate Ca reversal potential using Nernst equation
                ECax[j] = RToverzF * log(CaExt / Cax[j, i-1])                            # mV * 1 = mV

                # Calcium current is the sum of the fast and the slow Ca-current
                ICax[j, i] = (cCaTx[j] + cCaSx[j]) * (Vx[j, i-1] - ECax[j])              # mS??? * mV = muA

                # t_Ca d[Ca]/dt = -f * (I_CaT + I_CaS) - [Ca] + [Ca]_0
                # Catau is a constant defined above
                Cainf = Ca0 - f * ICax[j, i]                                                       # (muM / muA) * muA = muM
                Cax[j, i] = Cainf + (Cax[j, i-1] - Cainf) * exp(-dt * tau_temp_factor / Catau)     # muM; Exponent: ms/ms = 1

                Vcoeff = csx[j] + cNax[j] + cCaTx[j] + cCaSx[j] + cAx[j] + cKCax[j] + cKdx[j] + cHx[j] + cleakx[j] # mS
                # synapcit current Iscx plus all the other currents
                Vinf_ = Icsx[j] + cNax[j] * ENa + cCaTx[j] * ECax[j] + cCaSx[j] * ECax[j] + cAx[j] * EK + cKCax[j] * EK + \
                        cKdx[j] * EK + cHx[j] * EH + cleakx[j] * Eleak  + Ix[j, i]
                if Vcoeff == 0:
                    Vx[j, i] = Vx[j, i-1] + dt * Vinf_ / C
                else:
                    Vinf = Vinf_ / Vcoeff                       # muA / mS = mV
                    # this is the exponential euler
                    Vx[j, i] = Vinf + (Vx[j, i-1] - Vinf) * exp(-dt * Vcoeff / C)        # ms * mS / muF = 1

            # Update gating variables
            for j in range(n):
                # t_m * dm/dt = m_inf - m
                # Prinz used a truncating Forward Euler scheme for the gating variables
                # Use old values for V instead of new ones? i -> i-1
                mNainf = mhtt(Vx[j, i-1], INadata[1], INadata[2]) 
                mNatau = INadata[5] + INadata[6] * mhtt(Vx[j, i-1], INadata[7], INadata[8])       # ms
                mNax[j, i] = mNainf + (mNax[j, i-1] - mNainf) * exp(-dt * tau_temp_factor / mNatau)

                mCaTinf = mhtt(Vx[j, i-1], ICaTdata[1], ICaTdata[2]) 
                mCaTtau = ICaTdata[5] + ICaTdata[6] * mhtt(Vx[j, i-1], ICaTdata[7], ICaTdata[8])  # ms
                mCaTx[j, i] = mCaTinf + (mCaTx[j, i-1] - mCaTinf) * exp(-dt * tau_temp_factor / mCaTtau)

                mCaSinf = mhtt(Vx[j, i-1], ICaSdata[1], ICaSdata[2]) 
                mCaStau = ICaSdata[5] + ICaSdata[6] * mhtt2(Vx[j, i-1], ICaSdata[7], ICaSdata[8], ICaSdata[9], ICaSdata[10])   # ms
                mCaSx[j, i] = mCaSinf + (mCaSx[j, i-1] - mCaSinf) * exp(-dt * tau_temp_factor / mCaStau)

                mAinf = mhtt(Vx[j, i-1], IAdata[1], IAdata[2]) 
                mAtau = IAdata[5] + IAdata[6] * mhtt(Vx[j, i-1], IAdata[7], IAdata[8])            # ms
                mAx[j, i] = mAinf + (mAx[j, i-1] - mAinf) * exp(-dt * tau_temp_factor / mAtau)

                mKCainf = (Cax[j, i-1] / (Cax[j, i-1] + 3)) * mhtt(Vx[j, i-1], IKCadata[1], IKCadata[2]) 
                mKCatau = IKCadata[5] + IKCadata[6] * mhtt(Vx[j, i-1], IKCadata[7], IKCadata[8])  # ms
                mKCax[j, i] = mKCainf + (mKCax[j, i-1] - mKCainf) * exp(-dt * tau_temp_factor / mKCatau)

                mKdinf = mhtt(Vx[j, i-1], IKddata[1], IKddata[2]) 
                mKdtau = IKddata[5] + IKddata[6] * mhtt(Vx[j, i-1], IKddata[7], IKddata[8])       # ms
                mKdx[j, i] = mKdinf + (mKdx[j, i-1] - mKdinf) * exp(-dt * tau_temp_factor / mKdtau)

                mHinf = mhtt(Vx[j, i-1], IHdata[1], IHdata[2]) 
                mHtau = getIHtaum(Vx[j, i-1])
                mHx[j, i] = mHinf + (mHx[j, i-1] - mHinf) * exp(-dt * tau_temp_factor / mHtau)

                hNainf = mhtt(Vx[j, i-1], INadata[3], INadata[4]) 
                hNatau = getINatauh(Vx[j, i-1])                   # ms
                hNax[j, i] = hNainf + (hNax[j, i-1] - hNainf) * exp(-dt * tau_temp_factor / hNatau)

                hCaTinf = mhtt(Vx[j, i-1], ICaTdata[3], ICaTdata[4]) 
                hCaTtau = ICaTdata[11] + ICaTdata[12] * mhtt(Vx[j, i-1], ICaTdata[13], ICaTdata[14])      # ms
                hCaTx[j, i] = hCaTinf + (hCaTx[j, i-1] - hCaTinf) * exp(-dt * tau_temp_factor / hCaTtau)

                hCaSinf = mhtt(Vx[j, i-1], ICaSdata[3], ICaSdata[4]) 
                hCaStau = ICaSdata[11] + ICaSdata[12] * mhtt2(Vx[j, i-1], ICaSdata[13], ICaSdata[14], ICaSdata[15], ICaSdata[16])# ms
                hCaSx[j, i] = hCaSinf + (hCaSx[j, i-1] - hCaSinf) * exp(-dt * tau_temp_factor / hCaStau)

                hAinf = mhtt(Vx[j, i-1], IAdata[3], IAdata[4]) 
                hAtau = IAdata[11] + IAdata[12] * mhtt(Vx[j, i-1], IAdata[13], IAdata[14])                # ms
                hAx[j, i] = hAinf + (hAx[j, i-1] - hAinf) * exp(-dt * tau_temp_factor / hAtau)

            # updating synapses. Synapse model is described in Prinz et al. 2004. Based on Abbott et al. 1998
            for k in range(m):
                # Rewritten to avoid overflow under standard conditions
                npre = int(conns[k,1]) # 1 is the presynaptic neuron identifier
                e = exp((Vth - Vx[npre, i-1]) / Delta)
                sinf = 1 / (1 + e)
                stau =  conns[k,4] * (1 - sinf)  # 4 is kminus   # 1 / ms^-1 = ms

                #sx[k, i] = sinf + (sx[k, i-1] - sinf) * exp(-dt / stau)  # ms / ms = 1
                sx[k, i] = sx[k, i-1] + (sinf - sx[k, i-1]) * dt / stau  # ms / ms = 1
                if dt > stau: # for numerical stability. If steepness exceeds delta_t, just set it to max
                    sx[k, i] = sinf

        ret = { 'Vs' : Vx, 'Cas': Cax, 'ICas' : ICax, 'logs' : logs }
        return ret
