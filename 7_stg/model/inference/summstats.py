import numpy as np
import pdb
import prinzdb
import scipy
import scipy.signal

from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
from copy import deepcopy

neutypes = prinzdb.neutypes
ref_neuron = neutypes[0]

silent_thresh = 1           # minimum number of spikes per second for a neuron not to be considered silent
tonic_thresh = 30           # minimum number of spikes per second for a neuron to be considered tonically firing
burst_thresh = 0.5          # maximum percentage of single-spike bursts for a neuron to be considered
                            # bursting (and not simply spiking)
bl_thresh = 50              # minimum number of milliseconds for bursts not to be discounted as single spikes
spike_thresh = -10          # minimum voltage above which spikes are considered
ibi_threshold = 150         # maximum time between spikes in a given burst (ibi = inter-burst interval)
plateau_threshold = -30     # minimum voltage to be considered for plateaus

triphasic_thresh = 0.9      # percentage of triphasic periods for system to be considered triphasic
pyloric_thresh = 0.7        # percentage of pyloric periods for triphasic system to be pyloric_like


NaN = float("nan")
inf = float("inf")

# Use this instead of import from delfi if delfi is not installed
# class BaseSummaryStats:
#     def __init__(self, *args, **kwargs):
#         pass


########################################################
# This is an overview of Prinz' summary statistics detection routine
# to serve as a reference
#
# Saved Data in Prinz:
# cellmaxnum:   number of voltage maxima
# firstmaxtime
# cellimiends: ends of inter-maximum intervals
# cellimi: durations of IMIs
# meancellimi, mincellimi, maxcellimi

# spikenum: spikes are maxima that overshoot 0mV
# firstspiketime
# cellisiends
# cellisis
# mean, min, maxcellisi

# maxcellisi, mincellisi
# is_modulated_tonic: if largest IMI is less than 5 times the smalles ISI
# isithreshold: 99% of the largest ISI for modulated tonics, ie. only the largest ISI becomes an IBI
#               50% of the largest ISI if it is more than 2 times the mean ISI
#               otherwise, the mean ISI
#               but the maximum is always 0.5s

# winspikes: number of spikes in detection window
# winmin, winmax: smallest and largest ISI in the detection window

# for each ISI, if above the ISI threshold, it is classified as an IBI
# cellibiend
# cellibi
# ibinum

# if more than 10 spikes in detection window (5s), cell is tonic if diff between max and min isi as a fraction of their average is less than 5%
# a tonic cell is not a modulated tonic

# For non-tonic cells:
# spike_periodic: false if not tonic and no IBIs are saved
# cellperiod: 0 if no IBIs
# cellibispercycle: 0 if no IBIs

# to find out if a cell with IBIs is periodic, go backwards through the IBIs
# if difference between any two consecutive IBIs is less than 5% as a fraction of their average, this is consistend with periodic bursting
# so you can read off the burst period 1
# then compare the current ibi to second to last detected, then third, for burst period 2 and 3 resp.
# then take the smallest of the three periods, and corresponding ibispercycle

# now to find the number of IBIs per period
# temporary time window : end is middle of last IBI, start is one period before the end time (cellperiod * cyclespp)
# find longest IBI that ends within the temp time window. set window end to middle of this IBI
# window start is again one period before window end
# count IBIs within this window
# also count IBIs in window one period before this

# Do the same for ICIs, using IMIs instead of ISIs

# Detect for periodicity
# cell-wise:
# for periodicity analysis want more than 5 spikes since simulation start
# if cell tonic, it is periodic, period is mean cell ISI, IBIspercycle=1
# else use above procedure

# whole system
# if all cells are periodic or max periodic:
# find max period
# for each cell, cyclesppis maxperiod / cellperiod, rounded to integer
# count Ibispp or ICIspp

# if cyclesperpp equal values for last cycle, increase concistentcyclesperpp
# same for ibiicipp
# if one of these s true, cell is called consistent
# if consistent:
# averageperiod: average of cyclespp * cellperiod
# maxperioddeviation
# if maxperioddeviation < 0.5% avgperiod:
# period = avgperiod

# Now go through all cells. Tonic cells are to be checked. 
# Cells that spike periodically are to be checked if cyclespp * cellibispercycle == ibiicipp and the latter equal ibiicippbefore
# Analogous for non-spiking periodical cells

# If all cells are checked and we have enough time (at least 2 * period), periodic = true
# else periodic is false and no consistent cycles per period or IBIs or ICIs are found
# If no periodic spikes, do the same for subthreshold maxima


# for periodic cells, if cyclespp are all 1 andn ibispercycle are all 1, triphasic is true
# rhythm is pyloric if bursts are in correct order

# Maxima detected by slope switching from pos to neg
##############################################################

class PrinzStats(BaseSummaryStats):
    def __init__(self, t_on, t_off, include_pyloric_ness=True, include_plateaus=False, seed=None):
        """See SummaryStats.py for docstring"""
        super(PrinzStats, self).__init__(seed=seed)
        self.t_on = t_on
        self.t_off = t_off
        self.include_pyloric_ness = include_pyloric_ness
        self.include_plateaus = include_plateaus

        self.labels = [ 'cycle_period' ] + [ 'burst_length_{}'.format(nt) for nt in neutypes ] \
                                         + [ 'end_to_start_{}_{}'.format(neutypes[i], neutypes[j])
                                             for i, j in ((0,1), (1,2)) ] \
                                         + [ 'start_to_start_{}_{}'.format(neutypes[i], neutypes[j])
                                             for i, j in ((0,1), (0,2)) ] \
                                         + [ 'duty_cycle_{}'.format(nt) for nt in neutypes ] \
                                         + [ 'phase_gap_{}_{}'.format(neutypes[i], neutypes[j])
                                             for i, j in ((0,1), (1,2)) ] \
                                         + [ 'phase_{}'.format(neutypes[i]) for i in (1,2) ]

        if self.include_plateaus:
            self.labels += [ 'plateau_length_{}'.format(nt) for nt in neutypes ]
        if self.include_pyloric_ness:
            self.labels += ['pyloric_like']

        self.n_summary = len(self.labels)

    def calc(self, repetition_list):
        # does not make use of last element summ['pyloric_like']
        ret = np.empty((len(repetition_list), self.n_summary))

        for r in range(len(repetition_list)):
            x = repetition_list[r]

            tmax = x['tmax']
            dt = x['dt']
            t_on = min(self.t_on, tmax)
            t_off = min(self.t_off, tmax)

            Vx = x['data']
            Vx = Vx[:,int(t_on/dt):int(t_off/dt)]

            # retreive summary statistics stored in a dictionary
            summ = calc_summ_stats(Vx, tmax, dt, self.include_pyloric_ness)

            # create array of summary statistics using the data from the dictionary
            duty_cycles = [ summ['duty_cycles'][nt] for nt in neutypes ]
            burst_lengths = [ summ[nt]['avg_burst_length'] for nt in neutypes ]
            
            gaps = [ summ['ends_to_starts'][neutypes[i], neutypes[j]] for i, j in ((0,1), (1,2)) ]
            delays = [ summ['starts_to_starts'][neutypes[i], neutypes[j]] for i, j in ((0,1), (0,2)) ]

            phase_gaps = [ summ['phase_gaps'][neutypes[i], neutypes[j]] for i, j in ((0,1), (1,2)) ]
            phases = [ summ['start_phases'][neutypes[i]] for i in (1,2) ]

            plateau_lengths = [ summ['plateau_lengths'][nt] for nt in neutypes ]

            if self.include_pyloric_ness and self.include_plateaus:
                ret[r] =  np.asarray([ summ['cycle_period'], *burst_lengths, *gaps, *delays, *duty_cycles,
                                       *phase_gaps, *phases, *plateau_lengths, summ['pyloric_like'] ])
            if self.include_pyloric_ness and not self.include_plateaus:
                ret[r] =  np.asarray([ summ['cycle_period'], *burst_lengths, *gaps, *delays, *duty_cycles,
                                       *phase_gaps, *phases, summ['pyloric_like'] ])
            if not self.include_pyloric_ness and self.include_plateaus:
                ret[r] =  np.asarray([ summ['cycle_period'], *burst_lengths, *gaps, *delays, *duty_cycles,
                                       *phase_gaps, *phases, *plateau_lengths])
            if not self.include_pyloric_ness and not self.include_plateaus:
                ret[r] =  np.asarray([ summ['cycle_period'], *burst_lengths, *gaps, *delays, *duty_cycles,
                                       *phase_gaps, *phases])
                
            # impute unreasonably large values for NaNs (comment out if you want SNPE to impute values)
            # summ_stats[np.isnan(summ_stats)] = 1e6

        return ret


    @staticmethod
    def print_PrinzStats(summ_stats, percentage=False):
        if percentage:
            p=100
            p_str='%'
        else:
            p=1
            p_str=''
        print("----- Summary Statistics -----")
        print("Cycle_period:          ", summ_stats[0]*p,p_str, sep='')
        print("Burst_length AB:       ", summ_stats[1]*p,p_str, sep='')
        print("Burst_length LP:       ", summ_stats[2]*p,p_str, sep='')
        print("Burst_length PY:       ", summ_stats[3]*p,p_str, sep='')
        print("End_to_start AB-LP:    ", summ_stats[4]*p,p_str, sep='')
        print("End_to_start LP-PY:    ", summ_stats[5]*p,p_str, sep='')
        print("Start_to_start AB-LP:  ", summ_stats[6]*p,p_str, sep='')
        print("Start_to_start LP-PY:  ", summ_stats[7]*p,p_str, sep='')
        print("Duty cycle AB:         ", summ_stats[8]*p,p_str, sep='')
        print("Duty cycle LP:         ", summ_stats[9]*p,p_str, sep='')
        print("Duty cycle PY:         ", summ_stats[10]*p,p_str, sep='')
        print("Phase gap AB-LP:       ", summ_stats[11]*p,p_str, sep='')
        print("Phase gap LP-PY:       ", summ_stats[12]*p,p_str, sep='')
        print("Phase LP:              ", summ_stats[13]*p,p_str, sep='')
        print("Phase PY:              ", summ_stats[14]*p,p_str, sep='')


class SpikeStats(BaseSummaryStats):
    def __init__(self, t_on, t_off, seed=None):
        super(SpikeStats, self).__init__(seed=seed)
        self.t_on = t_on
        self.t_off = t_off

        self.labels = [ 'cycle_period' ] + [ 'burst_length_{}'.format(nt) for nt in neutypes ] \
                                         + [ 'ibi_length_{}'.format(nt) for nt in neutypes ] \
                                         + [ 'spike_count_{}'.format(nt) for nt in neutypes ] 

        self.n_summary = len(self.labels)

    def calc(self, repetition_list):
        ret = np.empty((len(repetition_list), self.n_summary))

        for r in range(len(repetition_list)):
            x = repetition_list[r]

            tmax = x['tmax']
            dt = x['dt']
            t_on = min(self.t_on, tmax)
            t_off = min(self.t_off, tmax)

            Vx = x['data']
            Vx = Vx[:,int(t_on/dt):int(t_off/dt)]

            # retreive summary statistics stored in a dictionary
            summ = calc_summ_stats(Vx, tmax, dt)

            # create array of summary statistics using the data from the dictionary
            burst_lengths = [ summ[nt]['avg_burst_length'] for nt in neutypes ]
            ibi_lengths = [ summ[nt]['avg_ibi_length'] for nt in neutypes ]
            spike_counts = [ summ[nt]['num_spikes'] for nt in neutypes ]

            ret[r] = np.asarray([ summ['cycle_period'], *burst_lengths, *ibi_lengths, *spike_counts ])

        return ret

class CPGStats(BaseSummaryStats):
    def __init__(self, t_on, t_off, with_cycle_length = False, seed=None):
        """See SummaryStats.py for docstring"""
        super(CPGStats, self).__init__(seed=seed)
        self.t_on = t_on
        self.t_off = t_off

        self.labels = [ "avg_burst_length_{}".format(nt) for nt in neutypes ] + [ "n_triphasic" ]
        if with_cycle_length:
            self.labels = [ "cycle_length" ] + self.labels

        self.n_summary = len(self.labels)

    def calc(self, repetition_list):
        ret = np.empty((len(repetition_list), self.n_summary))

        for r in range(len(repetition_list)):
            x = repetition_list[r]

            tmax = x['tmax']
            dt = x['dt']
            t_on = min(self.t_on, tmax)
            t_off = min(self.t_off, tmax)

            Vx = x['data']
            Vx = Vx[:,int(t_on/dt):int(t_off/dt)]

            # retreive summary statistics stored in a dictionary
            summ = calc_summ_stats(Vx, tmax, dt)

            burst_lengths = [ summ[nt]['avg_burst_length'] for nt in neutypes ]
            n_triphasic = len(summ['period_data']) * 100

            if self.labels[0] == "cycle_length":
                ret[r] =  np.asarray([ summ['cycle_period'], *burst_lengths, n_triphasic ])
            else:
                ret[r] =  np.asarray([ *burst_lengths, n_triphasic ])

        return ret

class SpikeStats(BaseSummaryStats):
    def __init__(self, t_on, t_off, seed=None):
        super(SpikeStats, self).__init__(seed=seed)
        self.t_on = t_on
        self.t_off = t_off

        self.labels = [ 'cycle_period' ] + [ 'burst_length_{}'.format(nt) for nt in neutypes ] \
                                         + [ 'ibi_length_{}'.format(nt) for nt in neutypes ] \
                                         + [ 'spike_count_{}'.format(nt) for nt in neutypes ] 

        self.n_summary = len(self.labels)

    def calc(self, repetition_list):
        ret = np.empty((len(repetition_list), self.n_summary))

        for r in range(len(repetition_list)):
            x = repetition_list[r]

            tmax = x['tmax']
            dt = x['dt']
            t_on = min(self.t_on, tmax)
            t_off = min(self.t_off, tmax)

            Vx = x['data']
            Vx = Vx[:,int(t_on/dt):int(t_off/dt)]

            # retreive summary statistics stored in a dictionary
            summ = calc_summ_stats(Vx, tmax, dt)

            # create array of summary statistics using the data from the dictionary
            burst_lengths = [ summ[nt]['avg_burst_length'] for nt in neutypes ]
            ibi_lengths = [ summ[nt]['avg_ibi_length'] for nt in neutypes ]
            spike_counts = [ summ[nt]['num_spikes'] for nt in neutypes ]

            ret[r] = np.asarray([ summ['cycle_period'], *burst_lengths, *ibi_lengths, *spike_counts ])

        return ret


classes = { 'PrinzStats' : PrinzStats, 'SpikeStats' : SpikeStats, 'CPGStats' : CPGStats }

# Analyse voltage trace of a single neuron
def analyse_neuron(t, V):
    tmax = t[-1]
    wsize = int(0.5 / (t[1] - t[0]))
    if wsize % 2 == 0:
        wsize += 1

    V = scipy.signal.savgol_filter(V,wsize,3)

    #V = scipy.signal.savgol_filter(V, int(1 / dt), 3)
    # remaining negative slopes are at spike peaks
    spike_inds = np.where((V[1:-1] > spike_thresh) & (np.diff(V[:-1]) >= 0) & (np.diff(V[1:]) <= 0))[0]
    spike_times = t[spike_inds]
    num_spikes = len(spike_times)

    # refractory period begins when slopes start getting positive again
    rebound_inds = np.where((V[1:-1] < spike_thresh) & (np.diff(V[:-1]) <= 0) & (np.diff(V[1:]) >= 0))[0]

    # assign rebounds to the corresponding spikes, save their times in rebound_times
    if len(rebound_inds) == 0:
        rebound_times = np.empty_like(spike_times) * NaN
    else:
        rebound_times = np.empty_like(spike_times)

        # for each spike, find the corresponding rebound (NaN if it doesn't exist)
        for i in range(num_spikes):
            si = spike_inds[i]
            rebound_ind = rebound_inds[np.argmax(rebound_inds > si)]
            if rebound_ind <= si:
                rebound_times[i:] = NaN
                break

            rebound_times[i] = t[rebound_ind]

    # neurons with no spikes are boring
    if len(spike_times) == 0:
        neuron_type = "silent"
        burst_start_times = []
        burst_end_times = []
        burst_times = []

        plateau_length = NaN
        avg_burst_length = NaN
        avg_ibi_length = NaN
        avg_cycle_length = NaN
        avg_spike_length = NaN
    else:
        #assert np.count_nonzero(np.isnan(rebound_times)) <= 1, "Found two nonterminating spikes"

        # calculate average spike lengths, using spikes which have terminated
        last_term_spike_ind = -1 if np.isnan(rebound_times[-1]) else len(spike_times)
        if len(spike_times) == 0 and last_term_spike_ind == -1:         # No terminating spike
            avg_spike_length = NaN
        else:
            avg_spike_length = np.mean(rebound_times[:last_term_spike_ind] - spike_times[:last_term_spike_ind])
        
        # group spikes into bursts, via finding the spikes that are followed by a gap of at least 100 ms
        # The last burst ends at the last spike by convention
        burst_end_spikes = np.append(np.where(np.diff(spike_times) >= ibi_threshold)[0], num_spikes -1)

        # The start of a burst is the first spike after the burst ends, or the first ever spike
        burst_start_spikes = np.insert(burst_end_spikes[:-1] + 1, 0, 0)

        # Find the times of the spikes
        burst_start_times = spike_times[burst_start_spikes]
        burst_end_times = spike_times[burst_end_spikes]

        burst_times = np.stack((burst_start_times, burst_end_times), axis=-1)

        burst_lengths = burst_times.T[1] - burst_times.T[0]

        cond = burst_lengths > bl_thresh

        burst_start_times = burst_start_times[cond]
        burst_end_times= burst_end_times[cond]
        burst_times = burst_times[cond]
        burst_lengths = burst_lengths[cond]

        # PLATEAUS
        # we cluster the voltage into blocks. Each block starts with the current burst's start time and ends with the
        # next burst's start time. Then, extract the longest sequence of values that are larger than plateau_threshold
        # within each block. Lastly, take the mean of those max values. If no plateaus exist, the longest sequence is
        # defined through the length of the action potentials. Thus, if the length does not exceed some threshold
        # we simply set it to 100.

        longest_list = []
        t = np.asarray(t)
        above_th_all = V > plateau_threshold
        stepping = 10 # subsampling for computational speed
        for running_ind in range(len(burst_start_times)):
            if running_ind == len(burst_start_times)-1:
                next_burst_start = spike_times[-1]
            else:
                next_burst_start = burst_start_times[running_ind+1]
            burst_start = burst_start_times[running_ind]
            burst_start_ind = np.where(t == burst_start)[0][0]
            next_burst_start_ind = np.where(t == next_burst_start)[0][0]#

            abouve_th = deepcopy(above_th_all[burst_start_ind:next_burst_start_ind])
            abouve_th = abouve_th[::stepping]
            longest = 0
            current = 0
            for num in abouve_th:
                if num:
                    current += 1
                else:
                    longest = max(longest, current)
                    current = 0
            running_ind += 1
            longest_list.append(longest*stepping)
        plateau_length = np.mean(longest_list)
        if plateau_length < 200: plateau_length=100 # make sure that the duration of a single spike is not a feature
        plateau_length *= (t[1] - t[0]) # convert to ms


        """
        # PLATEAUS option 2
        skew_list = []
        for running_ind in range(len(burst_start_times)):
            burst_start = burst_start_times[running_ind]
            burst_end = burst_end_times[running_ind]
            spike_times = np.asarray(spike_times)
            condition = np.logical_and(spike_times < burst_end, spike_times > burst_start)
            relevant_spikes = spike_times[condition]
            relevant_ISI = np.diff(relevant_spikes)
            burst_skew = scipy.stats.kurtosis(relevant_ISI)
            skew_list.append(burst_skew)
        plateau_length = np.mean(skew_list)
        """

        avg_burst_length = np.average(burst_lengths)

        if len(burst_times) == 1:
            avg_ibi_length = NaN

        else:
            ibi_lengths = burst_times.T[0][1:] - burst_times.T[1][:-1]
            avg_ibi_length = np.mean(ibi_lengths)

        # A neuron is classified as bursting if we can detect multiple bursts and not too many bursts consist
        # of single spikes (to separate bursting neurons from singly spiking neurons)
        if len(burst_times) == 1:
            neuron_type = "non-bursting"
            avg_cycle_length = NaN
        else:
            if len(burst_times) / len(spike_times) >= burst_thresh:
                neuron_type = "bursting"
            else:
                neuron_type = "non-bursting"

            cycle_lengths = np.diff(burst_times.T[0])
            avg_cycle_length = np.average(cycle_lengths)

    # A neuron is classified as silent if it doesn't spike enough and as tonic if it spikes too much
    # Recall that tmax is given in ms
    if len(spike_times) * 1e3 / tmax <= silent_thresh:
        neuron_type = "silent"
    elif len(spike_times) * 1e3 / tmax >= tonic_thresh:
        neuron_type = "tonic"

    return { "neuron_type" : neuron_type, "avg_spike_length" : avg_spike_length,\
             "num_spikes" : num_spikes, "spike_times" : spike_times, "rebound_times" : rebound_times, \
             "burst_start_times" : burst_start_times, "burst_end_times" : burst_end_times, \
             "avg_burst_length" : avg_burst_length, "avg_cycle_length" : avg_cycle_length, \
             "avg_ibi_length" : avg_ibi_length, "plateau_length" : plateau_length }

# Analyse voltage traces; check for triphasic (periodic) behaviour
def analyse_data(data, tmax, dt):
    t = np.arange(0, tmax, dt)
    Vx = data

    assert(len(Vx) == len(neutypes))

    stats = { neutype: analyse_neuron(t, V) for V, neutype in zip(Vx, neutypes) }

    cycle_lengths = np.asarray([ stats[nt]["avg_cycle_length"] for nt in neutypes ])

    # Is this really needed?
    if np.count_nonzero(np.isnan(cycle_lengths)) > 0:
        triphasic = False
        period_data = []

    # if one neuron does not have a periodic rhythm, the whole system is not considered triphasic
    if np.isnan(stats[ref_neuron]["avg_cycle_length"]):
        cycle_period = NaN
        period_times = []

        triphasic = False
        period_data = []
    else:
        # The system period is determined by the periods of a fixed neuron (PM)
        ref_stats = stats[ref_neuron]
        period_times = ref_stats["burst_start_times"]
        cycle_period = np.mean(np.diff(period_times))

        # Analyse the periods, store useful data and check if the neuron is triphasic
        n_periods = len(period_times)
        period_data = []
        period_triphasic = np.zeros(n_periods - 1)
        for i in range(n_periods - 1):
            # The start and end times of the given period, and the starts of the neurons' bursts
            # within this period
            pst, pet = period_times[i], period_times[i+1]
            burst_starts = {}
            burst_ends = {}

            for nt in neutypes:
                bs_nt = stats[nt]["burst_start_times"]
                be_nt = stats[nt]["burst_end_times"]

                if len(bs_nt) == 0:
                    burst_starts[nt] = []
                    burst_ends[nt] = []
                else:
                    cond = (pst <= bs_nt) & (bs_nt < pet)
                    burst_starts[nt] = bs_nt[cond] 
                    burst_ends[nt] = be_nt[cond]

            # A period is classified as triphasic if all neurons start to burst once within the period
            if np.all([ len(burst_starts[nt]) == 1 for nt in neutypes ]):
                period_triphasic[i] = 1
                period_data.append({ nt : (burst_starts[nt], burst_ends[nt]) for nt in neutypes })

        # if we have at least two periods and most of them are triphasic, classify the system as triphasic
        if n_periods >= 2:
            triphasic = np.mean(period_triphasic) >= triphasic_thresh
        else:  
            triphasic = False

    plateau_lengths = []
    for nt in neutypes:
        plateau_lengths.append(stats[nt]["plateau_length"])

    stats.update({ "cycle_period" : cycle_period, "period_times" : period_times, "triphasic" : triphasic,
                   "period_data" : period_data})

    return stats


# Calculate summary information on a system given the voltage traces; classify pyloric-like system
# Michael: The last element here is called 'pyloric_like' and is True or False --> this is used in 'find_pyloric'
def calc_summ_stats(data, tmax, dt, include_pyloric_ness):
    summ = analyse_data(data, tmax, dt)

    burst_durations = {}
    duty_cycles = {}
    start_phases = {}
    starts_to_starts = {}
    ends_to_starts = {}
    phase_gaps = {}
    plateau_lengths = {}

    for nt in neutypes:
        burst_durations[nt] = summ[nt]['avg_burst_length']
        duty_cycles[nt] = burst_durations[nt] / summ["cycle_period"]
        plateau_lengths[nt] = summ[nt]['plateau_length']

        if not summ['triphasic']:
            for nt2 in neutypes:
                ends_to_starts[nt,nt2] = NaN
                phase_gaps[nt,nt2] = NaN
                starts_to_starts[nt,nt2] = NaN

            start_phases[nt] = NaN
        else:
            # triphasic systems are candidate pyloric-like systems, so we collect some information
            for nt2 in neutypes:
                ends_to_starts[nt, nt2] = np.mean([ e[nt2][0] - e[nt][1] for e in summ['period_data'] ])
                phase_gaps[nt, nt2] = ends_to_starts[nt,nt2] / summ['cycle_period']
                starts_to_starts[nt, nt2] = np.mean([ e[nt2][0] - e[nt][0] for e in summ['period_data'] ])

            start_phases[nt] = starts_to_starts[neutypes[0], nt] / summ['cycle_period']
    
    # The three conditions from Prinz' paper must hold (most of the time) for the system to be considered pyloric-like
    pyloric_analysis = np.asarray([ (e[neutypes[1]][0] - e[neutypes[2]][0], e[neutypes[1]][1] -
                                     e[neutypes[2]][1], e[neutypes[0]][1] - e[neutypes[1]][0])
                                    for e in summ['period_data'] ])
    pyloric_like = summ['triphasic'] and np.mean(np.all(pyloric_analysis <= 0, axis=1)) >= pyloric_thresh

    if include_pyloric_ness:
        summ.update({ 'cycle_period' : summ['cycle_period'], 'burst_durations' : burst_durations,
                      'duty_cycles' : duty_cycles, 'start_phases' : start_phases,
                      'starts_to_starts' : starts_to_starts, 'ends_to_starts' : ends_to_starts,
                      'phase_gaps' : phase_gaps, 'plateau_lengths': plateau_lengths,
                      'pyloric_like' : pyloric_like })
    else:
        summ.update({'cycle_period': summ['cycle_period'], 'burst_durations': burst_durations,
                     'duty_cycles': duty_cycles, 'start_phases': start_phases,
                     'starts_to_starts': starts_to_starts, 'ends_to_starts': ends_to_starts,
                     'phase_gaps': phase_gaps, 'plateau_lengths': plateau_lengths })

    # This is just for plotting purposes, a convenience hack
    for nt in neutypes:
        summ[nt]['period_times'] = summ['period_times']

    return summ
