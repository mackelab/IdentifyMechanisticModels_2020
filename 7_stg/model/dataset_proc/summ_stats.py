import numpy as np
import pdb
import prinzdb
import scipy
import scipy.signal
from copy import deepcopy

from delfi.summarystats.BaseSummaryStats import BaseSummaryStats

neutypes = prinzdb.neutypes
ref_neuron = neutypes[0]

silent_thresh = 1  # minimum number of spikes per second for a neuron not to be considered silent
tonic_thresh = 30  # minimum number of spikes per second for a neuron to be considered tonically firing
burst_thresh = 0.5  # maximum percentage of single-spike bursts for a neuron to be considered
# bursting (and not simply spiking)
bl_thresh = 40  # minimum number of milliseconds for bursts not to be discounted as single spikes
spike_thresh = -10  # minimum voltage above which spikes are considered
ibi_threshold = 150  # maximum time between spikes in a given burst (ibi = inter-burst interval)

triphasic_thresh = 0.9  # percentage of triphasic periods for system to be considered triphasic
pyloric_thresh = 0.7  # percentage of pyloric periods for triphasic system to be pyloric_like

NaN = float("nan")
inf = float("inf")


# Use this instead of import from delfi if delfi is not installed
# class BaseSummaryStats:
#     def __init__(self, *args, **kwargs):
#         pass


class PrinzExperimStats(BaseSummaryStats):
    def __init__(self, include_pyloric_ness=True, include_plateaus=False, seed=None):
        """See SummaryStats.py for docstring"""
        super().__init__(seed=seed)
        self.include_pyloric_ness = include_pyloric_ness
        self.include_plateaus = include_plateaus

        self.labels = ['cycle_period'] + ['burst_length_{}'.format(nt) for nt in neutypes] \
                      + ['end_to_start_{}_{}'.format(neutypes[i], neutypes[j])
                         for i, j in ((0, 1), (1, 2))] \
                      + ['start_to_start_{}_{}'.format(neutypes[i], neutypes[j])
                         for i, j in ((0, 1), (0, 2))] \
                      + ['duty_cycle_{}'.format(nt) for nt in neutypes] \
                      + ['phase_gap_{}_{}'.format(neutypes[i], neutypes[j])
                         for i, j in ((0, 1), (1, 2))] \
                      + ['phase_{}'.format(neutypes[i]) for i in (1, 2)]
        if self.include_plateaus:
            self.labels += ['plateau_length_{}'.format(nt) for nt in neutypes]
        if self.include_pyloric_ness:
            self.labels += ['pyloric_like']


        self.n_summary = len(self.labels)

    def calc(self, reader):

        tmax = np.min([reader.t_PD[-1], reader.t_LP[-1], reader.t_PY[-1]])
        dt   = reader.dt

        #Vx   = [reader.PD_binary, reader.LP_binary, reader.PY_binary]
        Vx   = [reader.PD_spike_times, reader.LP_spike_times, reader.PY_spike_times]

        # retreive summary statistics stored in a dictionary
        summ = calc_summ_stats(Vx, tmax, dt, self.include_pyloric_ness)

        # create array of summary statistics using the data from the dictionary
        duty_cycles = [summ['duty_cycles'][nt] for nt in neutypes]
        burst_lengths = [summ[nt]['avg_burst_length'] for nt in neutypes]

        gaps = [summ['ends_to_starts'][neutypes[i], neutypes[j]] for i, j in ((0, 1), (1, 2))]
        delays = [summ['starts_to_starts'][neutypes[i], neutypes[j]] for i, j in ((0, 1), (0, 2))]

        phase_gaps = [summ['phase_gaps'][neutypes[i], neutypes[j]] for i, j in ((0, 1), (1, 2))]
        phases = [summ['start_phases'][neutypes[i]] for i in (1, 2)]

        plateau_lengths = [summ['plateau_lengths'][nt] for nt in neutypes]

        if self.include_pyloric_ness and self.include_plateaus:
            outs = np.asarray([summ['cycle_period'], *burst_lengths, *gaps, *delays, *duty_cycles,
                                 *phase_gaps, *phases, *plateau_lengths, summ['pyloric_like']])
        if self.include_pyloric_ness and not self.include_plateaus:
            outs = np.asarray([summ['cycle_period'], *burst_lengths, *gaps, *delays, *duty_cycles,
                                 *phase_gaps, *phases, summ['pyloric_like']])
        if not self.include_pyloric_ness and self.include_plateaus:
            outs = np.asarray([summ['cycle_period'], *burst_lengths, *gaps, *delays, *duty_cycles,
                                 *phase_gaps, *phases, *plateau_lengths])
        if not self.include_pyloric_ness and not self.include_plateaus:
            outs = np.asarray([summ['cycle_period'], *burst_lengths, *gaps, *delays, *duty_cycles,
                                 *phase_gaps, *phases])

        return outs

    @staticmethod
    def print_PrinzStats(summ_stats, percentage=False):
        if percentage:
            p = 100
            p_str = '%'
        else:
            p = 1
            p_str = ''
        print("----- Summary Statistics -----")
        print("Cycle_period:          ", summ_stats[0] * p, p_str, sep='')
        print("Burst_length AB:       ", summ_stats[1] * p, p_str, sep='')
        print("Burst_length LP:       ", summ_stats[2] * p, p_str, sep='')
        print("Burst_length PY:       ", summ_stats[3] * p, p_str, sep='')
        print("End_to_start AB-LP:    ", summ_stats[4] * p, p_str, sep='')
        print("End_to_start LP-PY:    ", summ_stats[5] * p, p_str, sep='')
        print("Start_to_start AB-LP:  ", summ_stats[6] * p, p_str, sep='')
        print("Start_to_start LP-PY:  ", summ_stats[7] * p, p_str, sep='')
        print("Duty cycle AB:         ", summ_stats[8] * p, p_str, sep='')
        print("Duty cycle LP:         ", summ_stats[9] * p, p_str, sep='')
        print("Duty cycle PY:         ", summ_stats[10] * p, p_str, sep='')
        print("Phase gap AB-LP:       ", summ_stats[11] * p, p_str, sep='')
        print("Phase gap LP-PY:       ", summ_stats[12] * p, p_str, sep='')
        print("Phase LP:              ", summ_stats[13] * p, p_str, sep='')
        print("Phase PY:              ", summ_stats[14] * p, p_str, sep='')


class SpikeStats(BaseSummaryStats):
    def __init__(self, t_on, t_off, seed=None):
        super(SpikeStats, self).__init__(seed=seed)
        self.t_on = t_on
        self.t_off = t_off

        self.labels = ['cycle_period'] + ['burst_length_{}'.format(nt) for nt in neutypes] \
                      + ['ibi_length_{}'.format(nt) for nt in neutypes] \
                      + ['spike_count_{}'.format(nt) for nt in neutypes]

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
            Vx = Vx[:, int(t_on / dt):int(t_off / dt)]

            # retreive summary statistics stored in a dictionary
            summ = calc_summ_stats(Vx, tmax, dt)

            # create array of summary statistics using the data from the dictionary
            burst_lengths = [summ[nt]['avg_burst_length'] for nt in neutypes]
            ibi_lengths = [summ[nt]['avg_ibi_length'] for nt in neutypes]
            spike_counts = [summ[nt]['num_spikes'] for nt in neutypes]

            ret[r] = np.asarray([summ['cycle_period'], *burst_lengths, *ibi_lengths, *spike_counts])

        return ret


class CPGStats(BaseSummaryStats):
    def __init__(self, t_on, t_off, with_cycle_length=False, seed=None):
        """See SummaryStats.py for docstring"""
        super(CPGStats, self).__init__(seed=seed)
        self.t_on = t_on
        self.t_off = t_off

        self.labels = ["avg_burst_length_{}".format(nt) for nt in neutypes] + ["n_triphasic"]
        if with_cycle_length:
            self.labels = ["cycle_length"] + self.labels

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
            Vx = Vx[:, int(t_on / dt):int(t_off / dt)]

            # retreive summary statistics stored in a dictionary
            summ = calc_summ_stats(Vx, tmax, dt)

            burst_lengths = [summ[nt]['avg_burst_length'] for nt in neutypes]
            n_triphasic = len(summ['period_data']) * 100

            if self.labels[0] == "cycle_length":
                ret[r] = np.asarray([summ['cycle_period'], *burst_lengths, n_triphasic])
            else:
                ret[r] = np.asarray([*burst_lengths, n_triphasic])

        return ret


class SpikeStats(BaseSummaryStats):
    def __init__(self, t_on, t_off, seed=None):
        super(SpikeStats, self).__init__(seed=seed)
        self.t_on = t_on
        self.t_off = t_off

        self.labels = ['cycle_period'] + ['burst_length_{}'.format(nt) for nt in neutypes] \
                      + ['ibi_length_{}'.format(nt) for nt in neutypes] \
                      + ['spike_count_{}'.format(nt) for nt in neutypes]

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
            Vx = Vx[:, int(t_on / dt):int(t_off / dt)]

            # retreive summary statistics stored in a dictionary
            summ = calc_summ_stats(Vx, tmax, dt)

            # create array of summary statistics using the data from the dictionary
            burst_lengths = [summ[nt]['avg_burst_length'] for nt in neutypes]
            ibi_lengths = [summ[nt]['avg_ibi_length'] for nt in neutypes]
            spike_counts = [summ[nt]['num_spikes'] for nt in neutypes]

            ret[r] = np.asarray([summ['cycle_period'], *burst_lengths, *ibi_lengths, *spike_counts])

        return ret


# Analyse voltage trace of a single neuron
def analyse_neuron(t, spike_times):
    tmax = t[-1]
    wsize = int(0.5 / (t[1] - t[0]))
    if wsize % 2 == 0:
        wsize += 1

    spike_times_ms = deepcopy(spike_times) * 1000.0

    # V = scipy.signal.savgol_filter(V, int(1 / dt), 3)
    # remaining negative slopes are at spike peaks
    num_spikes = len(spike_times_ms)

    avg_spike_length = 1

    # group spikes into bursts, via finding the spikes that are followed by a gap of at least 100 ms
    # The last burst ends at the last spike by convention
    burst_end_spikes = np.append(np.where(np.diff(spike_times_ms) >= ibi_threshold)[0], num_spikes - 1)

    # The start of a burst is the first spike after the burst ends, or the first ever spike
    burst_start_spikes = np.insert(burst_end_spikes[:-1] + 1, 0, 0)

    # Find the times of the spikes
    burst_start_times = spike_times_ms[burst_start_spikes]
    burst_end_times = spike_times_ms[burst_end_spikes]

    burst_times = np.stack((burst_start_times, burst_end_times), axis=-1)

    burst_lengths = burst_times.T[1] - burst_times.T[0]

    cond = burst_lengths > bl_thresh

    burst_start_times = burst_start_times[cond]
    burst_end_times = burst_end_times[cond]
    burst_times = burst_times[cond]
    burst_lengths = burst_lengths[cond]

    # PLATEAUS
    # no plateaus for real data
    plateau_length = 2.5

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
        if len(burst_times) / len(spike_times_ms) >= burst_thresh:
            neuron_type = "bursting"
        else:
            neuron_type = "non-bursting"

        cycle_lengths = np.diff(burst_times.T[0])
        avg_cycle_length = np.average(cycle_lengths)

    # A neuron is classified as silent if it doesn't spike enough and as tonic if it spikes too much
    # Recall that tmax is given in ms
    if len(spike_times_ms) * 1e3 / tmax <= silent_thresh:
        neuron_type = "silent"
    elif len(spike_times_ms) * 1e3 / tmax >= tonic_thresh:
        neuron_type = "tonic"

    return {"neuron_type": neuron_type, "avg_spike_length": avg_spike_length, \
            "num_spikes": num_spikes, "spike_times": spike_times_ms, \
            "burst_start_times": burst_start_times, "burst_end_times": burst_end_times, \
            "avg_burst_length": avg_burst_length, "avg_cycle_length": avg_cycle_length,
            "avg_ibi_length": avg_ibi_length, "plateau_length" : plateau_length }


# Analyse voltage traces; check for triphasic (periodic) behaviour
def analyse_data(data, tmax, dt):
    t = np.arange(0, tmax, dt)
    Vx = data

    assert (len(Vx) == len(neutypes))

    stats = {neutype: analyse_neuron(t, V) for V, neutype in zip(Vx, neutypes)}

    cycle_lengths = np.asarray([stats[nt]["avg_cycle_length"] for nt in neutypes])

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
            pst, pet = period_times[i], period_times[i + 1]
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
            if np.all([len(burst_starts[nt]) == 1 for nt in neutypes]):
                period_triphasic[i] = 1
                period_data.append({nt: (burst_starts[nt], burst_ends[nt]) for nt in neutypes})

        # if we have at least two periods and most of them are triphasic, classify the system as triphasic
        if n_periods >= 2:
            triphasic = np.mean(period_triphasic) >= triphasic_thresh
        else:
            triphasic = False

    plateau_lengths = []
    for nt in neutypes:
        plateau_lengths.append(stats[nt]["plateau_length"])

    stats.update({"cycle_period": cycle_period, "period_times": period_times, "triphasic": triphasic,
                  "period_data": period_data})

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
                ends_to_starts[nt, nt2] = NaN
                phase_gaps[nt, nt2] = NaN
                starts_to_starts[nt, nt2] = NaN

            start_phases[nt] = NaN
        else:
            # triphasic systems are candidate pyloric-like systems, so we collect some information
            for nt2 in neutypes:
                ends_to_starts[nt, nt2] = np.mean([e[nt2][0] - e[nt][1] for e in summ['period_data']])
                phase_gaps[nt, nt2] = ends_to_starts[nt, nt2] / summ['cycle_period']
                starts_to_starts[nt, nt2] = np.mean([e[nt2][0] - e[nt][0] for e in summ['period_data']])

            start_phases[nt] = starts_to_starts[neutypes[0], nt] / summ['cycle_period']

    # The three conditions from Prinz' paper must hold (most of the time) for the system to be considered pyloric-like
    pyloric_analysis = np.asarray([(e[neutypes[1]][0] - e[neutypes[2]][0], e[neutypes[1]][1] -
                                    e[neutypes[2]][1], e[neutypes[0]][1] - e[neutypes[1]][0])
                                   for e in summ['period_data']])
    pyloric_like = summ['triphasic'] and np.mean(np.all(pyloric_analysis <= 0, axis=1)) >= pyloric_thresh

    if include_pyloric_ness:
        summ.update({'cycle_period': summ['cycle_period'], 'burst_durations': burst_durations,
                     'duty_cycles': duty_cycles, 'start_phases': start_phases,
                     'starts_to_starts': starts_to_starts, 'ends_to_starts': ends_to_starts,
                     'phase_gaps': phase_gaps, 'plateau_lengths': plateau_lengths,
                     'pyloric_like': pyloric_like})
    else:
        summ.update({'cycle_period': summ['cycle_period'], 'burst_durations': burst_durations,
                     'duty_cycles': duty_cycles, 'start_phases': start_phases,
                     'starts_to_starts': starts_to_starts, 'ends_to_starts': ends_to_starts,
                     'phase_gaps': phase_gaps, 'plateau_lengths': plateau_lengths})

    # This is just for plotting purposes, a convenience hack
    for nt in neutypes:
        summ[nt]['period_times'] = summ['period_times']

    return summ
