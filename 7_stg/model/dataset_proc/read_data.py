import numpy as np
import experimental_data_utils as edu
import pyabf


class reader:
    def __init__(self, filedir):
        """
        Initialize reader

        :param filedir: string. Relative or absolute path to directory
        """
        self.filedir = filedir
        self.PY_spike_times = []
        self.LP_spike_times = []
        self.PD_spike_times = []
        self.PY_binary = []
        self.LP_binary = []
        self.PD_binary = []
        self.num = []
        self.t_PY = []
        self.t_LP = []
        self.t_PD = []
        self.lpn = []
        self.pdn = []
        self.lvn = []
        self.pyn = []
        self.dt = None
        self.sampling_rate = 0.0001
        self.t_max = 150.0

    def read_preparation(self, num, case='spike', subtract_PD=False):
        """
        Read spike data of one of the two preparations

        :param num: string. Either '0000' or '0001'
        :return: reads spike times for all three neurons
        """
        self.num = num
        if case=='spike':
            self.read_LP_spikes('828_042_2_LP_spikes_'+num+'.txt')
            self.read_PY_spikes('828_042_2_PY_spikes_'+num+'.txt')
            self.read_PD_spikes('828_042_2_PD_spikes_'+num+'.txt')
            if subtract_PD: self.subtract_PD_from_PY('828_042_2_PD_spikes_'+num+'.txt')
        elif case=='voltage':
            self.read_voltage('828_042_2_' + num + '_raw_trace.txt')

    def subtract_PD_from_PY(self, file):
        pass

    def binarize_spike_preparation(self, dt=0.001):
        """
        Binarize one of the two preparations

        :param dt: double, step-size
        :return: time vectors and binarized neuron traces
        """
        self.dt = dt
        self.t_PY, self.PY_binary = self.binarize_spike_data(self.PY_spike_times)
        self.t_LP, self.LP_binary = self.binarize_spike_data(self.LP_spike_times)
        self.t_PD, self.PD_binary = self.binarize_spike_data(self.PD_spike_times)

    def read_spike_data(self, file):
        """
        Read the spike data from a file

        :param file: string. Filename
        :return: data: list. Contains spike times
        """
        file_string = self.filedir + '/' + file

        infile = open(file_string, 'r')
        lines = infile.readlines()
        data = []
        for line in lines:
            try:
                data.append(float(line[:-2]))
            except:
                pass
        data = np.asarray(data)
        return data

    def read_voltage(self, file):
        """
        Read the voltage data from a file

        :param file: string. Filename
        :return: voltage traces. See 828_042_2_NOTES.txt for meaning
        """
        file_string = self.filedir + '/' + file

        infile = open(file_string, 'r')
        lines = infile.readlines()
        data = []
        last_worked = False
        for line in lines:
            try:
                data_per_neuron.append(float(line[:-2]))
                last_worked = True
            except:
                if last_worked: data.append(data_per_neuron)
                data_per_neuron = []
                last_worked = False
                pass
        self.lpn = np.asarray(data[0])
        self.pdn = np.asarray(data[1])
        self.lvn = np.asarray(data[2])
        self.pyn = np.asarray(data[3])

    def binarize_spike_data(self, data):
        """
        Take the spike times and put them into a vector of zeros (no spike) and ones (spike)

        :param data: list, spike times
        :return: t: time vector
        :return: binarized_data: np.array of zeros (no spike) and ones (spike)
        """
        t = np.arange(0, np.max(data), self.dt)
        inds = np.digitize(data, t)
        binarized_data = np.zeros_like(t)
        for ind in inds:
            binarized_data[ind-1] = 1.0
        return t, binarized_data

    def normalize_voltage_data(self):
        """
        Normalize voltage data

        :return: normalized data
        """
        self.lpn = (self.lpn - np.mean(self.lpn)) / np.std(self.lpn)
        self.pdn = (self.pdn - np.mean(self.pdn)) / np.std(self.pdn)
        self.lvn = (self.lvn - np.mean(self.lvn)) / np.std(self.lvn)
        self.pyn = (self.pyn - np.mean(self.pyn)) / np.std(self.pyn)

    def read_LP_spikes(self, file):
        """
        Read data for LP neuron

        :param file: string. Filename of the file to be read
        :return: list. Contains LP spike times
        """
        self.LP_spike_times = self.read_spike_data(file)

    def read_PY_spikes(self, file):
        """
        Read data for PY neuron

        :param file: string. Filename of the file to be read
        :return: list. Contains PY spike times
        """
        self.PY_spike_times = self.read_spike_data(file)

    def read_PD_spikes(self, file):
        """
        Read data for PD neuron

        :param file: string. Filename of the file to be read
        :return: list. Contains PD spike times
        """
        self.PD_spike_times = self.read_spike_data(file)


class ABF_reader(reader):
    def __init__(self, filedir):
        self.type = 'abf'
        super().__init__(filedir)

        abf = pyabf.ABF(filedir)
        channel_data = abf.data

        self.lpn = channel_data[3]
        self.pyn = channel_data[1]
        self.pdn = channel_data[2]
        self.lvn = channel_data[6]

        self.dt = 1 / abf.dataRate
        self.t_LP = np.arange(0, len(self.lpn)) * self.dt
        self.t_PY = np.arange(0, len(self.pyn)) * self.dt
        self.t_PD = np.arange(0, len(self.pdn)) * self.dt

    def read_LP_spikes(self, file):
        self.LP_binary = edu.calc_spikes(self.lpn, threshold=0.5)
        indizes = np.where(self.LP_binary == 1)[-1]
        self.LP_spike_times = indizes * self.dt

    def read_PY_spikes(self, file):
        self.PY_binary = edu.calc_spikes(self.pyn, threshold=10)
        indizes = np.where(self.PY_binary == 1)[-1]
        self.PY_spike_times = indizes * self.dt

    def read_PD_spikes(self, file):
        self.PD_binary = edu.calc_spikes(self.pdn, threshold=20)
        indizes = np.where(self.PD_binary == 1)[-1]
        self.PD_spike_times = indizes * self.dt



class ABF_reader_016(reader):
    def __init__(self, filedir):
        self.type = 'abf'
        super().__init__(filedir)

        abf = pyabf.ABF(filedir)
        channel_data = abf.data

        self.lpn = channel_data[1]
        self.pyn = channel_data[5]
        self.pdn = channel_data[4]
        self.lvn = channel_data[3]

        self.dt = 1 / abf.dataRate
        self.t_LP = np.arange(0, len(self.lpn)) * self.dt
        self.t_PY = np.arange(0, len(self.pyn)) * self.dt
        self.t_PD = np.arange(0, len(self.pdn)) * self.dt

    def read_LP_spikes(self, file):
        self.LP_binary = edu.calc_spikes(self.lpn, threshold=500)
        indizes = np.where(self.LP_binary == 1)[-1]
        self.LP_spike_times = indizes * self.dt

    def read_PY_spikes(self, file):
        self.PY_binary = edu.calc_spikes(self.pyn, threshold=4.1)
        indizes = np.where(self.PY_binary == 1)[-1]
        self.PY_spike_times = indizes * self.dt

    def read_PD_spikes(self, file):
        self.PD_binary = edu.calc_spikes(self.pdn, threshold=50)
        indizes = np.where(self.PD_binary == 1)[-1]
        self.PD_spike_times = indizes * self.dt

    def subtract_PD_from_PY(self, file):
        print(1/self.dt)
        self.PY_binary = edu.subtract_PDfromPY(self.PY_binary, self.PD_binary, vicinity=5, sampling_frequency=1/self.dt)
        indizes = np.where(self.PY_binary == 1)[-1]
        self.PY_spike_times = indizes * self.dt


class ABF_reader_078(reader):
    def __init__(self, filedir):
        self.type = 'abf'
        super().__init__(filedir)

        abf = pyabf.ABF(filedir)
        channel_data = abf.data

        self.lpn = channel_data[6]
        self.pyn = channel_data[5]
        self.pdn = channel_data[7]
        self.lvn = channel_data[4]

        self.dt = 1 / abf.dataRate
        self.t_LP = np.arange(0, len(self.lpn)) * self.dt
        self.t_PY = np.arange(0, len(self.pyn)) * self.dt
        self.t_PD = np.arange(0, len(self.pdn)) * self.dt

    def read_LP_spikes(self, file):
        self.LP_binary = edu.calc_spikes(self.lpn, threshold=50)
        indizes = np.where(self.LP_binary == 1)[-1]
        self.LP_spike_times = indizes * self.dt

    def read_PY_spikes(self, file):
        self.PY_binary = edu.calc_spikes(self.pyn, threshold=3)
        indizes = np.where(self.PY_binary == 1)[-1]
        self.PY_spike_times = indizes * self.dt

    def read_PD_spikes(self, file):
        self.PD_binary = edu.calc_spikes(self.pdn, threshold=.05)
        indizes = np.where(self.PD_binary == 1)[-1]
        self.PD_spike_times = indizes * self.dt

    def subtract_PD_from_PY(self, file):
        print(1/self.dt)
        self.PY_binary = edu.subtract_PDfromPY(self.PY_binary, self.PD_binary, vicinity=5, sampling_frequency=1/self.dt)
        indizes = np.where(self.PY_binary == 1)[-1]
        self.PY_spike_times = indizes * self.dt
