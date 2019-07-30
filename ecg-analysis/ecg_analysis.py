import numpy as np
import matplotlib.pyplot as plt

""" Add carputils functions (if not already present) """
try:
    from carputils.carpio import igb
except ImportError:
    import sys
    sys.path.append('/home/pg16/software/carputils/')
    from carputils.carpio import igb


class ECGClass:
    """ Class category for ECG data. Public variables for the ECG traces for each of the leads, and the filename.
        Private variables for all the data used to derive (including full phie.igb data - might need to rewrite to
        discard data if the variables get too big... """

    def __init__(self, phie_filename=None, electrode_filename=None):
        """ Assign values to ECG lead data. If filename is provided at time of instantiation, complete calculatons. """

        # Source data, including data that we extract before using
        self.phie_filename = phie_filename
        self.electrode_filename = electrode_filename
        self.__phie_v1 = None
        self.__phie_v2 = None
        self.__phie_v3 = None
        self.__phie_v4 = None
        self.__phie_v5 = None
        self.__phie_v6 = None
        self.__phie_ra = None
        self.__phie_la = None
        self.__phie_rl = None
        self.__phie_ll = None

        # V leads
        self.V1 = None
        self.V2 = None
        self.V3 = None
        self.V4 = None
        self.V5 = None
        self.V6 = None

        # Eindhoven limb leads
        self.LI = None
        self.LII = None
        self.LIII = None

        # Augmented leads
        self.aVR = None
        self.aVL = None
        self.aVF = None

        if phie_filename is not None:
            self.get_ecg(phie_filename, electrode_filename)

    def __repr__(self):
        return "ECGClass("+self.phie_filename+", "+self.electrode_filename+")"

    def __str__(self):
        return "ECG data for "+self.phie_filename+", using electrode data from "+self.electrode_filename

    def __getitem__(self, key):
        return self.__dict__[key]

    def get_ecg(self, phie_filename, electrode_filename=None):
        """ Extract ECG data from a given file """

        self.get_electrode_phie_data(phie_filename, electrode_filename)
        self.convert_phie_to_ecg()

        return None

    def get_electrode_phie_data(self, phie_filename, electrode_filename=None):
        """ Set phie.igb filename to reference, then read data """

        """ Extract full phie data """
        self.phie_filename = phie_filename
        phie_data, _, _ = igb.read(phie_filename)

        if electrode_filename is None:
            electrode_filename = '/home/pg16/Documents/ecg-scar/ecg-analysis/12LeadElectrodes.dat'
        pts_electrodes = np.loadtxt(electrode_filename, usecols=(1,), dtype=int)

        self.__phie_v1 = phie_data[pts_electrodes[0], :]
        self.__phie_v2 = phie_data[pts_electrodes[1], :]
        self.__phie_v3 = phie_data[pts_electrodes[2], :]
        self.__phie_v4 = phie_data[pts_electrodes[3], :]
        self.__phie_v5 = phie_data[pts_electrodes[4], :]
        self.__phie_v6 = phie_data[pts_electrodes[5], :]
        self.__phie_ra = phie_data[pts_electrodes[6], :]
        self.__phie_la = phie_data[pts_electrodes[7], :]
        self.__phie_rl = phie_data[pts_electrodes[8], :]
        self.__phie_ll = phie_data[pts_electrodes[9], :]

        del phie_data

        return None

    def convert_phie_to_ecg(self):
        """ Converts electrode phi_e data to ECG lead data """
        """ Wilson Central Terminal """
        wct = self.__phie_la + self.__phie_ra + self.__phie_ll

        """ V leads """
        self.V1 = self.__phie_v1 - wct / 3
        self.V2 = self.__phie_v2 - wct / 3
        self.V3 = self.__phie_v3 - wct / 3
        self.V4 = self.__phie_v4 - wct / 3
        self.V5 = self.__phie_v5 - wct / 3
        self.V6 = self.__phie_v6 - wct / 3

        """ Eindhoven limb leads """
        self.LI = self.__phie_la - self.__phie_ra
        self.LII = self.__phie_ll - self.__phie_ra
        self.LIII = self.__phie_ll - self.__phie_la

        """ Augmented leads """
        self.aVR = self.__phie_ra - 0.5*(self.__phie_la + self.__phie_ll)
        self.aVL = self.__phie_la - 0.5*(self.__phie_ra + self.__phie_ll)
        self.aVF = self.__phie_ll - 0.5*(self.__phie_la + self.__phie_ra)

        return None


def get_ecg(phie_file, electrode_file=None):
    """ Translate the phie.igb file(s) to ECG data """

    if isinstance(phie_file, str):
        phie_file = [phie_file]

    return [ECGClass(file, electrode_file) for file in phie_file]


def plot_ecg(ecg, legend=None, linewidth=3):
    """ Plots and labels the ECG data from simulation(s) """

    """ Initialise figure and axis handles """
    fig = plt.figure()
    i = 1
    ax = dict()
    plot_sequence = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'LI', 'LII', 'LIII', 'aVR', 'aVL', 'aVF']
    for key in plot_sequence:
        ax[key] = plt.subplot(2, 6, i)
        ax[key].set_title(key)
        i += 1

    """ Plot ECG data, potentially for several different simulations (ensure that ECG data is presented as a list, 
        as in the instance where several simulations are being plotted) """
    if not isinstance(ecg, list):
        ecg = [ecg]
    if legend is None:
        legend = [None for _ in range(len(ecg))]
    else:
        assert(len(legend) == len(ecg))
    for (sim_ecg, sim_label) in zip(ecg, legend):
        for key in plot_sequence:
            ax[key].plot(sim_ecg[key], linewidth=linewidth, label=sim_label)

    """ Add legend, title and axis labels """
    if legend[0] is not None:
        plt.legend(bbox_to_anchor=(1.04, 1.1), loc="center left")

    return fig, ax
