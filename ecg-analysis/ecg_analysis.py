import sys
import numpy as np
import matplotlib.pyplot as plt

""" Add carputils functions """
sys.path.append('/home/pg16/software/carputils/')
from carputils.carpio import igb


class ecg:
    """ Class category for ECG data. """

    V1 = None
    V2 = None
    V3 = None
    V4 = None
    V5 = None
    V6 = None
    RA = None
    LA = None
    RL = None
    LL = None


def get_ecg(phie_file, electrode_file=None):
    """ Translate the phie.igb file(s) to ECG data """

    if isinstance(phie_file, str):
        phie_file = [phie_file]
    data = [data_tmp for data_tmp, _, _ in (igb.read(filename) for filename in phie_file)]

    electrode_data = [get_electrode_phie(data_tmp, electrode_file) for data_tmp in data]

    ecg = [convert_electrodes_to_ecg(elec_tmp) for elec_tmp in electrode_data]

    return ecg


def get_electrode_phie(phie_data, electrode_file=None):
    """ Extract phi_e data corresponding to ECG electrode locations """

    """ Import default arguments """
    if electrode_file is None:
        electrode_file = '/home/pg16/Documents/ecg-scar/ecg-analysis/12LeadElectrodes.dat'

    """ Extract node locations for ECG data, then pull data corresponding to those nodes """
    pts_electrodes = np.loadtxt(electrode_file, usecols=(1,), dtype=int)

    electrode_data = {'V1': phie_data[pts_electrodes[0], :],
                      'V2': phie_data[pts_electrodes[1], :],
                      'V3': phie_data[pts_electrodes[2], :],
                      'V4': phie_data[pts_electrodes[3], :],
                      'V5': phie_data[pts_electrodes[4], :],
                      'V6': phie_data[pts_electrodes[5], :],
                      'RA': phie_data[pts_electrodes[6], :],
                      'LA': phie_data[pts_electrodes[7], :],
                      'RL': phie_data[pts_electrodes[8], :],
                      'LL': phie_data[pts_electrodes[9], :]}

    return electrode_data


def convert_electrodes_to_ecg(electrode_data):
    """ Converts electrode phi_e data to ECG lead data """

    """ Wilson Central Terminal """
    wct = electrode_data['LA']+electrode_data['RA']+electrode_data['LL']

    """ V leads """
    ecg = dict()
    ecg['V1'] = electrode_data['V1']-wct/3
    ecg['V2'] = electrode_data['V2']-wct/3
    ecg['V3'] = electrode_data['V3']-wct/3
    ecg['V4'] = electrode_data['V4']-wct/3
    ecg['V5'] = electrode_data['V5']-wct/3
    ecg['V6'] = electrode_data['V6']-wct/3

    """ Eindhoven limb leads """
    ecg['LI'] = electrode_data['LA']-electrode_data['RA']
    ecg['LII'] = electrode_data['LL']-electrode_data['RA']
    ecg['LIII'] = electrode_data['LL']-electrode_data['LA']

    """ Augmented leads """
    ecg['aVR'] = electrode_data['RA']-0.5*(electrode_data['LA']+electrode_data['LL'])
    ecg['aVL'] = electrode_data['LA']-0.5*(electrode_data['RA']+electrode_data['LL'])
    ecg['aVF'] = electrode_data['LL']-0.5*(electrode_data['LA']+electrode_data['RA'])

    return ecg


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
