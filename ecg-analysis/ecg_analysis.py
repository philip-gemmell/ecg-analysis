import sys
import numpy as np
import matplotlib.pyplot as plt

import common_analysis

# import matplotlib
# matplotlib.use('Agg')

""" Add carputils functions """
sys.path.append('/home/pg16/software/carputils/')
from carputils.carpio import igb


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


def plot_ecg(ecg, dt=2, legend=None, linewidth=3, qrs_start=None, qrs_end=None):
    """ Plots and labels the ECG data from simulation(s). Optional to add in QRS start/end boundaries for plotting """

    """ Ensure LaTeX rendering is used """

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

    time = [i*dt for i in range(len(ecg[0]['V1']))]
    colours = common_analysis.get_plot_colours(len(ecg))
    for (sim_ecg, sim_label, sim_colour) in zip(ecg, legend, colours):
        for key in plot_sequence:
            ax[key].plot(time, sim_ecg[key], linewidth=linewidth, label=sim_label, color=sim_colour)

    """ Add QRS limits, if supplied. """
    if qrs_start is not None:
        add_limits_to_plot(ax, ecg, qrs_start, colours, plot_sequence)
    if qrs_end is not None:
        add_limits_to_plot(ax, ecg, qrs_end, colours, plot_sequence)

    """ Add legend, title and axis labels """
    if legend[0] is not None:
        plt.rc('text', usetex=True)
        plt.legend(bbox_to_anchor=(1.04, 1.1), loc="center left")

    return fig, ax


def add_limits_to_plot(ax, ecg, limits, colours, plot_sequence):
    """ Plot limits to a given plot (e.g. add line marking start of QRS complex) """
    if not isinstance(limits, list):
        limits = [limits]
    assert len(ecg) == len(limits)
    for (sim_limit, sim_colour) in zip(limits, colours):
        for key in plot_sequence:
            ax[key].axvspan(sim_limit, sim_limit+0.1, color=sim_colour, alpha=0.5)

    return None
