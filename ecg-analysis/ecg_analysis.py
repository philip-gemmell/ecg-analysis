import sys
import numpy as np

""" Add carputils functions """
sys.path.append('/home/pg16/software/carputils/')
from carputils.carpio import igb


def get_ecg(phie_file, electrode_file=None):
    """
    Translate the phie.igb file(s) to 10-lead, 12-trace ECG data

    Extracts the complete mesh data from the phie.igb file using CARPutils, before then extracting only those nodes that
    are relevant to the 12-lead ECG, before converting to the ECG itself
    https://carpentry.medunigraz.at/carputils/generated/carputils.carpio.igb.IGBFile.html#carputils.carpio.igb.IGBFile

    Input parameters (required):
    ----------------------------

    phie_file       Filename for the phie.igb data to extract

    Input parameters (optional):
    ----------------------------

    electrode_file  None    File which contains the node indices in the mesh that correspond to the placement of the
                            leads for the 10-lead ECG

    Output parameters:
    ------------------

    ecg     Dictionary with Vm data for each of the labelled leads (the dictionary keys are the names of the leads)

    """

    if isinstance(phie_file, str):
        phie_file = [phie_file]
    data = [data_tmp for data_tmp, _, _ in (igb.read(filename) for filename in phie_file)]

    electrode_data = [get_electrode_phie(data_tmp, electrode_file) for data_tmp in data]

    ecg = [convert_electrodes_to_ecg(elec_tmp) for elec_tmp in electrode_data]

    return ecg


def get_electrode_phie(phie_data, electrode_file=None):
    """
    Extract phi_e data corresponding to ECG electrode locations

    Input parameters (required):
    ----------------------------

    phie_data   Numpy array that holds all phie data for all nodes in a given mesh

    Input parameters (optional):
    ----------------------------

    electrode_file      None    File containing entries corresponding to the nodes of the mesh which determine the
                                location of the 10 leads for the ECG. Will default to very project specific location.
                                The input text file has each node on a separate line (zero-indexed), with the node
                                locations given in order: V1, V2, V3, V4, V5, V6, RA, LA, RL, LL

    Output parameters:
    ------------------

    electrode_data  Dictionary of phie data for each node, with the dictionary key labelling which node it is.

    """

    # Import default arguments
    if electrode_file is None:
        electrode_file = '/home/pg16/Documents/ecg-scar/ecg-analysis/12LeadElectrodes.dat'

    # Extract node locations for ECG data, then pull data corresponding to those nodes
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
    """
    Converts electrode phi_e data to ECG lead data

    Takes dictionary of phi_e data for 10-lead ECG, and converts these data to standard ECG trace data

    Input parameters:
    -----------------

    electrode_data  Dictionary with keys corresponding to lead locations

    Output parameters:
    ------------------

    ecg     Dictionary with keys corresponding to the ECG traces
    """

    # Wilson Central Terminal
    wct = electrode_data['LA']+electrode_data['RA']+electrode_data['LL']

    # V leads
    ecg = dict()
    ecg['V1'] = electrode_data['V1']-wct/3
    ecg['V2'] = electrode_data['V2']-wct/3
    ecg['V3'] = electrode_data['V3']-wct/3
    ecg['V4'] = electrode_data['V4']-wct/3
    ecg['V5'] = electrode_data['V5']-wct/3
    ecg['V6'] = electrode_data['V6']-wct/3

    # Eindhoven limb leads
    ecg['LI'] = electrode_data['LA']-electrode_data['RA']
    ecg['LII'] = electrode_data['LL']-electrode_data['RA']
    ecg['LIII'] = electrode_data['LL']-electrode_data['LA']

    # Augmented leads
    ecg['aVR'] = electrode_data['RA']-0.5*(electrode_data['LA']+electrode_data['LL'])
    ecg['aVL'] = electrode_data['LA']-0.5*(electrode_data['RA']+electrode_data['LL'])
    ecg['aVF'] = electrode_data['LL']-0.5*(electrode_data['LA']+electrode_data['RA'])

    return ecg
