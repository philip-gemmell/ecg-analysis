import sys
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional

import general_analysis

# Add carputils functions (https://git.opencarp.org/openCARP/carputils)
# sys.path.append('/home/pg16/software/carputils/')
sys.path.append('/home/philip/Documents/carputils/')
from carputils.carpio import igb  # type: ignore


def read_ecg_from_igb(phie_file: Union[List[str], str],
                      electrode_file: Optional[str] = None,
                      normalise: bool = True) -> List[Dict[str, np.ndarray]]:
    """Translate the phie.igb file(s) to 10-lead, 12-trace ECG data

    Extracts the complete mesh data from the phie.igb file using CARPutils, before then extracting only those nodes that
    are relevant to the 12-lead ECG, before converting to the ECG itself
    https://carpentry.medunigraz.at/carputils/generated/carputils.carpio.igb.IGBFile.html#carputils.carpio.igb.IGBFile

    Parameters
    ----------
    phie_file : list or str
        Filename for the phie.igb data to extract
    electrode_file : str, optional
        File which contains the node indices in the mesh that correspond to the placement of the leads for the
        10-lead ECG. Default given in get_electrode_phie function.
    normalise : bool, optional
        Whether or not to normalise the ECG signals on a per-lead basis, default=True

    Returns
    -------
    ecg : list(dict)
        List of dictionaries with Vm data for each of the labelled leads (the dictionary keys are the names of the
        leads)
    """

    if isinstance(phie_file, str):
        phie_file = [phie_file]
    data = [data_tmp for data_tmp, _, _ in (igb.read(filename) for filename in phie_file)]

    electrode_data = [get_electrode_phie(data_tmp, electrode_file) for data_tmp in data]

    ecg = [get_ecg_from_electrodes(elec_tmp) for elec_tmp in electrode_data]

    if normalise:
        return [general_analysis.normalise_signal(sim_ecg) for sim_ecg in ecg]
    else:
        return ecg


def read_ecg_from_dat(ecg_files: Union[List[str], str],
                      normalise: bool = True) -> Tuple[List[dict], List[np.ndarray]]:
    """Read ECG data from .dat file

    Parameters
    ----------
    ecg_files : str or list of str
        Name/location of the .dat file to read
    normalise : bool, optional
        Whether or not to normalise the ECG signals on a per-lead basis, default=True

    Returns
    -------
    ecg : dict
        Extracted data for the 12-lead ECG
    times : np.ndarray
        Time data associated with the ECG data
    """
    if isinstance(ecg_files, str):
        ecg_files = [ecg_files]

    ecgs = list()
    times = list()
    for ecg_file in ecg_files:
        ecgdata = np.loadtxt(ecg_file, dtype=float)
        times.append(ecgdata[:, 0])

        ecg = dict()
        # Limb Leads
        ecg['LI'] = ecgdata[:, 1]
        ecg['LII'] = ecgdata[:, 2]
        ecg['LIII'] = ecgdata[:, 3]
        # Augmented leads
        ecg['aVR'] = ecgdata[:, 4]
        ecg['aVL'] = ecgdata[:, 5]
        ecg['aVF'] = ecgdata[:, 6]
        # Precordeal leads
        ecg['V1'] = ecgdata[:, 7]
        ecg['V2'] = ecgdata[:, 8]
        ecg['V3'] = ecgdata[:, 9]
        ecg['V4'] = ecgdata[:, 10]
        ecg['V5'] = ecgdata[:, 11]
        ecg['V6'] = ecgdata[:, 12]

        ecgs.append(ecg)

    if normalise:
        ecgs = [general_analysis.normalise_signal(ecg) for ecg in ecgs]

    return ecgs, times


def read_ecg_from_csv(ecg_files: Union[List[str], str],
                      normalise: bool = True) -> Tuple[List[dict], List[np.ndarray]]:
    """Extract ECG data from CSV file exported from St Jude Medical ECG recording

    Parameters
    ----------
    ecg_files : str or list of str
        Name/location of the .dat file to read
    normalise : bool, optional
        Whether or not to normalise the ECG signals on a per-lead basis, default=True

    Returns
    -------
    ecg : dict
        Extracted data for the 12-lead ECG
    times : np.ndarray
        Time data associated with the ECG data
    """
    if isinstance(ecg_files, str):
        ecg_files = [ecg_files]

    ecgs = list()
    times = list()
    for ecg_file in ecg_files:
        line_count = 0
        with open(ecg_file, 'r') as pFile:
            while True:
                line_count += 1
                line = pFile.readline()
                if 'number of samples' in line.lower():
                    n_rows = int(re.search(r'\d+', line).group())
                    break
                if not line:
                    raise EOFError('Number of Samples entry not found - check file input')
        ecgdata = pd.read_csv(ecg_file, skiprows=line_count, index_col=False)
        ecgdata.drop(ecgdata.tail(1).index, inplace=True)
        n_rows_read, _ = ecgdata.shape
        assert n_rows_read == n_rows, "Mismatch between expected data and read data"

        times.append(ecgdata['t_ref'].values)

        ecg = dict()
        # Limb Leads
        ecg['LI'] = ecgdata['I'].values
        ecg['LII'] = ecgdata['II'].values
        ecg['LIII'] = ecgdata['III'].values
        # Augmented leads
        ecg['aVR'] = ecgdata['aVR'].values
        ecg['aVL'] = ecgdata['aVL'].values
        ecg['aVF'] = ecgdata['aVF'].values
        # Precordeal leads
        ecg['V1'] = ecgdata['V1'].values
        ecg['V2'] = ecgdata['V2'].values
        ecg['V3'] = ecgdata['V3'].values
        ecg['V4'] = ecgdata['V4'].values
        ecg['V5'] = ecgdata['V5'].values
        ecg['V6'] = ecgdata['V6'].values

        ecgs.append(ecg)

    if normalise:
        ecgs = [general_analysis.normalise_signal(ecg) for ecg in ecgs]

    return ecgs, times


def get_electrode_phie(phie_data: np.ndarray, electrode_file: Optional[str] = None) -> dict:
    """Extract phi_e data corresponding to ECG electrode locations

    Parameters
    ----------
    phie_data : np.ndarray
        Numpy array that holds all phie data for all nodes in a given mesh
    electrode_file : str, optional
        File containing entries corresponding to the nodes of the mesh which determine the location of the 10 leads
        for the ECG. Will default to very project specific location. The input text file has each node on a separate
        line (zero-indexed), with the node locations given in order: V1, V2, V3, V4, V5, V6, RA, LA, RL,
        LL. Will default to '/home/pg16/Documents/ecg-scar/ecg-analysis/12LeadElectrodes.dat'

    Returns
    -------
    electrode_data : dict
        Dictionary of phie data for each node, with the dictionary key labelling which node it is.

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


def get_ecg_from_electrodes(electrode_data: dict) -> dict:
    """Converts electrode phi_e data to ECG lead data

    Takes dictionary of phi_e data for 10-lead ECG, and converts these data to standard ECG trace data

    Parameters
    ----------
    electrode_data : dict
        Dictionary with keys corresponding to lead locations

    Returns
    -------
    ecg : dict
        Dictionary with keys corresponding to the ECG traces
    """

    # Wilson Central Terminal
    wct = electrode_data['LA'] + electrode_data['RA'] + electrode_data['LL']

    # V leads
    ecg = dict()
    ecg['V1'] = electrode_data['V1'] - wct / 3
    ecg['V2'] = electrode_data['V2'] - wct / 3
    ecg['V3'] = electrode_data['V3'] - wct / 3
    ecg['V4'] = electrode_data['V4'] - wct / 3
    ecg['V5'] = electrode_data['V5'] - wct / 3
    ecg['V6'] = electrode_data['V6'] - wct / 3

    # Eindhoven limb leads
    ecg['LI'] = electrode_data['LA'] - electrode_data['RA']
    ecg['LII'] = electrode_data['LL'] - electrode_data['RA']
    ecg['LIII'] = electrode_data['LL'] - electrode_data['LA']

    # Augmented leads
    ecg['aVR'] = electrode_data['RA'] - 0.5 * (electrode_data['LA'] + electrode_data['LL'])
    ecg['aVL'] = electrode_data['LA'] - 0.5 * (electrode_data['RA'] + electrode_data['LL'])
    ecg['aVF'] = electrode_data['LL'] - 0.5 * (electrode_data['LA'] + electrode_data['RA'])

    return ecg
