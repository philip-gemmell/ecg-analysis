import sys
import numpy as np  # type: ignore
import scipy
import pandas as pd
import re
from typing import Union, List, Optional, Tuple, Dict

import tools_maths
import tools_python
# import ecg_plot as ep
# import vcg_plot as vp

# Add carputils functions (https://git.opencarp.org/openCARP/carputils)
# sys.path.append('/home/pg16/software/carputils/')
sys.path.append('/home/philip/Documents/carputils/')
from carputils.carpio import igb  # type: ignore


def get_ecg_from_igb(phie_file: Union[List[str], str],
                     electrode_file: Optional[str] = None,
                     normalise: bool = True) -> List[Dict[str, np.ndarray]]:
    """
    Translate the phie.igb file(s) to 10-lead, 12-trace ECG data

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

    ecg = [convert_electrodes_to_ecg(elec_tmp) for elec_tmp in electrode_data]

    if normalise:
        return [normalise_ecg(sim_ecg) for sim_ecg in ecg]
    else:
        return ecg


def get_electrode_phie(phie_data: np.ndarray, electrode_file: Optional[str] = None) -> dict:
    """
    Extract phi_e data corresponding to ECG electrode locations

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


def convert_electrodes_to_ecg(electrode_data: dict) -> dict:
    """
    Converts electrode phi_e data to ECG lead data

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


def get_ecg_from_dat(ecg_files: Union[List[str], str],
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
        ecgs = [normalise_ecg(ecg) for ecg in ecgs]

    return ecgs, times


def get_ecg_from_csv(ecg_files: Union[List[str], str],
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
        ecgs = [normalise_ecg(ecg) for ecg in ecgs]

    return ecgs, times


def normalise_ecg(ecg: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalise ECG leads, so that the maximum value is either 1 or the minimum value is -1

    Parameters
    ----------
    ecg: dict of np.ndarray
        ECG data

    Returns
    -------
    ecg: dict of np.ndarray
        Normalised ECG data
    """

    for key in ecg:
        ecg[key] = np.divide(ecg[key], np.amax(np.absolute(ecg[key])))

    return ecg


def get_ecg_rms(ecgs: List[Dict[str, List[float]]]) -> List[List[float]]:
    """ Calculate the ECG(RMS) of the ECG as a scalar

    Parameters
    ----------
    ecgs: list of dict
        ECG data

    Returns
    -------
    ecgs_rms : list of list of float
        Scalar RMS ECG data

    Notes
    -----
    The scalar RMS is calculated according to

    .. math:: \sqrt{\frac{1}{12}\sum_{i=1}^12 (\textnormal{ECG}_i^2(t))}

    for all leads available from the ECG. This is a slight alteration (for simplicity) on the method presented in
    Hermans et al.

    References
    ----------
    The development and validation of an easy to use automatic QT-interval algorithm
        Hermans BJM, Vink AS, Bennis FC, Filippini LH, Meijborg VMF, Wilde AAM, Pison L, Postema PG, Delhaas T
        PLoS ONE, 12(9), 1–14 (2017)
        https://doi.org/10.1371/journal.pone.0184352
    """

    if isinstance(ecgs, dict):
        ecgs = [ecgs]

    ecgs_rms = list()
    for ecg in ecgs:
        ecg_squares = [[x**2 for x in ecg[key]] for key in ecg]
        ecg_rms = [sum(x) for x in zip(*ecg_squares)]
        ecgs_rms.append([x/9 for x in ecg_rms])
    return ecgs_rms


def get_twave_end(ecgs: Union[List[Dict[str, np.ndarray]], Dict[str, np.ndarray]],
                  leads: Union[str, List[str]] = 'LII',
                  times: Union[List[np.ndarray], np.ndarray, None] = None,
                  dt: Union[List[float], float] = 2,
                  t_end: Union[List[float], float] = 200,
                  i_distance: int = 200,
                  ecg_filter: bool = False,
                  return_median: bool = True,
                  remove_outliers: bool = True,
                  plot_result: bool = False) -> Union[List[float], List[List[float]]]:
    """ Return the time point at which it is estimated that the T-wave has bee completed

    Parameters
    ----------
    ecgs : dict of np.ndarray
        ECG data
    leads : str, optional
        Which lead to check for the T-wave - usually this is either 'LII' or 'V5', but can be set to a list of
        various leads. If set to 'global', then all T-wave values will be calculated. Will return all values unless
        return_average flag is set. Default 'LII'
    times : np.ndarray, optional
        Time data associated with ECG (provided instead of dt, t_end), default=None
    dt : float, optional
        Time interval between recording points in the ECG (provided with t_end instead of time), default=2
    t_end : float, optional
        Duration of the ECG recording (provided with dt instead of time), default=200
    i_distance : int, optional
        Distance between peaks in the gradient, i.e. will direct that the function will only find the points of
        maximum gradient (representing T-wave, etc.) with a minimum distance given here (in terms of indices,
        rather than time). Helps prevent being overly sensitive to 'wobbles' in the signal. Default=200
    ecg_filter: bool, optional
        Whether or not to apply a Butterworth filter to the ECG data to try and simplify the task of finding the
        actual T-wave gradient, default=False
    return_median : bool, optional
        Whether or not to return an average of the leads requested, default=True
    remove_outliers : bool, optional
        Whether to remove T-wave end values that are greater than 1 standard deviation from the mean from the data. Only
        has an effect if more than one lead is provided, and return_average is True. Default=True
    plot_result : bool, optional
        Whether to plot the results or not, default=False

    Returns
    -------
    twave_end : list of float or list of list of float
        Time value for when T-wave is estimated to have ended. This will be returned as either:
            [t1, t2, t3,...] for [ecg1, ecg2, ecg3,...]
                if only a single lead is given or return_average is true
            [[t1a, t1b,...], [t2a, t2b,...],...] for [ecg1, ecg,...] in leads [a, b,...]
                if multiple leads are given and return_average is false

    References
    ----------
    The measurement of the QT interval
        Postema PG, Wilde AA.
        Curr Cardiol Rev. 2014 Aug;10(3):287-94.
        doi: 10.2174/1573403x10666140514103612. PMID: 24827793; PMCID: PMC4040880.
    The development and validation of an easy to use automatic QT-interval algorithm
        Hermans BJM, Vink AS, Bennis FC, Filippini LH, Meijborg VMF, Wilde AAM, Pison L, Postema PG, Delhaas T
        PLoS ONE, 12(9), 1–14 (2017)
        https://doi.org/10.1371/journal.pone.0184352
    """

    if isinstance(ecgs, dict):
        ecgs = [ecgs]

    if leads == 'global':
        leads = ecgs[0].keys()
    elif not isinstance(leads, list):
        leads = [leads]
    for sim_ecg in ecgs:
        for lead in leads:
            assert lead in sim_ecg, 'Lead not present in ECG'
    if len(leads) > 1 and not return_median:
        print("Lead output: {}".format(leads))

    # Extract ECG data for the required leads, then calculate the gradient and the normalised gradient
    if ecg_filter:
        ecgs_leads = [[tools_maths.filter_egm(ecg[lead]) for lead in leads] for ecg in ecgs]
    else:
        ecgs_leads = [[ecg[lead] for lead in leads] for ecg in ecgs]
    times, dt, _ = tools_python.get_time(times, dt, t_end, n_vcg=len(ecgs))
    ecg_grad = [[np.gradient(ecg_lead, dt[0]) for ecg_lead in ecgs_lead] for ecgs_lead in ecgs_leads]
    ecg_grad_norm = [[tools_maths.normalise_signal(ecg_grad_leads) for ecg_grad_leads in ecg_grad_ecgs]
                     for ecg_grad_ecgs in ecg_grad]

    # Find last peak in gradient (with the limitations imposed by only looking for a single peak within the range
    # defined by i_distance, to minimise the effect of 'wibbles in the signal), then by basic trig find the
    # x-intercept (which is used as the T-wave end point)
    # noinspection PyUnresolvedReferences
    i_maxgradient = [[scipy.signal.find_peaks(ecg_grad_norm_lead, distance=i_distance)[0][-1]
                      for ecg_grad_norm_lead in ecg_grad_norm_ecg]
                     for ecg_grad_norm_ecg in ecg_grad_norm]
    twave_end = [[t_ecg[i_max_lead] - (ecg_lead_lead[i_max_lead]/ecg_grad_lead[i_max_lead])
                  for (ecg_lead_lead, ecg_grad_lead, i_max_lead) in zip(ecg_lead_ecg, ecg_grad_ecg, i_max_ecg)]
                 for (t_ecg, ecg_lead_ecg, ecg_grad_ecg, i_max_ecg) in zip(times, ecgs_leads, ecg_grad, i_maxgradient)]

    if plot_result:
        print("Not coded yet")
        # ecg_lead_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'RA', 'LA', 'RL', 'LL']
        # vcg_lead_names = ['x', 'y', 'z']
        # if any(key in ecg_lead_names for key in leads):
        #     figs_ecg = list()
        #     axes_ecg = list()
        #     for ecg in ecgs:
        #         fig_ecg_temp, ax_ecg_temp = ep.plot(ecg)
        #         figs_ecg.append(fig_ecg_temp)
        #         axes_ecg.append(ax_ecg_temp)
        # if any(key in vcg_lead_names for key in leads):
        #     figs_vcg = list()
        #     axes_vcg = list()
        #     for ecg in ecgs:
        #         fig_vcg_temp, ax_vcg_temp = vp.plot_spatial_velocity(ecg.values())
        #         figs_vcg.append(fig_vcg_temp)
        #         axes_vcg.append(ax_vcg_temp)

    if len(leads) == 1:
        twave_end = [twave[0] for twave in twave_end]
    elif return_median:
        twave_end_median = [np.median(twave) for twave in twave_end]
        if remove_outliers:
            for i_sim, sim_twave_end in enumerate(twave_end):
                median_val = np.median(sim_twave_end)
                stddev_val = np.std(sim_twave_end)
                while True:
                    no_outliers = [i for i in sim_twave_end if abs(i - median_val) <= 2 * stddev_val]
                    if len(no_outliers) == len(sim_twave_end):
                        break
                    else:
                        sim_twave_end = no_outliers
                        median_val = np.median(sim_twave_end)
                        stddev_val = np.std(sim_twave_end)
                twave_end_median[i_sim] = median_val
        twave_end = twave_end_median

    return twave_end
