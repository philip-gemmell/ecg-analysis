import numpy as np
from typing import List, Dict, Tuple, Union, Optional

import tools_maths
import tools_python


def get_vcg_from_ecg(ecg: Union[List[dict], dict]) -> List[np.ndarray]:
    """Convert ECG data to vectorcardiogram (VCG) data using the Kors matrix method

    Parameters
    ----------
    ecg : list of dict or list
        List of ECG dict data, or ECG dict data directly, with dict keys corresponding to ECG outputs

    Returns
    -------
    vcg: list of np.ndarray
        List of VCG output data

    References
    ----------
    Kors JA, van Herpen G, Sittig AC, van Bemmel JH.
        Reconstruction of the Frank vectorcardiogram from standard electrocardiographic leads: diagnostic comparison
        of different methods
        Eur Heart J. 1990 Dec;11(12):1083-92.
    """

    kors = np.array([[0.38, -0.07, 0.11],
                     [-0.07, 0.93, -0.23],
                     [-0.13, 0.06, -0.43],
                     [0.05, -0.02, -0.06],
                     [-0.01, -0.05, -0.14],
                     [0.14, 0.06, -0.20],
                     [0.06, -0.17, -0.11],
                     [0.54, 0.13, 0.31]])

    if isinstance(ecg, dict):
        ecg = [ecg]

    vcg = list()
    for sim_ecg in ecg:
        ecg_matrix = np.array([sim_ecg['LI'], sim_ecg['LII'], sim_ecg['V1'], sim_ecg['V2'], sim_ecg['V3'],
                               sim_ecg['V4'], sim_ecg['V5'], sim_ecg['V6']])
        vcg.append(np.dot(ecg_matrix.transpose(), kors))

    return vcg


def normalise_signal(ecg: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
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


def get_signal_rms(signals: List[Dict[str, List[float]]]) -> List[List[float]]:
    """Calculate the ECG(RMS) of the ECG as a scalar

    Parameters
    ----------
    signals: list of dict
        ECG data

    Returns
    -------
    signals_rms : list of list of float
        Scalar RMS ECG data

    Notes
    -----
    The scalar RMS is calculated according to

    .. math:: \sqrt{\frac{1}{n}\sum_{i=1}^n (\textnormal{ECG}_i^2(t))}

    for all leads available from the signal (12 for ECG, 3 for VCG). This is a slight alteration (for simplicity) on
    the method presented in Hermans et al.

    References
    ----------
    The development and validation of an easy to use automatic QT-interval algorithm
        Hermans BJM, Vink AS, Bennis FC, Filippini LH, Meijborg VMF, Wilde AAM, Pison L, Postema PG, Delhaas T
        PLoS ONE, 12(9), 1–14 (2017)
        https://doi.org/10.1371/journal.pone.0184352
    """

    if isinstance(signals, dict):
        signals = [signals]

    signals_rms = list()
    for signal in signals:
        ecg_squares = [[x ** 2 for x in signal[key]] for key in signal]
        signal_rms = [sum(x) for x in zip(*ecg_squares)]
        signals_rms.append([x / 9 for x in signal_rms])

    return signals_rms


def get_twave_end(signals: Union[List[Dict[str, np.ndarray]], Dict[str, np.ndarray]],
                  leads: Union[str, List[str]] = 'LII',
                  times: Union[List[np.ndarray], np.ndarray, None] = None,
                  dt: Union[List[float], float] = 2,
                  t_end: Union[List[float], float] = 200,
                  i_distance: int = 200,
                  filter_signal: bool = False,
                  return_median: bool = True,
                  remove_outliers: bool = True,
                  plot_result: bool = False) -> Union[List[float], List[List[float]]]:
    """ Return the time point at which it is estimated that the T-wave has been completed

    Parameters
    ----------
    signals : dict of np.ndarray
        Signal data, either ECG or VCG
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
    filter_signal: bool, optional
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

    if isinstance(signals, dict):
        signals = [signals]

    if leads == 'global':
        leads = signals[0].keys()
    elif not isinstance(leads, list):
        leads = [leads]
    for signal in signals:
        for lead in leads:
            assert lead in signal, 'Lead not present in ECG'
    if len(leads) > 1 and not return_median:
        print("Lead output: {}".format(leads))

    # Extract ECG data for the required leads, then calculate the gradient and the normalised gradient
    if filter_signal:
        signals_leads = [[tools_maths.filter_egm(signal[lead]) for lead in leads] for signal in signals]
    else:
        signals_leads = [[signal[lead] for lead in leads] for signal in signals]
    times, dt, _ = tools_python.get_time(times, dt, t_end, n_vcg=len(signals))
    signals_grad = [[np.gradient(signal_lead, dt[0]) for signal_lead in signals_lead] for signals_lead in signals_leads]
    signals_grad_norm = [[tools_maths.normalise_signal(signal_grad_lead) for signal_grad_lead in signal_grad]
                         for signal_grad in signals_grad]

    # Find last peak in gradient (with the limitations imposed by only looking for a single peak within the range
    # defined by i_distance, to minimise the effect of 'wibbles in the signal), then by basic trig find the
    # x-intercept (which is used as the T-wave end point)
    # noinspection PyUnresolvedReferences
    i_maxgrad = [[scipy.signal.find_peaks(signal_grad_norm_lead, distance=i_distance)[0][-1]
                  for signal_grad_norm_lead in signal_grad_norm]
                 for signal_grad_norm in signals_grad_norm]
    twave_end = [[t_signal[i_max_lead] - (signal_lead[i_max_lead] / signal_grad_lead[i_max_lead])
                  for (signal_lead, signal_grad_lead, i_max_lead) in zip(signal_leads, signal_grad, i_max_signal)]
                 for (t_signal, signal_leads, signal_grad, i_max_signal) in zip(times, signals_leads, signals_grad,
                                                                                i_maxgrad)]

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
