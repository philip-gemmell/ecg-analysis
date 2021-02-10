import numpy as np
import scipy.signal
import pandas as pd
from typing import List, Dict, Union, Optional

import tools_maths


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


def get_twave_end(ecgs: Union[List[pd.DataFrame], pd.DataFrame],
                  leads: Union[str, List[str]] = 'LII',
                  i_distance: int = 200,
                  filter_signal: Optional[str] = None,
                  return_median: bool = True,
                  remove_outliers: bool = True,
                  plot_result: bool = False) -> List[pd.DataFrame]:
    """ Return the time point at which it is estimated that the T-wave has been completed

    Parameters
    ----------
    ecgs : pd.DataFrame or list of pd.DataFrame
        Signal data, either ECG or VCG
    leads : str, optional
        Which lead to check for the T-wave - usually this is either 'LII' or 'V5', but can be set to a list of
        various leads. If set to 'global', then all T-wave values will be calculated. Will return all values unless
        return_average flag is set. Default 'LII'
    i_distance : int, optional
        Distance between peaks in the gradient, i.e. will direct that the function will only find the points of
        maximum gradient (representing T-wave, etc.) with a minimum distance given here (in terms of indices,
        rather than time). Helps prevent being overly sensitive to 'wobbles' in the ecg. Default=200
    filter_signal: {'butterworth', 'savitzky-golay'}, optional
        Whether or not to apply a filter to the data prior to trying to find the actual T-wave gradient. Can pass 
        either a Butterworth filter or a Savitzky-Golay filter, in which case the required kwargs for each can be 
        provided. Default=None (no filter applied)
    return_median : bool, optional
        Whether or not to return an average of the leads requested, default=True
    remove_outliers : bool, optional
        Whether to remove T-wave end values that are greater than 1 standard deviation from the mean from the data. Only
        has an effect if more than one lead is provided, and return_average is True. Default=True
    plot_result : bool, optional
        Whether to plot the results or not, default=False

    Returns
    -------
    twave_ends : list of float or list of list of float
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

    if isinstance(ecgs, pd.DataFrame):
        ecgs = [ecgs]

    if leads == 'global':
        leads = ecgs[0].columns
    elif not isinstance(leads, list):
        leads = [leads]

    for ecg in ecgs:
        for lead in leads:
            assert lead in ecg, 'Lead not present in ECG'
    
    if filter_signal is not None:
        assert filter_signal in ['butterworth', 'savitzky-golay'], "Unknown value for filter_signal passed"

    # Extract ECG data for the required leads, then calculate the gradient and the normalised gradient
    ecgs_leads = [ecg[leads] for ecg in ecgs]
    if filter_signal == 'butterworth':
        ecgs_leads = [tools_maths.filter_butterworth(ecg) for ecg in ecgs_leads]
    elif filter_signal == 'savitzky-golay':
        ecgs_leads = [tools_maths.filter_savitzkygolay(ecg) for ecg in ecgs_leads]
    ecgs_grad = [pd.DataFrame(index=ecg.index, columns=ecg.columns) for ecg in ecgs_leads]
    ecgs_grad_normalised = [pd.DataFrame(index=ecg.index, columns=ecg.columns) for ecg in ecgs_leads]
    for i_ecg, ecg in enumerate(ecgs_leads):
        for col in ecg:
            ecgs_grad[i_ecg].loc[:, col] = np.gradient(ecg[col], ecg.index)
            ecgs_grad_normalised[i_ecg].loc[:, col] = tools_maths.normalise_signal(np.gradient(ecg[col], ecg.index))

    # Find last peak in gradient (with the limitations imposed by only looking for a single peak within the range
    # defined by i_distance, to minimise the effect of 'wibbles in the ecg), then by basic trig find the
    # x-intercept (which is used as the T-wave end point)
    # noinspection PyUnresolvedReferences
    i_tpeak = [pd.DataFrame(columns=ecg.columns, index=[0]) for ecg in ecgs_leads]
    i_tpeak_full = [pd.DataFrame(columns=ecg.columns, index=[0]) for ecg in ecgs_leads]
    twave_ends = [pd.DataFrame(columns=ecg.columns, index=[0]) for ecg in ecgs_leads]
    for i_ecg, ecg in enumerate(ecgs_leads):
        for col in ecg:
            i_tpeak_temp = scipy.signal.find_peaks(ecgs_grad_normalised[i_ecg][col], distance=i_distance)[0][-1]
            i_tpeak_full_temp = scipy.signal.find_peaks(ecgs_grad_normalised[i_ecg][col], distance=i_distance)[0]
            t_tpeak_temp = ecg.index[i_tpeak_temp]
            twave_end_temp = t_tpeak_temp - (ecg.loc[t_tpeak_temp, col] / ecgs_grad[i_ecg].loc[t_tpeak_temp, col])
            i_tpeak[i_ecg].loc[0, col] = i_tpeak_temp
            i_tpeak_full[i_ecg].loc[0, col] = i_tpeak_full_temp
            twave_ends[i_ecg].loc[0, col] = twave_end_temp

    if return_median:
        for i_twave, twave_end in enumerate(twave_ends):
            twave_end_median = np.median(twave_end)
            if remove_outliers:
                twave_end_std = np.std(twave_end.values)
                while True:
                    no_outliers = pd.DataFrame(np.abs((twave_end-twave_end_median)) < 2*twave_end_std)
                    if all(no_outliers.values[0]):
                        break
                    else:
                        twave_end = twave_end[no_outliers]
                        twave_end.dropna(axis='columns', inplace=True)
                        twave_end_median = np.median(twave_end)
                        twave_end_std = np.std(twave_end.values)
            twave_ends[i_twave].loc[0, 'median'] = twave_end_median

    if plot_result:
        import matplotlib.pyplot as plt
        import tools_plotting
        colours = tools_plotting.get_plot_colours(len(ecgs))
        for (ecg, twave_end, i_peak, i_peak_full, ecg_grad_normalised, colour) in zip(ecgs, twave_ends, i_tpeak,
                                                                                      i_tpeak_full,
                                                                                      ecgs_grad_normalised, colours):
            fig = plt.figure()
            axes = dict()
            i_ax = 1
            for lead in leads:
                axes[lead] = fig.add_subplot(2, 6, i_ax)
                axes[lead].set_title(lead)
                i_ax += 1
                axes[lead].plot(ecg.index, ecg[lead], color=colour, linestyle='-')
                axes[lead].plot(ecg.index, ecg_grad_normalised[lead], color=colour, linestyle='--')
                axes[lead].axhline(0, color='k')
                axes[lead].axvline(twave_end[lead].values, color='k')
                t_peak = ecg.index[i_peak[lead].values[0]]
                t_peak_full = ecg.index[i_peak_full[lead][0]]
                axes[lead].plot(t_peak_full, ecg[lead][t_peak_full], 's', markerfacecolor='none', markeredgecolor='g')
                axes[lead].plot(t_peak, ecg[lead][t_peak], 'o', markerfacecolor='none', markeredgecolor='r')

    return twave_ends
