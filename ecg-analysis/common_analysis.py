import numpy as np
import matplotlib.cm as cm
from scipy import signal
from typing import Union, List, Tuple, Optional


def filter_egm(egm: list, sample_freq: Union[int, float] = 500, freq_filter: Union[int, float] = 40, order: int = 2,
               filter_type: str = 'low') -> np.ndarray:
    """
    Filter EGM data (low pass)

    Filter a given set of EGM data using a Butterworth filter, designed to have a specific passband for desired
    frequencies.

    Parameters
    ----------
    egm : list
        Data to filter
    sample_freq : int or float
        Sampling rate of data (Hz), default=500
    freq_filter : int or float
        Cut-off frequency for filter, default=40
    order : int
        Order of the Butterworth filter, default=2
    filter_type : {'low', 'high', 'band'}
        Type of filter to use, default='low'

    Returns
    -------
    filter_out : np.ndarray
        Output filtered data
    """

    # Define filter window (expressed as a fraction of the Nyquist frequency, which is half the sampling rate)
    window = freq_filter/(sample_freq*0.5)

    [b, a] = signal.butter(order, window, filter_type)
    filter_out = signal.filtfilt(b, a, egm)

    return filter_out


def get_plot_colours(n: int = 10, colourmap: str = None) -> list:
    """
    Return iterable list of RGB colour values that can be used for custom plotting functions

    Returns a list of RGB colours values, potentially according to a specified colourmap. If n is low enough, will use
    the custom 'tab10' colourmap by default, which will use alternating colours as much as possible to maximise
    visibility. If n is too big, then the default setting is 'viridis', which should provide a gradation of colour from
    first to last.

    Parameters
    ----------
    n : int, optional
        Number of distinct colours required, default=10
    colourmap : str
        Matplotlib colourmap to base the end result on. Will default to 'tab10' if n<11, 'viridis' otherwise

    Returns
    -------
    list
        List of RGB values
    """
    if colourmap is None:
        if n < 11:
            colourmap = 'tab10'
        else:
            colourmap = 'viridis'

    if n < 11:
        cmap = cm.get_cmap(colourmap)
        return [cmap(i) for i in np.linspace(0, 1, 10)]
    else:
        cmap = cm.get_cmap(colourmap, n)
        return [cmap(i) for i in np.linspace(0, 1, n)]


def recursive_len(item: list):
    """ Return the total number of elements with a potentially nested list """

    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1


def convert_time_to_index(qrs_start: Optional[float, int] = None, qrs_end: Optional[float, int] = None,
                          t_start: Optional[float, int] = 0, t_end: Optional[float, int] = 200,
                          dt: Optional[float, int] = 2) -> Tuple[int, int]:
    """
    Return indices of QRS start and end points. NB: indices returned match Matlab output

    Parameters
    ----------
    qrs_start : float or int, optional
        Start time to convert to index. If not given, will default to the same as the start time of the entire list
    qrs_end : float or int, optional
        End time to convert to index. If not given, will default to the same as the end time of the entire list
    t_start : float or int, optional
         Start time of overall data, default=0
    t_end : float or int, optional
        End time of overall data, default=200
    dt : float or int, optional
        Interval between time points, default=2

    Returns
    -------
    i_qrs_start : int
        Index of start time
    i_qrs_end : int
        Index of end time

    """
    if qrs_start is None:
        qrs_start = t_start
    if qrs_end is None:
        qrs_end = t_end
    x_val = np.array(range(t_start, t_end + dt, dt))
    i_qrs_start = np.where(x_val >= qrs_start)[0][0]
    i_qrs_end = np.where(x_val > qrs_end)[0][0]-1

    return i_qrs_start, i_qrs_end
