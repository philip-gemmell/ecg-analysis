import numpy as np  # type: ignore
import matplotlib.cm as cm  # type: ignore
from scipy import signal  # type: ignore
from typing import List, Tuple, Optional, Union


def filter_egm(egm: np.ndarray,
               sample_freq: float = 500,
               freq_filter: float = 40,
               order: int = 2,
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


def get_plot_colours(n: int = 10, colourmap: Optional[str] = None) -> List[Tuple[float]]:
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
    cmap : list of tuple
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


def get_time(time: Optional[np.ndarray] = None,
             dt: Optional[float] = None,
             t_end: Optional[float] = None,
             n_vcg: Optional[int] = 1,
             len_vcg: Optional[List[int]] = None):
    """Returns variables for time, dt and t_end, depending on input.

    Parameters
    ----------
    time : np.ndarray, optional
        Time data for a given VCG, default=None
    dt : float, optional
        Interval between recording points for the VCG, default=None
    t_end : float, optional
        Total duration of the VCG recordings, default=None
    n_vcg : int, optional
        Number of VCGs being assessed, default=1
    len_vcg : int, optional
        Number of data points for each VCG being assessed, None

    Returns
    -------
    time : np.ndarray
        Time data for a given VCG
    dt : list of float
        Mean time interval for a given VCG recording
    t_end : list of float
        Total duration of each VCG recording

    Notes
    -----
    Time OR t_end/dt/len_vcg must be passed to this function
    """

    if time is None or time[0] is None:
        assert dt is not None, "Must pass either time or dt/t_end/len_vcg"
        assert t_end is not None, "Must pass either time or dt/t_end/len_vcg"
        assert len_vcg is not None, "Must pass either time or dt/t_end/len_vcg"
        if isinstance(dt, (int, float)):
            dt = [dt for _ in range(n_vcg)]
        if isinstance(t_end, (int, float)):
            t_end = [t_end for _ in range(n_vcg)]
        time = [np.arange(0, sim_t_end+sim_dt, sim_dt) for (sim_dt, sim_t_end) in zip(dt, t_end)]
        for sim_len_vcg, sim_time in zip(len_vcg, time):
            assert sim_len_vcg == len(sim_time), "vcg and time variables mis-aligned"
    else:
        if isinstance(time, np.ndarray):
            time = [time]
        assert len(time) == n_vcg, "vcg and time variables must be same length"
        for sim_time in time:
            assert max(np.diff(sim_time))-min(np.diff(sim_time)) < 0.0001,\
                "dt not constant for across provided time variable"
        dt = [np.mean(np.diff(sim_time)) for sim_time in time]
        t_end = [t[-1] for t in time]

    return time, dt, t_end


def convert_time_to_index(qrs_start: Optional[float] = None,
                          qrs_end: Optional[float] = None,
                          t_start: float = 0,
                          t_end: float = 200,
                          dt: float = 2) -> Tuple[int, int]:
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
    x_val = np.array(np.arange(t_start, t_end + dt, dt))
    i_qrs_start = np.where(x_val >= qrs_start)[0][0]
    i_qrs_end = np.where(x_val > qrs_end)[0][0]-1

    return i_qrs_start, i_qrs_end


def convert_index_to_time(idx: int,
                          t_start: float = 0,
                          t_end: float = 200,
                          dt: float = 2) -> float:
    """
    Return 'real' time for a given index

    Parameters
    ----------
    idx : int
        Index to convert
    t_start : float, optional
        Start time for overall data, default=0
    t_end : float, optional
        End time for overall data, default=200
    dt : float, optional
        Interval between time points, default=2

    Returns
    -------
    time : float
        The time value that corresponds to the given index
    """
    x_val = np.array(np.arange(t_start, t_end+dt, dt))
    return x_val[idx]

