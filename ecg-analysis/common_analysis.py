import numpy as np  # type: ignore
import matplotlib.cm as cm  # type: ignore
from scipy import signal  # type: ignore
from typing import List, Tuple, Optional, Union, Any


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


def get_plot_lines(n: int = 4) -> Union[List[tuple], List[str]]:
    """Returns different line-styles for plotting

    Parameters
    ----------
    n : int, optional
        Number of different line-styles required

    Returns
    -------
    lines : list of str or list of tuple
        List of different line-styles
    """

    if n <= 4:
        return['-', '--', '-.', ':']
    elif n < 15:
        lines = list()
        dash_gap = 2
        i_lines = 0
        while i_lines < 5:
            # First few iterations to be '-----', '-.-.-.', '-..-..-..-',...
            lines.append((0, tuple([5, dash_gap]+[1, dash_gap]*i_lines)))
            i_lines += 1
        while i_lines < 10:
            # Following iterations to be '--.--', '--..--'. '--...---',...
            lines.append((0, tuple([5, dash_gap, 5, dash_gap, 1, dash_gap]+[1, dash_gap]*(i_lines-5))))
            i_lines += 1
        while i_lines < 15:
            # Following iterations to be '---.---', '---..---', '---...---',...
            lines.append((0, tuple([5, dash_gap, 5, dash_gap, 5, dash_gap, 1, dash_gap]+[1, dash_gap]*(i_lines-10))))
            i_lines += 1
        return lines
    else:
        raise Exception('Unsure of how effective this number of different linestyles will be...')


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


def convert_input_to_list(input_data: Any,
                          n_list: int = 1,
                          list_depth: int = 1,
                          default_entry: Optional[str] = None) -> list:
    """Convert a given input to a list of inputs of required length. If already a list, will confirm that it's the
    right length.

    Parameters
    ----------
    input_data : Any
        Input argument to be checked
    n_list : int, optional
        Number of entries required in input, default=1
    list_depth : int
        Number of nested lists required. If just a simple list of e.g. VCGs, then will be 1 ([vcg1, vcg2,...]). If a
        list of lists (e.g. [[qrs_start1, qrs_start2,...], [qrs_end1, qrs_end2,...]), then 2.
    default_entry : {'colour', 'line', None}, optional
        Default entry to put into list. If set to None, will just repeat the input data to match n_list. However,
        if set to either 'colour' or 'line', will return the default potential settings, default=None

    Returns
    -------
    output : list
        Formatted output
    """

    if isinstance(input_data, list):
        if list_depth == 1:
            # Simplest option - just want a list of equal length to the variable of interest
            assert len(input_data) == n_list, "Incorrect number of entries in input_data"
        elif list_depth == 2:
            # More complicated - we require data to be passed in form [[x1a, x1b,...],[x2a,x2b,...],...],
            # where the length of [xna, xnb,...] is equal to the variable of interest
            for i_input_data in range(len(input_data)):
                # This is the instance where there is only a single variable of interest, i.e. we require the data to
                # be reformatted from [x1a, x2a, x3a,...] to [[x1a],[x2a],[x3a],...]
                if not isinstance(input_data[i_input_data], list):
                    input_data[i_input_data] = [input_data[i_input_data]]
            for inner_data in input_data:
                assert len(inner_data) == n_list, "inner_data of input incorrectly formatted"
        else:
            raise Exception("Not coded for this eventuality...")
        return input_data
    else:
        if default_entry is None:
            return [input_data for _ in range(n_list)]
        elif default_entry == 'colour':
            return get_plot_colours(n=n_list)
        elif default_entry == 'line':
            return get_plot_lines(n=n_list)
        else:
            raise Exception("Not coded for this eventuality...")


def check_list_depth(input_list, depth_count=1, max_depth=0, n_args=0):
    """ Function to calculate the depth of nested loops

    TODO: Finish this damn code

    Parameters
    ----------
    input_list : list
        Input argument to check
    depth_count : int, optional
        Depth of nested loops thus far
    max_depth : int, optional
        Maximum expected depth of list, default=0 (not checked)
    n_args : int, optional
        Required length of 'base' list, default=0 (not checked)

    Returns
    -------
    depth_count : int
        Depth of nested loops

    Notes
    -----
    A list of form [a1, a2, a3, ...] has depth 1.
    A list of form [[a1, a2, a3, ...], [b1, b2, b3, ...], ...] has depth 2.
    And so forth...

    If n_args is set to an integer greater than 0, it will check that the lowest level of lists (for all entries)
    will be of the required length
        if depth=1 as above, len([a1, a2, a3, ...]) == n_args
        if depth=2 as above, len([a1, a2, a3, ...]) == n_args && len([b1, b2, b3, ...]) == n_args
    """

    for input_list_inner in input_list:
        if isinstance(input_list_inner, list):
            depth_count += 1

    if not isinstance(input_list[0], list):
        assert all([not isinstance(input_list_inner, list) for input_list_inner in input_list])
        if n_args > 0:
            for input_list_inner in input_list:
                assert len(input_list_inner) == n_args, "Incorrect list lengths"
    else:
        depth_count += 1
        if max_depth > 0:
            assert depth_count <= max_depth, "Maximum depth exceeded"
        for input_list_inner in input_list:
            check_list_depth(input_list_inner, depth_count=depth_count)
    return depth_count


def normalise_signal(data: np.ndarray) -> np.ndarray:
    """Returns a normalised signal, such that the maximum value in the signal is 1, or the minimum is -1

    Parameters
    ----------
    data : np.ndarray
        Signal to be normalised

    Returns
    -------
    normalised_data : np.ndarray
        Normalised signal
    """

    return np.divide(np.absolute(data), np.amax(np.absolute(data)))
