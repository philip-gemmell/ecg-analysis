import numpy as np
from typing import List, Tuple, Union, Optional, Any

import tools_plotting


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
             len_vcg: Optional[List[int]] = None) -> Tuple[List[np.ndarray], List[float], List[float]]:
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
    time : list of np.ndarray
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
        assert len(time) == n_vcg, "vcg ({}) and time ({}) variables must be same length".format(n_vcg, len(time))
        for sim_time in time:
            assert max(np.diff(sim_time))-min(np.diff(sim_time)) < 0.0001,\
                "dt not constant for across provided time variable"
        dt = [np.mean(np.diff(sim_time)) for sim_time in time]
        t_end = [t[-1] for t in time]

    return time, dt, t_end


def convert_time_to_index(time_point: float,
                          time: Union[List[float], np.ndarray, None] = None,
                          t_start: Optional[float] = None,
                          t_end: Optional[float] = None,
                          dt: Optional[float] = None) -> int:
    """Converts a given time point to the relevant index value

    Parameters
    ----------
    time_point : float
        Time point for which we wish to find the corresponding index. If set to -1, will return the final index
    time : float or np.ndarray, optional
        Time data from which we wish to extract the index. If set to None, the time will be constructed based on the
        assumed t_start, t_end and dt values
    t_start : float, optional
        Start point of time; only used if `time' variable not given, default=None
    t_end : float, optional
        End point of time; only used if `time' variable not given, default=None
    dt : float, optional
        Interval between time points; only used if time not given, default=None

    Returns
    -------
    i_time : int
        Index corresponding to the time point given

    Raises
    ------
    AssertionError
        If insufficient data are provided to the function to enable it to function
    """

    if time is None:
        assert t_start is not None, "t_start not provided"
        assert t_end is not None, "t_end not provided"
        assert dt is not None, "dt not provided"
        time = np.array(np.arange(t_start, t_end + dt, dt))

    if time_point == -1:
        return len(time)-1
    else:
        return np.where(time >= time_point)[0][0]


def convert_index_to_time(idx: int,
                          time: Optional[np.ndarray] = None,
                          t_start: float = 0,
                          t_end: float = 200,
                          dt: float = 2) -> float:
    """
    Return 'real' time for a given index

    Parameters
    ----------
    idx : int
        Index to convert
    time : np.ndarray, optional
        Time data; if not provided, will be assumed from t_start, t_end and dt variables, default=None
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

    if time is None:
        time = np.array(np.arange(t_start, t_end+dt, dt))
    return time[idx]


def convert_input_to_list(input_data: Any,
                          n_list: int = 1,
                          n_list2: int = -1,
                          list_depth: int = 1,
                          default_entry: Any = None) -> list:
    """Convert a given input to a list of inputs of required length. If already a list, will confirm that it's the
    right length.

    Parameters
    ----------
    input_data : Any
        Input argument to be checked
    n_list : int, optional
        Number of entries required in input; if set to -1, will not perform any checks beyond 'depth' of lists,
        default=1
    n_list2 : int, optional
        Number of entries for secondary input; if set to -1, will not perform any checks
    list_depth : int
        Number of nested lists required. If just a simple list of e.g. VCGs, then will be 1 ([vcg1, vcg2,...]). If a
        list of lists (e.g. [[qrs_start1, qrs_start2,...], [qrs_end1, qrs_end2,...]), then 2.
    default_entry : {'colour', 'line', None, Any}, optional
        Default entry to put into list. If set to None, will just repeat the input data to match n_list. However,
        if set to either 'colour' or 'line', will return the default potential settings, default=None

    Returns
    -------
    output : list
        Formatted output

    Notes
    -----
    If the data are already provided as a list and list_depth==1, function will simply check that the list is of the
    correct length. If list_depth==2, will check that deepest level of nesting has the correct length; if n_list2 is
    provided, it will check the top level of the list is of the correct length. This is used, for example,
    when several different limits are provided for several different VCGs, and a legend is needed. Thus, if there are n
    different VCGs to be plotted, and each has m different limits to be plotted, the legend can be checked to be of
    the form [[x11, x21,...,xn1], [x12, x22,...,xn2],...[x1m, x2m,...xnm]]

    If the data are not in list form, will:
        (a) if default_entry==None, will replicate input_data to match n_vcg, e.g. '-' becomes ['-', '-',...]
        (b) if default_entry=='colour', will return list of RBG values for colours
        (c) if default_entry=='line', will return list of line entries
        (d) for any other value of default_entry, will reproduce that value
    """

    if isinstance(input_data, list):
        output_data = input_data.copy()  # Prevent changes to original input data (which may affect future iterations!)
        if list_depth == 1:
            # Simplest option - just want a list of equal length to the variable of interest
            if n_list != -1:
                assert len(output_data) == n_list, "Incorrect number of entries in input_data"
        elif list_depth == 2:
            # More complicated - we require data to be passed in form [[x1a, x1b,...],[x2a,x2b,...],...],
            # where the length of [xna, xnb,...] is equal to the variable of interest. For example, for n ECG traces,
            # we wish to plot QRS start ([x1a, x1b,...]), QRS end ([x2a, x2b,...]) and so on
            if not isinstance(output_data[0], list):
                if n_list == 1:
                    # This is the instance where there is only a single variable of interest, i.e. we require the data
                    # to be reformatted from [x1a, x2a, x3a,...] to [[x1a],[x2a],[x3a],...]
                    output_data = [[output_data1] for output_data1 in output_data]
                else:
                    output_data = [output_data]

            if n_list != -1:
                for inner_data in output_data:
                    assert len(inner_data) == n_list,\
                        "inner_data of input incorrectly formatted (len(inner_data)=={})".format(len(inner_data))
            if n_list2 != -1:
                assert len(output_data) == n_list2, "Incompatible length for top level of list provided"
        else:
            raise Exception("Not coded for this eventuality...")
    else:
        if default_entry is None:
            output_data = [input_data for _ in range(n_list)]
        elif default_entry == 'colour':
            output_data = tools_plotting.get_plot_colours(n=n_list)
        elif default_entry == 'line':
            output_data = tools_plotting.get_plot_lines(n=n_list)
        else:
            output_data = [default_entry for _ in range(n_list)]

        # Adjust the nesting of these default lists depending on requirements
        list_embed = list_depth-1
        while list_embed > 0:
            output_data = [output_data]
            list_embed -= 1
    return output_data


def get_i_colour(axis_handle) -> int:
    """ Get index appropriate to colour value to plot on a figure (will be 0 if brand new figure) """
    if axis_handle is None:
        return 0
    else:
        if len(axis_handle.lines) == 0:
            return 0
        else:
            return len(axis_handle.lines)-1


def deprecated_convert_time_to_index(qrs_start: Optional[float] = None,
                                     qrs_end: Optional[float] = None,
                                     time: Optional[List[float]] = None,
                                     t_start: float = 0,
                                     t_end: float = 200,
                                     dt: float = 2) -> Tuple[int, int]:
    """Return indices of QRS start and end points. NB: indices returned match Matlab output

    Parameters
    ----------
    qrs_start : float or int, optional
        Start time to convert to index. If not given, will default to the same as the start time of the entire list
    qrs_end : float or int, optional
        End time to convert to index. If not given, will default to the same as the end time of the entire list
    time : float, optional
        Time data to be used to calculate index. If given, will over-ride the values used for dt/t_start/t_end.
        Default=None
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

    if time is None:
        time = np.array(np.arange(t_start, t_end + dt, dt))

    if qrs_start is None:
        i_qrs_start = 0
    else:
        i_qrs_start = np.where(time >= qrs_start)[0][0]

    if qrs_end is None:
        i_qrs_end = -1
    else:
        i_qrs_end = np.where(time >= qrs_end)[0][0]-1

    return i_qrs_start, i_qrs_end
