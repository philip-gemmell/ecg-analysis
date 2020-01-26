import matplotlib.pyplot as plt
from typing import Union, Optional, List, Tuple

import common_analysis as ca


def plot(ecg: Union[List[dict], dict], dt: Union[int, float] = 2, legend: Optional[List[str]] = None,
         linewidth: float = 3, qrs_limits: Optional[list, float] = None, plot_sequence: Optional[List[str]] = None,
         single_fig: bool = True, colours: Optional[list] = None, linestyles: Optional[List[str]] = None,
         fig: Optional = None, ax: Optional = None) -> tuple:
    """
    Plot and label the ECG data from simulation(s). Optional to add in QRS start/end boundaries for plotting

    Parameters
    ----------
    ecg : dict or list
        Dictionary or list of dictionaries for ECG data, with dictionary keys corresponding to the trace name
    dt : int or float, optional
        Time interval at which data is recorded, default=2
    legend : list of str, optional
        List of names for each given set of ECG data e.g. ['BCL=300ms', 'BCL=600ms']
    linewidth : float, optional
        Width to use for plotting lines, default=3
    qrs_limits : float or list of float, optional
        Optional temporal limits (e.g. QRS limits) to add to ECG plots. Can add multiple limits, which will be
        plotted identically on all axes
    plot_sequence : list of str, optional
        Sequence in which to plot the ECG traces. Will default to: V1, V2, V3, V4, V5, V6, LI, LII, LIII, aVR, aVL, aVF
    single_fig : bool, optional
        If true, will plot all axes on a single figure window. If false, will plot each axis on a separate figure
        window. Default is True
    colours : list, optional
        Colours to be used to plot ECG traces. Will default to common_analysis.get_plot_colours()
    linestyles : str or list, optional
        Linestyles to be used to plot ECG traces. Will default to '-'
    fig : optional
        If given, will plot data on existing figure window
    ax: optional
        If given, will plot data using existing axis handles

    Returns
    -------
    fig
        Handle to output figure window, or dictionary to several handles if traces are all plotted in separate figure
        windows (if single_fig=False)
    ax : dict
        Dictionary to axis handles for ECG traces

    Raises
    ------
    AssertionError
        Checks that various list lengths are the same
    """

    # Prepare axes and inputs
    if plot_sequence is None:
        plot_sequence = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'LI', 'LII', 'LIII', 'aVR', 'aVL', 'aVF']
    if fig is None and ax is None:
        fig, ax = __init_axes(plot_sequence, single_fig)
    else:
        assert ((fig is not None) and (ax is not None)), 'Fig and ax handles must be passed simultaneously'
    ecg, legend = __process_inputs(ecg, legend)

    if colours is None:
        colours = ca.get_plot_colours(len(ecg))
    elif isinstance(colours, list):
        assert len(colours) == len(ecg), "Length of colour plotting list must be same as length of ECG data"
    else:
        colours = [colours for _ in ecg]

    if linestyles is None:
        linestyles = ['-' for _ in ecg]
    elif isinstance(linestyles, list):
        assert len(linestyles) == len(ecg), "Length of linestyle list must be same as length of ECG data"
    else:
        linestyles = [linestyles for _ in ecg]

    # Plot data
    time = [i*dt for i in range(len(ecg[0]['V1']))]
    for (sim_ecg, sim_label, sim_colour, sim_linestyle) in zip(ecg, legend, colours, linestyles):
        for key in plot_sequence:
            ax[key].plot(time, sim_ecg[key], linewidth=linewidth, label=sim_label, color=sim_colour,
                         linestyle=sim_linestyle)

    # Add QRS limits, if supplied
    if qrs_limits is not None:
        # Cycle through each limit provided, e.g. QRS start, QRS end...
        for qrs_limit in qrs_limits:
            __plot_limits(ax, qrs_limit, colours, linestyles)

    # Add legend, title and axis labels
    if legend[0] is not None:
        plt.rc('text', usetex=True)
        plt.legend(bbox_to_anchor=(1.04, 1.1), loc="center left")

    return fig, ax


def __process_inputs(ecg: Union[List[dict], dict], legend: Union[None, List[str]]) -> Tuple[List[dict], List[str]]:
    """
    Process input arguments

    Ensure ECG data is presented as list of traces (so subsequent data can always expect the same data input, i.e. a
    list of traces to plot rather than just a single trace). Also processes the legend argument, to ensure that it is a
    list of equal length

    Parameters
    ----------
    ecg : list of dict or dict
        ECG data for plotting. Presented as either dict of ECG outputs, or as a list of similar dictionaries
    legend : None or list
        Legend to be used for plotting. If none, no legend entries will be plotted. If a list, it must be of the same
        length as the list provided for ecg

    Returns
    -------
    ecg : list
        Correctly formatted list of ECG data
    legend : list
        Correctly formatted list of legend entries

    Raises
    ------
    AssertionError
        Check that list lengths are the same
    """

    if not isinstance(ecg, list):
        ecg = [ecg]
    if legend is None:
        legend = [None for _ in range(len(ecg))]
    else:
        assert len(legend) == len(ecg), "Length of legend entries must be same as lenght of ECG entries"
    return ecg, legend


def __init_axes(plot_sequence: List[str], single_fig: bool = True):
    """
    Initialise figure and axis handles

    Based on the required plot_sequence (order in which to plot ECG leads), and whether or not it is required to have
    all the plots on a single figure or on separate figures, will return the required figure and axis handles

    Parameters
    ----------
    plot_sequence : list of str
        Sequence in which to plot the ECG leads (only really important if plotted on single figure rather than
        separate figures)
    single_fig : bool, optional
        Whether or not to plot all ECG data on a single figure, or whether to plot each lead data in a separate figure

    Returns
    -------
    fig
        Handle to figure window(s)
    ax
        Handle to axes
    """

    if single_fig:
        fig = plt.figure()
        i = 1
        ax = dict()
        for key in plot_sequence:
            ax[key] = fig.add_subplot(2, 6, i)
            ax[key].set_title(key)
            i += 1
    else:
        fig = dict()
        ax = dict()
        for key in plot_sequence:
            fig[key] = plt.figure()
            ax[key] = fig[key].add_subplot(1, 1, 1)
            ax[key].set_title(key)
    return fig, ax


def __plot_limits(ax, limits: Union[List[float], float], colours: List[List[float]], linestyles: List[str]) -> None:
    """
    Add limit markers to a given plot (e.g. add line marking start of QRS complex)

    Parameters
    ----------
    ax
        Handle to axis
    limits: list of float or float
        Limits to plot on the axis
    colours : list of list of float
        RGB values of colours for the individual limits to plot. For plotting n limits, then should be given as
        [[R1, G1, B1], [R2, G2, B2], ... [Rn, Gn, Bn]]
    linestyles : list of str
        Linestyles to plot for each limit.
    """

    if not isinstance(limits, list):
        limits = [limits]
    assert len(limits) == len(colours), "Incompatible length of limits to plot and colours"
    assert len(limits) == len(linestyles), "Incompatible length of limits to plot and linestyles"
    for (sim_limit, sim_colour, sim_linestyle) in zip(limits, colours, linestyles):
        for key in ax:
            ax[key].axvline(sim_limit, color=sim_colour, alpha=0.5, linestyle=sim_linestyle)

    return None
