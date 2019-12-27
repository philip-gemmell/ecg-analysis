import matplotlib.pyplot as plt

import common_analysis as ca


def plot(ecg, dt=2, legend=None, linewidth=3, qrs_limits=None, plot_sequence=None, single_fig=True, colours=None,
         linestyles=None, fig=None, ax=None):
    """
    Plot and label the ECG data from simulation(s). Optional to add in QRS start/end boundaries for plotting

    Input parameters (required):
    ----------------------------

    ecg     Dictionary or list of dictionaries for ECG data, with dictionary keys corresponding to the trace name

    Input parameters (optional):
    ----------------------------

    dt              2       Time interval at which data is recorded
    legend          None    List of names for each given set of ECG data e.g. ['BCL=300ms', 'BCL=600ms']
    linewidth       3       Width to use for plotting lines
    qrs_limits      None    Optional temporal limits (e.g. QRS limits) to add to ECG plots. Can add multiple limits,
                            which will be plotted identically on all axes
    plot_sequence   None    Sequence in which to plot the ECG traces.
                            Will default to: V1, V2, V3, V4, V5, V6, LI, LII, LIII, aVR, aVL, aVF
    single_fig      True    Boolean: if true, will plot all axes on a single figure window. If false, will plot each
                            axis on a separate figure window
    colours         None    Colours to be used to plot ECG traces. Will default to common_analysis.get_plot_colours()
    linestyles      None    Linestyles to be used to plot ECG traces. Will default to '-'
    fig             None    If given, will plot data on existing figure window
    ax              None    If given, will plot data using existing axis handles

    Output parameters:
    ------------------

    fig     Handle to output figure window, or dictionary to several handles if traces are all plotted in separate
            figure windows (if single_fig=False)
    ax      Dictionary to axis handles for ECG traces

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
        assert len(colours) == len(ecg)
    else:
        colours = [colours for _ in ecg]

    if linestyles is None:
        linestyles = ['-' for _ in ecg]
    elif isinstance(linestyles, list):
        assert len(linestyles) == len(ecg)
    else:
        linestyles = [linestyles for _ in ecg]

    # Plot data
    time = [i*dt for i in range(len(ecg[0]['V1']))]
    for (sim_ecg, sim_label, sim_colour, sim_linestyle) in zip(ecg, legend, colours, linestyles):
        __plot_data(time, sim_ecg, sim_label, sim_colour, ax, plot_sequence, linewidth, sim_linestyle)

    # Add QRS limits, if supplied
    if qrs_limits is not None:
        # Cycle through each limit provided, e.g. QRS start, QRS end...
        for qrs_limit in qrs_limits:
            __plot_limits(ax, qrs_limit, colours, plot_sequence, linestyles)

    # Add legend, title and axis labels
    if legend[0] is not None:
        plt.rc('text', usetex=True)
        plt.legend(bbox_to_anchor=(1.04, 1.1), loc="center left")

    return fig, ax


def __process_inputs(ecg, legend):
    """
    Process input arguments

    Ensure ECG data is presented as list of traces (so subsequent data can always expect the same data input, i.e. a
    list of traces to plot rather than just a single trace). Also processes the legend argument, to ensure that it is a
    list of equal length

    """

    if not isinstance(ecg, list):
        ecg = [ecg]
    if legend is None:
        legend = [None for _ in range(len(ecg))]
    else:
        assert len(legend) == len(ecg)
    return ecg, legend


def __init_axes(plot_sequence, single_fig=True):
    """ Initialise figure and axis handles """

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


def __plot_data(time_val, ecg_data, label, colour, ax, plot_sequence, linewidth, linestyle):
    """ Plot ECG data for each trace on the appropriate axis """

    for key in plot_sequence:
        ax[key].plot(time_val, ecg_data[key], linewidth=linewidth, label=label, color=colour, linestyle=linestyle)
    return None


def __plot_limits(ax, limits, colours, plot_sequence, linestyles):
    """ Add limit markers to a given plot (e.g. add line marking start of QRS complex) """

    if not isinstance(limits, list):
        limits = [limits]
    for (sim_limit, sim_colour, sim_linestyle) in zip(limits, colours, linestyles):
        for key in plot_sequence:
            ax[key].axvline(sim_limit, color=sim_colour, alpha=0.5, linestyle=sim_linestyle)

    return None
