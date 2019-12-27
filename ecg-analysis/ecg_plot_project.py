import matplotlib.pyplot as plt

import common_analysis as ca
import ecg_plot as ep


def plot_ecg_multilimits(ecg, qrs_limits, legend=None, dt=2, linewidth=3):
    """
    Plots a single ECG trace, with several different QRS limits

    Useful to show the effects of different settings to establish QRS limits, by showing how multiple different QRS
    limits compare to the actual ECG trace.

    qrs_limits must be presented in form e.g. [[qrs_start1, qrs_start2, ...], [qrs_end1, qrs_end2, ...]]
    """

    plot_sequence = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'LI', 'LII', 'LIII', 'aVR', 'aVL', 'aVF']
    fig, ax = ep.__init_axes(plot_sequence)
    ecg, legend = ep.__process_inputs(ecg, legend)

    time = [i * dt for i in range(len(ecg[0]['V1']))]
    ep.__plot_data(time, ecg[0], None, None, ax, plot_sequence, linewidth)

    colours = ca.get_plot_colours(n=len(qrs_limits[0]))
    import matplotlib.lines as mlines
    line_handles = None
    for qrs_limit in qrs_limits:
        line_handles = list()
        for i in range(len(qrs_limit)):
            line_handles.append(mlines.Line2D([], [], color=colours[i], label=legend[i]))
            # Add 'fudge factor' if multiple limits are meant to be plotted on top of each other, so that they're
            # slightly offset and thus visible
            if qrs_limit[i] in qrs_limit[0:i]:
                qrs_limit[i] += 0.1
        ep.__plot_limits(ax, qrs_limit, colours, plot_sequence)

    plt.rc('text', usetex=True)
    plt.legend(handles=line_handles, bbox_to_anchor=(1.04, 1.1), loc="center left")

    return fig, ax
