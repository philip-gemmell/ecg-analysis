import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from math import sin, cos, acos, atan
import math
import warnings

import common_analysis

# import matplotlib
# matplotlib.use('Agg')


def plot_vcg_single(vcg, legend=None):
    """ Plot the 3 spatial components of VCG. If multiple VCGs given, will plot VCGs on separate figures"""

    if isinstance(vcg, np.ndarray):
        vcg = [vcg]
    if legend is not None:
        assert len(legend) == len(vcg)

    fig = list()
    ax = list()
    i_sim = 0
    for sim_vcg in vcg:
        fig.append(plt.figure())
        ax.append(fig[i_sim].add_subplot(111))
        """ Check data is presented in the correct manner (i.e. printing VCG time-course for x,y,z components, 
            rather than x,y,z elements at each point in the time course) """
        if sim_vcg.shape[0] < sim_vcg.shape[1]:
            for sim_vcg_xyz in sim_vcg:
                ax[i_sim].plot(sim_vcg_xyz)
        else:
            for sim_vcg_xyz in sim_vcg.T:
                ax[i_sim].plot(sim_vcg_xyz)
        plt.legend(['X', 'Y', 'Z'])
        if legend is not None:
            ax[i_sim].set_title(legend[i_sim])
        i_sim += 1

    return fig, ax


def plot_vcg_multiple(vcg, legend=None, layout=None):
    """ Plot multiple instances of VCGs. To avoid too much cross-talk, plot x,y,z components on separate sub-figures """

    """ Layout options:
        figures     Each x,y,z plot is on a separate figure 
        row         x,y,z plots are arranged on a horizontal row in one figure
        column      x,y,z plots are arranged in a vertical column in one figure
        best        x,y,z plots are arranged to try and optimise space
        grid        x,y,z plots are arranged in a grid (like best, but more rigid grid) """

    if legend is not None:
        assert len(vcg) == len(legend)
        plt.rc('text', usetex=True)
    else:
        legend = [None for _ in vcg]
    if layout is None:
        layout = 'best'

    """ Create and assign figure handles, including a dummy variable for the figure handles for cross-compatability """
    if layout == 'figures':
        fig = [plt.figure() for _ in range(3)]
        fig_h = fig
    else:
        fig = plt.figure()
        fig_h = [fig for _ in range(3)]

    ax = dict()
    if layout == 'figures':
        ax['x'] = fig[0].add_subplot(1, 1, 1, ylabel='x')
        ax['y'] = fig[1].add_subplot(1, 1, 1, ylabel='y', sharex=ax['x'], sharey=ax['x'])
        ax['z'] = fig[2].add_subplot(1, 1, 1, ylabel='z', sharex=ax['x'], sharey=ax['x'])
    elif layout == 'row':
        gs = gridspec.GridSpec(3, 1)
        ax['x'] = fig_h[0].add_subplot(gs[0], ylabel='x')
        ax['y'] = fig_h[1].add_subplot(gs[1], ylabel='y', sharex=ax['x'], sharey=ax['x'])
        ax['z'] = fig_h[2].add_subplot(gs[2], ylabel='z', sharex=ax['x'], sharey=ax['x'])
        # ax['x'] = fig_h[0].add_subplot(1, 3, 1, ylabel='x')
        # ax['y'] = fig_h[1].add_subplot(1, 3, 2, ylabel='y', sharex=ax['x'], sharey=ax['x'])
        # ax['z'] = fig_h[2].add_subplot(1, 3, 3, ylabel='z', sharex=ax['x'], sharey=ax['x'])
        plt.setp(ax['y'].get_yticklabels(), visible=False)
        plt.setp(ax['z'].get_yticklabels(), visible=False)
        gs.update(wspace=0.025, hspace=0.05)
    elif layout == 'column':
        gs = gridspec.GridSpec(1, 3)
        ax['x'] = fig_h[0].add_subplot(gs[0], ylabel='x')
        ax['y'] = fig_h[1].add_subplot(gs[1], ylabel='y', sharex=ax['x'], sharey=ax['x'])
        ax['z'] = fig_h[2].add_subplot(gs[2], ylabel='z', sharex=ax['x'], sharey=ax['x'])
        # ax['x'] = fig_h[0].add_subplot(3, 1, 1, ylabel='x')
        # ax['y'] = fig_h[1].add_subplot(3, 1, 2, ylabel='y', sharex=ax['x'], sharey=ax['x'])
        # ax['z'] = fig_h[2].add_subplot(3, 1, 3, ylabel='z', sharex=ax['x'], sharey=ax['x'])
        plt.setp(ax['y'].get_xticklabels(), visible=False)
        plt.setp(ax['z'].get_xticklabels(), visible=False)
        gs.update(wspace=0.025, hspace=0.05)
    elif layout == 'best':
        gs = gridspec.GridSpec(2, 6)
        ax['x'] = fig_h[0].add_subplot(gs[0, :3], ylabel='x')
        ax['y'] = fig_h[1].add_subplot(gs[0, 3:], ylabel='y', sharex=ax['x'], sharey=ax['x'])
        ax['z'] = fig_h[2].add_subplot(gs[1, 2:4], ylabel='z', sharex=ax['x'], sharey=ax['x'])
        plt.setp(ax['y'].get_yticklabels(), visible=False)
        gs.update(wspace=0.025, hspace=0.05)
    elif layout == 'grid':
        gs = gridspec.GridSpec(2, 2)
        ax['x'] = fig_h[0].add_subplot(gs[0], ylabel='x')
        ax['y'] = fig_h[1].add_subplot(gs[1], ylabel='y', sharex=ax['x'], sharey=ax['x'])
        ax['z'] = fig_h[2].add_subplot(gs[2], ylabel='z', sharex=ax['x'], sharey=ax['x'])
        # ax['x'] = fig_h[0].add_subplot(2, 2, 1, ylabel='x')
        # ax['y'] = fig_h[1].add_subplot(2, 2, 2, ylabel='y', sharex=ax['x'], sharey=ax['x'])
        # ax['z'] = fig_h[2].add_subplot(2, 2, 3, ylabel='z', sharex=ax['x'], sharey=ax['x'])
        plt.setp(ax['x'].get_xticklabels(), visible=False)
        plt.setp(ax['y'].get_yticklabels(), visible=False)
        gs.update(wspace=0.025, hspace=0.05)

    """ Plot data and add legend if required """
    xyz_label = ['x', 'y', 'z']

    for (sim_vcg, sim_legend) in zip(vcg, legend):
        if sim_vcg.shape[0] < sim_vcg.shape[1]:
            for (sim_vcg_xyz, xyz) in zip(sim_vcg, xyz_label):
                ax[xyz].plot(sim_vcg_xyz, label=sim_legend)
        else:
            for (sim_vcg_xyz, xyz) in zip(sim_vcg.T, xyz_label):
                ax[xyz].plot(sim_vcg_xyz, label=sim_legend)

    if legend[0] is not None:
        if layout == 'figures':
            for xyz in xyz_label:
                ax[xyz].legend()
        else:
            ax['x'].legend()

    return fig, ax


def convert_ecg_to_vcg(ecg):
    """ Convert ECG data to vectorcardiogram (VCG) data using the Kors matrix method """

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


# def get_qrs_start_end(vcg, dt=2, velocity_offset=2, low_p=40, order=2, threshold_frac=0.2, plot_sv=False,
#                       legend=None, fig=None, t_end=200, matlab_match=False):
def get_qrs_start_end(vcg, dt=2, velocity_offset=2, low_p=40, order=2, threshold_frac=0.2, filter_sv=True, t_end=200,
                      matlab_match=False):
    """ Calculate the extent of the VCG QRS complex on the basis of max derivative """

    # vcg                       List of VCG data to get QRS start and end points for

    # dt                2ms
    # velocity_offset   2       Offset between values in VCG over which to calculate spatial velocity, i.e. 1 will
    #                           use neighbouring values to calculate the gradient/velocity
    # low_p             40      Low frequency for bandpass filter
    # order             2       Order for Butterworth filter
    # threshold_frac    0.2     Fraction of maximum spatial velocity to trigger QRS detection
    # plot_sv           False   Plot the calculated spatial velocity and the original VCG, both showing derived QRS
    #                           limits
    # legend            None    Legend entries for spatial velocity plot
    # fig               None    Fig of existing figure to plot results on (must have matching axes to figure produced
    #                           using this function)
    # t_end             200     End time of simulation
    # matlab_math       False   Apply fudge factor to match Matlab results

    if isinstance(vcg, np.ndarray):
        vcg = [vcg]

    """ Create indices to track (1) which colour to plot, and (2) which of the current set of VCGs is currently under
        consideration """
    i_vcg = 0
    x_val, sv, threshold = get_spatial_velocity(vcg=vcg, velocity_offset=velocity_offset, t_end=t_end, dt=dt,
                                                threshold_frac=threshold_frac, matlab_match=matlab_match,
                                                filter_sv=filter_sv, low_p=low_p, order=order)
    qrs_start = list()
    qrs_end = list()
    qrs_duration = list()
    for (sim_sv, sim_x, sim_threshold) in zip(sv, x_val, threshold):
        if matlab_match:
            i_qrs_start = np.where(sim_sv > sim_threshold)[0][0] + 2
        else:
            i_qrs_start = np.where(sim_sv > sim_threshold)[0][0]

        """ Find end of QRS complex where it reduces below threshold (searching backwards from end). Fudge factors 
            are added to ensure uniformity with Matlab results """
        # i_qrs_end = np.where(sv_filtered[i_qrs_start+1:] < threshold)[0][0]+(i_qrs_start+1)
        i_qrs_end = len(sim_sv) - (np.where(np.flip(sim_sv) > sim_threshold)[0][0] - 1)
        assert i_qrs_start < i_qrs_end
        assert i_qrs_end < len(sim_sv)

        qrs_start_temp = sim_x[i_qrs_start]
        qrs_end_temp = sim_x[i_qrs_end]

        qrs_start.append(qrs_start_temp)
        qrs_end.append(qrs_end_temp)
        qrs_duration.append(qrs_end_temp - qrs_start_temp)

        i_vcg += 1

    return qrs_start, qrs_end, qrs_duration


def get_spatial_velocity(vcg, velocity_offset=2, t_end=200, dt=2, threshold_frac=0.2, matlab_match=False,
                         filter_sv=True, low_p=40, order=2):
    """ Calculate spatial velocity """

    if isinstance(vcg, np.ndarray):
        vcg = [vcg]

    sv = list()
    x_val = list()
    threshold_full = list()
    for sim_vcg in vcg:
        """ Compute spatial velocity of VCG """
        dvcg = ((sim_vcg[velocity_offset:] - sim_vcg[:-velocity_offset]) / 2) * dt

        # Calculates Euclidean distance based on spatial velocity in x, y and z directions
        sim_sv = np.linalg.norm(dvcg, axis=1)

        """ Determine threshold for QRS complex, then find start of QRS complex. Iteratively remove more of the plot 
            if the 'start' is found to be 0 (implies it is still getting confused by the preceding wave). 
            Alternatively, just cut off the first 10ms of the beat (original Matlab method) """
        sample_freq = 1000/dt
        if matlab_match:
            sim_sv = sim_sv[5:]
            sim_x = list(range(velocity_offset, t_end, dt))[5:]
            if filter_sv:
                sim_sv = common_analysis.filter_egm(sim_sv, sample_freq, low_p, order)
        else:
            sim_x = list(range(velocity_offset, t_end, dt))
            threshold = max(sim_sv)*threshold_frac
            if filter_sv:
                sv_filtered = common_analysis.filter_egm(sim_sv, sample_freq, low_p, order)
            else:
                sv_filtered = sim_sv
            i_qrs_start = np.where(sv_filtered > threshold)[0][0]
            while i_qrs_start == 0:
                sim_sv = sim_sv[1:]
                sim_x = sim_x[1:]
                threshold = max(sim_sv) * threshold_frac

                if filter_sv:
                    sv_filtered = common_analysis.filter_egm(sim_sv, sample_freq, low_p, order)
                else:
                    sv_filtered = sim_sv
                i_qrs_start = np.where(sv_filtered > threshold)[0][0]
            sim_sv = sv_filtered
        sv.append(sim_sv)
        x_val.append(sim_x)
        threshold_full.append(threshold)

    return x_val, sv, threshold_full


def plot_spatial_velocity(vcg, sv=None, qrs_limits=None, fig=None, legend=None, t_end=200, dt=2, filter_sv=True):
    """ Plot the spatial velocity and VCG elements, with limits (e.g. QRS limits) if provided. Note that if spatial
        velocity is not provided, default values will be used to calculate it - if anything else is desired,
        then spatial velocity must be calculated first and provided to the function.
    """

    fig, ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z, colours = __plot_spatial_velocity_prep_axes(vcg, fig)
    vcg, legend = __plot_spatial_velocity_preprocess_inputs(vcg, legend)
    x_val, sv = __plot_spatial_velocity_get_plot_data(sv, vcg, t_end, dt, filter_sv)

    """ Plot spatial velocity and VCG components"""
    i_colour = get_i_colour(ax_sv)
    x_vcg_data = list(range(0, t_end + dt, dt))
    for (sim_x, sim_vcg, sim_sv, sim_label) in zip(x_val, vcg, sv, legend):
        __plot_spatial_velocity_plot_data(sim_x, sim_sv, x_vcg_data, sim_vcg, sim_label, colours[i_colour],
                                          ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z)
        i_colour += 1

    """ Plot QRS limits, if provided """
    if qrs_limits is not None:
        # Cycle through each limit provided, e.g. QRS start, QRS end...
        for qrs_limit in qrs_limits:
            i_colour = get_i_colour(ax_sv)-len(vcg)
            assert len(qrs_limit) == len(vcg)

            # Plot limits for each given VCG
            for sim_qrs_limit in qrs_limit:
                __plot_spatial_velocity_plot_limits(sim_qrs_limit, ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z,
                                                    colours[i_colour])
                i_colour += 1

    """ Add legend (or legend for limits, if appropriate """
    labels = [line.get_label() for line in ax_sv.get_lines()]
    plt.rc('text', usetex=True)
    ax_sv.legend(labels)

    return fig


def plot_spatial_velocity_multilimit(vcg, sv=None, qrs_limits=None, fig=None, legend=None, t_end=200, dt=2,
                                     filter_sv=True):
    """ Plot a single instance of a spatial velocity curve, but with multiple limits for QRS """

    """ Confirm VCG and limit data are correctly formatted """
    assert isinstance(vcg, np.ndarray)
    for qrs_limit in qrs_limits:
        assert len(qrs_limit) == len(qrs_limits[0])

    fig, ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z, colours = __plot_spatial_velocity_prep_axes(vcg, fig)
    vcg, legend = __plot_spatial_velocity_preprocess_inputs(vcg, legend)
    x_val, sv = __plot_spatial_velocity_get_plot_data(sv, vcg, t_end, dt, filter_sv)

    """ Plot spatial velocity and VCG components"""
    x_vcg_data = list(range(0, t_end + dt, dt))
    __plot_spatial_velocity_plot_data(x_val[0], sv[0], x_vcg_data, vcg[0], None, None,
                                      ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z)

    """ Plot QRS limits, along with proxy patches for the legend. Adjust values for QRS limits to prevent overlap. """
    colours = common_analysis.get_plot_colours(n=len(qrs_limits[0]))
    import matplotlib.lines as mlines
    for qrs_limit in qrs_limits:
        line_handles = list()
        for i in range(len(qrs_limit)):
            # blue_line = mlines.Line2D([], [], color=colours[i], label=legend_limits[i])
            line_handles.append(mlines.Line2D([], [], color=colours[i], label=legend[i]))
            if i > 0:
                if qrs_limit[i] <= qrs_limit[i-1]:
                    qrs_limit[i] += 0.1
            __plot_spatial_velocity_plot_limits(qrs_limit[i], ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z, colours[i])
    # ax_sv.legend(handles=[blue_line])
    ax_sv.legend(handles=line_handles)
    return None


def __plot_spatial_velocity_preprocess_inputs(vcg, legend):
    """ Preprocess other inputs """
    if isinstance(vcg, np.ndarray):
        vcg = [vcg]
    if isinstance(legend, str):
        legend = [legend]
    elif legend is None:
        legend = [str(i) for i in range(len(vcg))]
    return vcg, legend


def __plot_spatial_velocity_prep_axes(vcg, fig):
    """ Prepare figure and axes """
    if fig is None:
        fig = plt.figure()
        gs = gridspec.GridSpec(3, 3)
        ax_sv = fig.add_subplot(gs[:, :-1])
        ax_vcg_x = fig.add_subplot(gs[0, -1])
        ax_vcg_y = fig.add_subplot(gs[1, -1])
        ax_vcg_z = fig.add_subplot(gs[2, -1])
        plt.setp(ax_vcg_x.get_xticklabels(), visible=False)
        plt.setp(ax_vcg_y.get_xticklabels(), visible=False)
        gs.update(hspace=0.05)
        colours = common_analysis.get_plot_colours(len(vcg))
    else:
        ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z = fig.get_axes()
        colours = common_analysis.get_plot_colours(len(ax_sv.lines) + len(vcg))
        """ If too many lines already exist on the plot, need to recolour them all to prevent cross-talk """
        if len(ax_sv.lines) + len(vcg) > 10:
            for ax in ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z:
                lines = ax.get_lines()
                i_vcg = 0
                for line in lines:
                    line.set_color(colours[i_vcg])
                    i_vcg += 1

    """ Add labels to axes """
    ax_sv.set_xlabel('Time (ms)')
    ax_sv.set_ylabel('Spatial velocity')
    ax_vcg_x.set_ylabel('VCG (x)')
    ax_vcg_y.set_ylabel('VCG (y)')
    ax_vcg_z.set_ylabel('VCG (z)')
    ax_vcg_z.set_xlabel('Time (ms)')

    return fig, ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z, colours


def __plot_spatial_velocity_get_plot_data(sv, vcg, t_end, dt, filter_sv):
    """ Prepare spatial velocity """
    if sv is None:
        x_val, sv, _ = get_spatial_velocity(vcg=vcg, t_end=t_end, dt=dt, filter_sv=filter_sv)
    else:
        x_val = list()
        for sim_sv in sv:
            x_val.append([(i * dt) + 2 for i in range(len(sim_sv))])
    return x_val, sv


def __plot_spatial_velocity_plot_data(x_sv_data, sv_data, x_vcg_data, vcg_data, data_label, plot_colour,
                                      ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z):
    ax_vcg_x.plot(x_vcg_data, vcg_data[:, 0], color=plot_colour)
    ax_vcg_y.plot(x_vcg_data, vcg_data[:, 1], color=plot_colour)
    ax_vcg_z.plot(x_vcg_data, vcg_data[:, 2], color=plot_colour)
    ax_sv.plot(x_sv_data, sv_data, color=plot_colour, label=data_label)
    return None


def __plot_spatial_velocity_plot_limits(qrs_limit, ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z, limit_colour):
    for ax in [ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z]:
        ax.axvspan(qrs_limit, qrs_limit+0.1, color=limit_colour, alpha=0.5)
    return None


def get_i_colour(axis_handle):
    """ Get index appropriate to colour value to plot on a figure (will be 0 if brand new figure) """
    if axis_handle is None:
        return 0
    else:
        return len(axis_handle.lines)-1


def get_qrs_area(vcg, qrs_start=None, qrs_end=None, dt=2, t_end=200, matlab_match=False):
    """ Calculate area under QRS complex on VCG. """

    if isinstance(vcg, np.ndarray):
        vcg = [vcg]

    if qrs_start is None:
        qrs_start, _, _ = get_qrs_start_end(vcg)
    else:
        if len(qrs_start) != len(vcg):
            qrs_start = [qrs_start]
    if qrs_end is None:
        _, qrs_end, _ = get_qrs_start_end(vcg)
    elif qrs_end == -1:
        qrs_end = [t_end for _ in vcg]
    else:
        if len(qrs_end) != len(vcg):
            qrs_end = [qrs_end]

    qrs_area = list()
    qrs_area_components = list()
    for sim_vcg, sim_qrs_start, sim_qrs_end in zip(vcg, qrs_start, qrs_end):
        """ Recalculate indices for start and end points of QRS, and extract relevant data """
        i_qrs_start, i_qrs_end = common_analysis.convert_time_to_index(sim_qrs_start, sim_qrs_end, t_end=t_end, dt=dt)
        if matlab_match:
            sim_vcg_qrs = sim_vcg[i_qrs_start - 1:i_qrs_end + 1]
        else:
            sim_vcg_qrs = sim_vcg[i_qrs_start:i_qrs_end + 1]

        """ Calculate area under x,y,z curves by trapezium rule, then combine """
        qrs_area_temp = np.trapz(sim_vcg_qrs, dx=dt, axis=0)
        qrs_area_components.append(qrs_area_temp)
        qrs_area.append(np.linalg.norm(qrs_area_temp))

    return qrs_area, qrs_area_components


def get_azimuth_elevation(vcg, t_start=None, t_end=None):
    """ Calculate azimuth and elevation angles for a specified section of the VCG. """

    if isinstance(vcg, np.ndarray):
        vcg = [vcg]
    assert len(vcg) == len(t_start)
    assert len(vcg) == len(t_end)

    azimuth = list()
    elevation = list()
    for (sim_vcg, sim_t_start, sim_t_end) in zip(vcg, t_start, t_end):
        i_start, i_end = common_analysis.convert_time_to_index(sim_t_start, sim_t_end)
        sim_vcg = sim_vcg[i_start:i_end + 1]
        dipole_magnitude = np.linalg.norm(sim_vcg, axis=1)

        phi = [acos(sim_vcg_t[1] / dipole_magnitude_t) for (sim_vcg_t, dipole_magnitude_t) in
               zip(sim_vcg, dipole_magnitude)]
        theta = [atan(sim_vcg_t[2] / sim_vcg_t[0]) for (sim_vcg_t, dipole_magnitude_t) in
                 zip(sim_vcg, dipole_magnitude)]

        azimuth.append(theta)
        elevation.append(phi)

    return azimuth, elevation


def get_weighted_dipole_angles(vcg, t_start=None, t_end=None, matlab_match=False):
    """ Calculates metrics relating to the angles of the weighted dipole of the VCG. Usually used with QRS limits. """

    if isinstance(vcg, np.ndarray):
        vcg = [vcg]
    assert len(vcg) == len(t_start)
    assert len(vcg) == len(t_end)

    weighted_average_azimuth = list()
    weighted_average_elev = list()
    unit_weighted_dipole = list()
    for (sim_vcg, sim_t_start, sim_t_end) in zip(vcg, t_start, t_end):
        """ Calculate dipole at all points """
        i_start, i_end = common_analysis.convert_time_to_index(sim_t_start, sim_t_end)
        if matlab_match:
            sim_vcg = sim_vcg[i_start - 1:i_end]
        else:
            sim_vcg = sim_vcg[i_start:i_end + 1]
        dipole_magnitude = np.linalg.norm(sim_vcg, axis=1)

        # Weighted Elevation
        phi_weighted = [acos(sim_vcg_t[1] / dipole_magnitude_t) * dipole_magnitude_t for (sim_vcg_t, dipole_magnitude_t)
                        in
                        zip(sim_vcg, dipole_magnitude)]
        # Weighted Azimuth
        theta_weighted = [atan(sim_vcg_t[2] / sim_vcg_t[0]) * dipole_magnitude_t for (sim_vcg_t, dipole_magnitude_t) in
                          zip(sim_vcg, dipole_magnitude)]

        wae = sum(phi_weighted) / sum(dipole_magnitude)
        waa = sum(theta_weighted) / sum(dipole_magnitude)

        weighted_average_elev.append(wae)
        weighted_average_azimuth.append(waa)
        unit_weighted_dipole.append([sin(wae) * cos(waa), cos(wae), sin(wae) * sin(waa)])

    return weighted_average_azimuth, weighted_average_elev, unit_weighted_dipole


def get_weighted_dipole_magnitudes(vcg, t_start=None, t_end=None, matlab_match=False):
    """ Calculates metrics relating to the magnitude of the weighted dipole of the VCG: mean weighted dipole
    magnitude, maximum dipole magnitude and x,y.z components of the maximum dipole """

    if isinstance(vcg, np.ndarray):
        vcg = [vcg]
    assert len(vcg) == len(t_start)
    assert len(vcg) == len(t_end)

    weighted_magnitude = list()
    max_dipole_magnitude = list()
    max_dipole_components = list()
    max_dipole_time = list()
    for (sim_vcg, sim_t_start, sim_t_end) in zip(vcg, t_start, t_end):
        """ Calculate dipole at all points """
        i_start, i_end = common_analysis.convert_time_to_index(sim_t_start, sim_t_end)
        if matlab_match:
            sim_vcg_qrs = sim_vcg[i_start - 1:i_end]
        else:
            sim_vcg_qrs = sim_vcg[i_start:i_end + 1]
        dipole_magnitude = np.linalg.norm(sim_vcg_qrs, axis=1)

        weighted_magnitude.append(sum(dipole_magnitude) / len(sim_vcg_qrs))
        max_dipole_magnitude.append(max(dipole_magnitude))
        i_max = np.where(dipole_magnitude == max(dipole_magnitude))
        max_dipole_components.append(sim_vcg[i_max])
        max_dipole_time.append(common_analysis.convert_index_to_time(i_max, sim_t_start, sim_t_end))

    return weighted_magnitude, max_dipole_magnitude, max_dipole_components, max_dipole_time


def calculate_delta_dipole_angle(azimuth1, elevation1, azimuth2, elevation2, convert_to_degrees=False):
    """ Calculates the angular difference between two VCGs based on difference in azimuthal and elevation angles.
        Useful for calculating difference between weighted averages. """

    dt = list()
    for az1, ele1, az2, ele2 in zip(azimuth1, elevation1, azimuth2, elevation2):
        dot_product = (sin(ele1) * cos(az1) * sin(ele2) * cos(az2)) + \
                      (cos(ele1) * cos(ele2)) + \
                      (sin(ele1) * sin(az1) * sin(ele2) * sin(az2))
        if abs(dot_product) > 1:
            warnings.warn("abs(dot_product) > 1: dot_product = {}".format(dot_product))
            if dot_product > 1:
                dot_product = 1
            else:
                dot_product = -1

        dt.append(acos(dot_product))

    if convert_to_degrees:
        return [dt_i * 180 / math.pi for dt_i in dt]
    else:
        return dt


def compare_dipole_angles(vcg1, vcg2, t_start1=0, t_end1=None, t_start2=0, t_end2=None, n_compare=10,
                          convert_to_degrees=False, matlab_match=False):
    """ Calculates the angular differences between two VCGs are multiple points during their evolution """

    """ Calculate indices for the two VCG traces that correspond to the time points to be compared"""
    i_start1, i_end1 = common_analysis.convert_time_to_index(t_start1, t_end1)
    i_start2, i_end2 = common_analysis.convert_time_to_index(t_start2, t_end2)

    if matlab_match:
        i_start1 -= 1
        i_end1 -= 1
        i_start2 -= 1
        i_end2 -= 1
        idx_list1 = [int(round(i_start1 + i * (i_end1 - i_start1) / 10)) for i in range(1, n_compare + 1)]
        idx_list2 = [int(round(i_start2 + i * (i_end2 - i_start2) / 10)) for i in range(1, n_compare + 1)]
    else:
        idx_list1 = [int(round(i)) for i in np.linspace(start=i_start1, stop=i_end1, num=n_compare)]
        idx_list2 = [int(round(i)) for i in np.linspace(start=i_start2, stop=i_end2, num=n_compare)]

    """ Calculate the dot product and magnitudes of vectors. If the fraction of the two is slightly greater than 1 or
        less than -1, give a warning and correct accordingly. """
    cosdt = [np.dot(vcg1[i1], vcg2[i2]) / (np.linalg.norm(vcg1[i1]) * np.linalg.norm(vcg2[i2])) for i1, i2 in
             zip(idx_list1, idx_list2)]
    greater_less_warning = [True if ((cosdt_i < -1) or (cosdt_i > 1)) else False for cosdt_i in cosdt]
    if any(greater_less_warning):
        warnings.warn("Values found beyond bounds.")
        for i in range(len(greater_less_warning)):
            if greater_less_warning[i]:
                print("cosdt[{}] = {}".format(i, cosdt[i]))
                if cosdt[i] < -1:
                    cosdt[i] = -1
                elif cosdt[i] > 1:
                    cosdt[i] = 1

    dt = [acos(cosdt_i) for cosdt_i in cosdt]

    if convert_to_degrees:
        return [dt_i * 180 / math.pi for dt_i in dt]
    else:
        return dt
