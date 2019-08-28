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


def plot_xy_vcg(vcg_x, vcg_y, xlabel='VCG (x)', ylabel='VCG (y)', linestyle='-', fig=None):
    """ Plot x vs y (or y vs z, or other combination) for VCG trace, with line colour shifting to show time
        progression. """

    from matplotlib.collections import LineCollection

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = fig.gca()

    """ Prepare line segments for plotting """
    t = np.linspace(0, 1, vcg_x.shape[0])  # "time" variable
    points = np.array([vcg_x, vcg_y]).transpose().reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=plt.get_cmap('viridis'), linestyle=linestyle)
    lc.set_array(t)

    ax.add_collection(lc)  # add the collection to the plot
    # line collections don't auto-scale the plot - set it up for a square plot
    __set_axis_limits([vcg_x, vcg_y], ax)
    # ax_min = min([vcg_x.min(), vcg_y.min()])
    # ax_max = max([vcg_x.max(), vcg_y.max()])
    # if abs(ax_min) > abs(ax_max):
    #     ax_max = -ax_min
    # else:
    #     ax_min = -ax_max
    # ax.set_xlim(ax_min, ax_max)
    # ax.set_ylim(ax_min, ax_max)
    # ax.set_aspect('equal', adjustable='box')

    """ Change the positioning of the axes """
    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Move the labels to the edges of the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation='horizontal')
    ax.xaxis.set_label_coords(1.05, 0.5)
    ax.yaxis.set_label_coords(0.5, 1.02)

    return fig


def plot_xyz_vcg(vcg_x, vcg_y, vcg_z, linestyle='-', fig=None):
    """ Plot the evolution of VCG in 3D space """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    """ Prepare line segments for plotting """
    t = np.linspace(0, 1, vcg_x.shape[0])  # "time" variable
    points = np.array([vcg_x, vcg_y, vcg_z]).transpose().reshape(-1, 1, 3)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segs, cmap=plt.get_cmap('viridis'), linestyle=linestyle)
    lc.set_array(t)

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.gca()
    ax.add_collection3d(lc)  # add the collection to the plot

    """ Set axis limits (not automatic for line collections) """
    __set_axis_limits([vcg_x, vcg_y, vcg_z], ax)

    ax.set_xlabel('VCG (x)')
    ax.set_ylabel('VCG (y)')
    ax.set_zlabel('VCG (z)')

    add_unit_sphere(ax)

    return fig


def plot_xyz_vector(vector=None, x=None, y=None, z=None, fig=None, linecolour='C0', linestyle='-'):
    """ Plots a specific vector in 3D space (e.g. to reflect maximum dipole) """
    # draw a vector
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    if (vector is None) == (x is None):
        raise ValueError("Exactly one of vertices and x,y,z must be given")
    if vector is not None:
        x = vector[0]
        y = vector[1]
        z = vector[2]

    class Arrow3D(FancyArrowPatch):

        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.gca()

    """ Adapt linestyle variable if required
    (see http://matplotlib.1069221.n5.nabble.com/linestyle-option-for-FancyArrowPatch-and-similar-commands-td39913.html)
    """
    if linestyle == '--':
        linestyle = 'dashed'
    elif linestyle == ':':
        linestyle = 'dotted'
    elif linestyle == '-.':
        linestyle = 'dashdot'
    elif linestyle == '-':
        linestyle = 'solid'
    else:
        warnings.warn('Unrecognised value for linestyle variable...')

    a = Arrow3D([0, x], [0, y], [0, z], mutation_scale=20, lw=1, arrowstyle="-|>", color=linecolour,
                linestyle=linestyle)
    ax.add_artist(a)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('VCG (x)')
    ax.set_ylabel('VCG (y)')
    ax.set_zlabel('VCG (z)')

    return fig


def __set_axis_limits(data, ax):
    """ Set axis limits (not automatic for line collections) """
    ax_min = min([i.min() for i in data])
    ax_max = max([i.max() for i in data])
    if abs(ax_min) > abs(ax_max):
        ax_max = -ax_min
    else:
        ax_min = -ax_max
    if ax_max < 1:
        ax_min = -1
        ax_max = 1
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    if len(data) == 3:
        ax.set_zlim(ax_min, ax_max)
    ax.set_aspect('equal', adjustable='box')
    return None


def add_unit_sphere(ax):
    """ Add a unit sphere to a 3D plot"""
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="k", linewidth=0.5, alpha=0.25)
    return None


def add_xyz_axes(ax, uniform_axes=False, square_axes=False, unit_axes=False):
    """ Plot dummy axes (can't move splines in 3D plots) """

    """ Construct dummy 3D axes - make sure they're equal sizes """
    # Extract all current axis properties before we start plotting anything new and changing them!
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()
    ax_min = min([x_min, y_min, z_min])
    ax_max = max([x_max, y_max, z_max])
    if square_axes:
        if abs(ax_min) > abs(ax_max):
            ax_max = -ax_min
        else:
            ax_min = -ax_max
    if unit_axes:
        if ax_max < 1:
            ax_min = -1
            ax_max = 1
    if uniform_axes:
        x_min = ax_min
        y_min = ax_min
        z_min = ax_min
        x_max = ax_max
        y_max = ax_max
        z_max = ax_max
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    z_ticks = ax.get_zticks()
    x_ticks = x_ticks[(x_ticks >= x_min) & (x_ticks <= x_max)]
    y_ticks = y_ticks[(y_ticks >= y_min) & (y_ticks <= y_max)]
    z_ticks = z_ticks[(z_ticks >= z_min) & (z_ticks <= z_max)]

    # Plot splines
    ax.plot([0, 0], [0, 0], [x_min, x_max], 'k', linewidth=1.5)
    ax.plot([0, 0], [y_min, y_max], [0, 0], 'k', linewidth=1.5)
    ax.plot([z_min, z_max], [0, 0], [0, 0], 'k', linewidth=1.5)

    # Import tick markers (use only those tick markers for the longest axis, as the changes are made to encourage a
    # square set of axes)
    x_tick_range = (x_max-x_min)/100
    y_tick_range = (y_max-y_min)/100
    z_tick_range = (z_max-z_min)/100
    for x_tick in x_ticks:
        ax.plot([x_tick, x_tick], [-x_tick_range, x_tick_range], [0, 0], 'k', linewidth=1.5)
    for y_tick in y_ticks:
        ax.plot([-y_tick_range, y_tick_range], [y_tick, y_tick], [0, 0], 'k', linewidth=1.5)
    for z_tick in z_ticks:
        ax.plot([0, 0], [-z_tick_range, z_tick_range], [z_tick, z_tick], 'k', linewidth=1.5)

    # Label tick markers (only at the extremes, to prevent a confusing plot)
    ax.text(x_ticks[0], -x_tick_range*12, 0, x_ticks[0], None)
    ax.text(x_ticks[-1], -x_tick_range*12, 0, x_ticks[-1], None)
    ax.text(y_tick_range*4, y_ticks[0], 0, y_ticks[0], None)
    ax.text(y_tick_range*4, y_ticks[-1], 0, y_ticks[-1], None)
    ax.text(z_tick_range*4, 0, z_ticks[0], z_ticks[0], None)
    ax.text(z_tick_range*4, 0, z_ticks[-1], z_ticks[-1], None)

    # Import axis labels
    ax.text(x_max+x_tick_range, 0, 0, ax.get_xlabel(), None)
    ax.text(0, y_max+y_tick_range, 0, ax.get_ylabel(), None)
    ax.text(0, 0, z_max+z_tick_range*4, ax.get_zlabel(), None)

    # Remove original axes
    ax.set_axis_off()
    return None


def plot_arc3d(vector1, vector2, radius=0.2, fig=None, colour='C0'):
    """ Plot arc between two given vectors in 3D space. """

    """ Confirm correct input arguments """
    assert len(vector1) == 3
    assert len(vector2) == 3

    """ Calculate vector between two vector end points, and the resulting spherical angles for various points along 
        this vector. From this, derive points that lie along the arc between vector1 and vector2 """
    v = [i-j for i, j in zip(vector1, vector2)]
    v_points_direct = [(vector2[0]+v[0]*l, vector2[1]+v[1]*l, vector2[2]+v[2]*l) for l in np.linspace(0, 1)]
    v_phis = [math.atan2(v_point[1], v_point[0]) for v_point in v_points_direct]
    v_thetas = [math.acos(v_point[2]/np.linalg.norm(v_point)) for v_point in v_points_direct]

    v_points_arc = [(radius*sin(theta)*cos(phi), radius*sin(theta)*sin(phi), radius*cos(theta))
                    for theta, phi in zip(v_thetas, v_phis)]
    v_points_arc.append((0, 0, 0))

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.gca()

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    """ Plot polygon (face colour must be set afterwards, otherwise it over-rides the transparency)
        https://stackoverflow.com/questions/18897786/transparency-for-poly3dcollection-plot-in-matplotlib """
    points_collection = Poly3DCollection([v_points_arc], alpha=0.4)
    points_collection.set_facecolor(colour)
    ax.add_collection3d(points_collection)

    return fig


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


def get_qrs_start_end(vcg, dt=2, velocity_offset=2, low_p=40, order=2, threshold_frac=0.15, filter_sv=True, t_end=200,
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


def get_spatial_velocity(vcg, velocity_offset=2, t_end=200, dt=2, threshold_frac=0.15, matlab_match=False,
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
            threshold = max(sim_sv)*threshold_frac
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
                if sim_x[0] > 50:
                    raise Exception('More than 50ms of trace removed - try changing threshold_frac')
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
    line_handles = None
    for qrs_limit in qrs_limits:
        line_handles = list()
        for i in range(len(qrs_limit)):
            line_handles.append(mlines.Line2D([], [], color=colours[i], label=legend[i]))
            if i > 0:
                if qrs_limit[i] <= qrs_limit[i-1]:
                    qrs_limit[i] += 0.1
            __plot_spatial_velocity_plot_limits(qrs_limit[i], ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z, colours[i])
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
        phi_weighted = [acos(sim_vcg_t[1] / dipole_magnitude_t) * dipole_magnitude_t
                        for (sim_vcg_t, dipole_magnitude_t) in zip(sim_vcg, dipole_magnitude)]
        # Weighted Azimuth
        theta_weighted = [atan(sim_vcg_t[2] / sim_vcg_t[0]) * dipole_magnitude_t
                          for (sim_vcg_t, dipole_magnitude_t) in zip(sim_vcg, dipole_magnitude)]

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
