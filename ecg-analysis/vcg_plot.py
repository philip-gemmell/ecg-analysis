import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
import numpy as np
from math import sin, cos, acos, atan2
import warnings

import common_analysis
import vcg_analysis

__all__ = ['Axes3D']    # Workaround to prevent Axes3D import statement to be labelled as unused


def plot_xyz_components(vcg, dt=2, legend=None, qrs_limits=None, layout=None, colours=None, linestyles=None,
                        limit_colours=None, limit_linestyles=None):
    """
    Plot x, y, z components of VCG data

    Multiple options given for layout of resulting plot

    Input parameters (required):
    ----------------------------

    vcg     List of vcg data: [vcg_data1, vcg_data2, ...]

    Input parameters (optional):
    ----------------------------
    dt                  2       Time interval for data
    legend              None    Legend names for each VCG trace
    qrs_limits          None    QRS limits to plot on axes
                                To be presented in form [[qrs_start1, qrs_starts, ...], [qrs_end1, qrs_end2, ...], ...]
    layout              None    Layout of resulting plot
                                figures     Each x,y,z plot is on a separate figure
                                combined    x,y,z plots are combined on a single set of axes
                                row         x,y,z plots are arranged on a horizontal row in one figure
                                column      x,y,z plots are arranged in a vertical column in one figure
                                best        x,y,z plots are arranged to try and optimise space
                                            (nb: figures not equal sizes...)
                                *grid       x,y,z plots are arranged in a grid (like best, but more rigid grid)
    colours             None    Colours to use for plotting
    linestyles          None    Linestyles to use for plotting
    limit_colours       None    Colours to use when plotting limits
    limit_linestyles    None    Linestyles to use when plotting limits

    if layout != 'combined':
        len(colours) == len(linestyles) == len(vcg)
    else:
        len(colours) == len(linestyles) == 3*len(vcg)

    len(limit_colours) == len(limit_linestyles) == recursive_len(qrs_limits)
    If no values passed, will default to same colour and linestyle for corresponding start/end limits, with linestyle
    and colour both varying between limits1, limits2, etc.

    Output parameters:
    ------------------

    fig     Handle for resulting figure(s)
    ax      Handle for resulting axis/axes

    """

    qrs_limits = __set_metric_to_metrics(qrs_limits)
    n_limits = common_analysis.recursive_len(qrs_limits)
    vcg, colours, linestyles, legend, layout, limit_colours, limit_linestyles = \
        __process_inputs(vcg, colours, linestyles, legend, layout, limit_colours, limit_linestyles,
                         (n_limits, len(qrs_limits[0])))
    fig, ax = __init_axes(layout)

    # Plot data and add legend if required
    xyz_label = ['x', 'y', 'z']
    time = [i * dt for i in range(len(vcg[0]))]
    for (sim_vcg, sim_legend, sim_colour, sim_linestyle) in zip(vcg, legend, colours, linestyles):
        if sim_vcg.shape[0] < sim_vcg.shape[1]:
            for (sim_vcg_xyz, xyz) in zip(sim_vcg, xyz_label):
                ax[xyz].plot(time, sim_vcg_xyz, label=sim_legend, color=sim_colour, linestyle=sim_linestyle)
        else:
            for (sim_vcg_xyz, xyz) in zip(sim_vcg.T, xyz_label):
                ax[xyz].plot(time, sim_vcg_xyz, label=sim_legend, color=sim_colour, linestyle=sim_linestyle)

    if legend[0] is not None:
        if layout == 'figures':
            for xyz in xyz_label:
                ax[xyz].legend()
        else:
            ax['x'].legend()

    if qrs_limits is not None:
        __plot_limits(ax, qrs_limits, colours, linestyles)

    return fig, ax


def __process_inputs(vcg, colours, linestyles, legend, layout, limit_colours, limit_linestyles, n_limits):
    """
    Assess and adapt input arguments to plot_vcg_multiple.

    Input parameters:
    -----------------

    vcg                 VCG data
                        [vcg1, vcg2, ...]
    colours             Colours to be used to plot
                        [colour1, colour2, ...]
    linestyles          Linestyles to be used to plot data
                        [linestyle1, linestyle2, ...]
    legend              Labels for each VCG plot
    layout              Assigned figure layout (used to determine the number of different colours/linestyles to be
                        required)
    limit_colours       Colours to be used to plot QRS limits
    limit_linestyles    Linestyles to be used to plot QRS limits
    n_limits            (total number of limits to plot, number of each limit to plot)
                        e.g. plotting 3 different QRS start/end values -> (6, 3)
    """

    if not isinstance(vcg, list):
        vcg = [vcg]

    if legend is not None:
        assert len(vcg) == len(legend)
        plt.rc('text', usetex=True)
    else:
        legend = [None for _ in vcg]

    if layout is None:
        layout = 'grid'
    if layout == 'combined':
        n_colours = len(vcg)*3
    else:
        n_colours = len(vcg)

    # Adapt colours and linestyle depending on whether x,y,z plots are combined. Preference is to plot different colours
    # for different VCG traces, with different linestyles representing different x/y/z
    if colours is None:
        colours = common_analysis.get_plot_colours(n_colours)
    elif isinstance(colours, list):
        assert len(colours) == n_colours
    else:
        colours = [colours for _ in range(n_colours)]

    if linestyles is None:
        linestyles = ['-' for _ in range(n_colours)]
    elif isinstance(linestyles, list):
        assert len(linestyles) == n_colours
    else:
        linestyles = [linestyles for _ in range(n_colours)]

    if limit_colours is None:
        limit_colours = common_analysis.get_plot_colours(n_limits)
    elif isinstance(limit_colours, list):
        assert len(limit_colours) == n_limits
    else:
        limit_colours = [limit_colours for _ in range(n_limits)]

    if limit_linestyles is None:
        limit_linestyles = ['-' for _ in range(n_limits)]
    elif isinstance(limit_linestyles, list):
        assert len(limit_linestyles) == n_limits
    else:
        limit_linestyles = [limit_linestyles for _ in range(n_limits)]

    return vcg, colours, linestyles, legend, layout, limit_colours, limit_linestyles


def __init_axes(layout):
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
    elif layout == 'combined':
        ax['x'] = fig.add_subplot(1, 1, 1)
        ax['y'] = ax['x']
        ax['z'] = ax['x']
    elif layout == 'row':
        gs = gridspec.GridSpec(3, 1)
        ax['x'] = fig_h[0].add_subplot(gs[0], ylabel='x')
        ax['y'] = fig_h[1].add_subplot(gs[1], ylabel='y', sharex=ax['x'], sharey=ax['x'])
        ax['z'] = fig_h[2].add_subplot(gs[2], ylabel='z', sharex=ax['x'], sharey=ax['x'])
        plt.setp(ax['y'].get_yticklabels(), visible=False)
        plt.setp(ax['z'].get_yticklabels(), visible=False)
        gs.update(wspace=0.025, hspace=0.05)
    elif layout == 'column':
        gs = gridspec.GridSpec(1, 3)
        ax['x'] = fig_h[0].add_subplot(gs[0], ylabel='x')
        ax['y'] = fig_h[1].add_subplot(gs[1], ylabel='y', sharex=ax['x'], sharey=ax['x'])
        ax['z'] = fig_h[2].add_subplot(gs[2], ylabel='z', sharex=ax['x'], sharey=ax['x'])
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
        plt.setp(ax['x'].get_xticklabels(), visible=False)
        plt.setp(ax['y'].get_yticklabels(), visible=False)
        gs.update(wspace=0.025, hspace=0.05)
    return fig, ax


def __plot_limits(axes, limits, colours, linestyles):
    """ Plot limits to a given plot (e.g. add line marking start of QRS complex) """

    if not isinstance(limits, list):
        limits = [limits]
    i = 0
    for sim_limit in limits:
        for sim_limit_startEnd in sim_limit:
            for key in axes:
                axes[key].axvline(sim_limit_startEnd, color=colours[i], alpha=0.5, linestyle=linestyles[i])
            i += 1
    return None


def plot_2d(vcg_x, vcg_y, xlabel='VCG (x)', ylabel='VCG (y)', linestyle='-', colourmap='viridis', linewidth=3,
            axis_limits=None, time_limits=None, fig=None):
    """
    Plot x vs y (or y vs z, or other combination) for VCG trace, with line colour shifting to show time progression.

    Plot a colour-varying course of a VCG in 2D space

    Input parameters (required):
    ----------------------------

    vcg_x       VCG data to be plotted along the x-axis of the output
    vcg_y       VCG data to be plotted along the y-axis of the output

    Input parameters (optional):
    ----------------------------

    xlabel          'VCG (x)'   Label to apply to the x-axis
    ylabel          'VCG (y)'   Label to apply to the y-axis
    linestyle       '-'         Linestyle to apply to the plot
    colourmap       'viridis'   Colourmap to use for the line
    linewidth       3           Linewidth to use
    axis_limits     None        Limits to apply to the axes
    time_limits     None        Start and end time of data. If given, will add a colourbar to the plot
    fig             None        Handle to pre-existing figure (if present) on which to plot data

    Output parameters:
    ------------------

    fig         Handle to output figure window

    """

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = fig.gca()

    # Prepare line segments for plotting
    t = np.linspace(0, 1, vcg_x.shape[0])
    points = np.array([vcg_x, vcg_y]).transpose().reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=plt.get_cmap(colourmap), linestyle=linestyle, linewidths=linewidth)
    lc.set_array(t)

    # Add the collection to the plot
    ax.add_collection(lc)
    # Line collections don't auto-scale the plot - set it up for a square plot
    __set_axis_limits([vcg_x, vcg_y], ax, unit_min=False, axis_limits=axis_limits)

    # Change the positioning of the axes
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

    if time_limits is not None:
        __add_colourbar(time_limits, colourmap, vcg_x.shape[0])

    return fig


def plot_3d(vcg, linestyle='-', colourmap='viridis', linewidth=3, axis_limits=None, time_limits=None, fig=None):
    """
    Plot the evolution of VCG in 3D space

    Input parameters (required):
    ----------------------------

    vcg     VCG data

    Input parameters (optional):
    ----------------------------

    linestyle       '-'         Linestyle to plot data
    colourmap       'viridis'   Colourmap to use when plotting data
    linewidth       3           Linewidth to use
    axis_limits     None        Limits to apply to the axes
    time_limits     None        Start and end time of data. If given, will add a colourbar to the plot
    fig             None        Handle to existing figure (if exists)

    Output parameters:
    ------------------

    fig     Figure handle
    """

    vcg_x, vcg_y, vcg_z = __get_xyz_from_vcg(vcg)

    # Prepare line segments for plotting
    t = np.linspace(0, 1, vcg_x.shape[0])  # "time" variable
    points = np.array([vcg_x, vcg_y, vcg_z]).transpose().reshape(-1, 1, 3)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segs, cmap=plt.get_cmap(colourmap), linestyle=linestyle, linewidths=linewidth)
    lc.set_array(t)

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.gca()

    # add the collection to the plot
    ax.add_collection3d(lc)

    # Set axis limits (not automatic for line collections)
    __set_axis_limits([vcg_x, vcg_y, vcg_z], ax, axis_limits=axis_limits)

    ax.set_xlabel('VCG (x)')
    ax.set_ylabel('VCG (y)')
    ax.set_zlabel('VCG (z)')

    if time_limits is not None:
        __add_colourbar(time_limits, colourmap, vcg_x.shape[0])

    return fig


def animate_3d(vcg, limits=None, linestyle='-', output_file='vcg_xyz.mp4'):
    """
    Animate the evolution of the VCG in 3D space, saving that animation to a file.

    Input parameters (required):
    ----------------------------

    vcg         VCG data

    Input parameters (optional):
    ----------------------------

    limits          None            Limits for the axes. If none, will set to the min/max values of the provided data
    linestyle       '-'             Linestyle for the data
    output_file     'vcg_xyz.mp4'   Name of the file to save the animation to

    Output parameters:
    ------------------

    None
    """

    from matplotlib import animation

    vcg_x, vcg_y, vcg_z = __get_xyz_from_vcg(vcg)

    # Extract limits
    if limits is None:
        max_lim = max(max(vcg_x), max(vcg_y), max(vcg_z))
        min_lim = min(min(vcg_x), min(vcg_y), min(vcg_z))
        limits = [min_lim, max_lim]

    # Process inputs to ensure the correct formats are used.
    if linestyle is None:
        linestyle = '-'

    # Set up figure and axes
    fig = plt.figure()
    ax = Axes3D(fig)
    add_xyz_axes(ax, axis_limits=limits, symmetrical_axes=False, equal_limits=False, unit_axes=False)
    line, = ax.plot([], [], lw=3)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        # Prepare line segments for plotting
        t = np.linspace(0, 1, vcg_x[:i].shape[0])  # "time" variable
        points = np.array([vcg_x[:i], vcg_y[:i], vcg_z[:i]]).transpose().reshape(-1, 1, 3)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segs, cmap=plt.get_cmap(colourmap), linestyle=linestyle, linewidths=linewidth)
        lc.set_array(t)
        ax.add_collection3d(lc)
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(vcg_x), interval=30, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be installed.  The extra_args ensure that the
    # x264 codec is used, so that the video can be embedded in html5.  You may need to adjust this for your system: for
    # more information, see http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save(output_file, fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()
    return None


def __add_colourbar(limits, colourmap, n_elements):
    """ Add arbitrary colourbar to a figure. """

    cmap = plt.get_cmap(colourmap, n_elements)
    norm = mpl.colors.Normalize(vmin=limits[0], vmax=limits[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    return None


def __get_xyz_from_vcg(vcg):
    """ Extract individual elements of VCG (x, y, z). """

    if vcg.shape[0] == 3:
        vcg_x = vcg[0, :]
        vcg_y = vcg[1, :]
        vcg_z = vcg[2, :]
    else:
        vcg_x = vcg[:, 0]
        vcg_y = vcg[:, 1]
        vcg_z = vcg[:, 2]

    return vcg_x, vcg_y, vcg_z


def plot_xyz_vector(vector=None, x=None, y=None, z=None, fig=None, linecolour='C0', linestyle='-', linewidth=2):
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

    a = Arrow3D([0, x], [0, y], [0, z], mutation_scale=20, lw=linewidth, arrowstyle="-|>", color=linecolour,
                linestyle=linestyle)
    ax.add_artist(a)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('VCG (x)')
    ax.set_ylabel('VCG (y)')
    ax.set_zlabel('VCG (z)')

    return fig


def __set_axis_limits(data, ax, unit_min=True, axis_limits=None):
    """ Set axis limits (not automatic for line collections) """
    if axis_limits is None:
        ax_min = min([i.min() for i in data])
        ax_max = max([i.max() for i in data])
        if abs(ax_min) > abs(ax_max):
            ax_max = -ax_min
        else:
            ax_min = -ax_max
        if unit_min:
            if ax_max < 1:
                ax_min = -1
                ax_max = 1
    else:
        if isinstance(axis_limits, list):
            ax_min = axis_limits[0]
            ax_max = axis_limits[1]
        else:
            if axis_limits < 0:
                axis_limits = -axis_limits
            ax_min = -axis_limits
            ax_max = axis_limits
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


def add_xyz_axes(ax, axis_limits=None, symmetrical_axes=False, equal_limits=False, unit_axes=False):
    """ Plot dummy axes (can't move splines in 3D plots)
        ax                  Axis handles
        symmetrical_axes    Boolean: apply same limits to x, y and z axes
        equal_limits        Boolean: set axis minimum to minus axis maximum (or vice versa)
        unit_axes           Boolean: apply minimum of -1 -> 1 for axis limits
    """

    """ Construct dummy 3D axes - make sure they're equal sizes """
    # Extract all current axis properties before we start plotting anything new and changing them!
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()
    ax_min = min([x_min, y_min, z_min])
    ax_max = max([x_max, y_max, z_max])
    if equal_limits:
        x_min, x_max = __set_symmetrical_axis_limits(x_min, x_max, unit_axes=unit_axes)
        y_min, y_max = __set_symmetrical_axis_limits(y_min, y_max, unit_axes=unit_axes)
        z_min, z_max = __set_symmetrical_axis_limits(z_min, z_max, unit_axes=unit_axes)
        ax_min, ax_max = __set_symmetrical_axis_limits(ax_min, ax_max, unit_axes=unit_axes)
    if symmetrical_axes:
        x_min = ax_min
        y_min = ax_min
        z_min = ax_min
        x_max = ax_max
        y_max = ax_max
        z_max = ax_max
    if axis_limits is not None:
        if not isinstance(axis_limits, list):
            if axis_limits < 0:
                axis_limits = -axis_limits
            if -axis_limits > min([x_min, y_min, z_min]):
                warnings.warn('Lower limit provided greater than automatic.')
            if axis_limits < max([x_max, y_max, z_max]):
                warnings.warn('Upper limit provided less than automatic.')
            x_min = -axis_limits
            x_max = axis_limits
            y_min = -axis_limits
            y_max = axis_limits
            z_min = -axis_limits
            z_max = axis_limits
        elif not isinstance(axis_limits[0], list):
            # If same axis limits applied to all 3 dimensions
            if axis_limits[0] > min([x_min, y_min, z_min]):
                warnings.warn('Lower limit provided greater than automatic.')
            if axis_limits[1] < max([x_max, y_max, z_max]):
                warnings.warn('Upper limit provided less than automatic.')
            x_min = axis_limits[0]
            x_max = axis_limits[1]
            y_min = axis_limits[0]
            y_max = axis_limits[1]
            z_min = axis_limits[0]
            z_max = axis_limits[1]
        else:
            # Different axis limits provided for each dimension
            x_min = axis_limits[0][0]
            x_max = axis_limits[0][1]
            y_min = axis_limits[1][0]
            y_max = axis_limits[1][1]
            z_min = axis_limits[2][0]
            z_max = axis_limits[2][1]
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

    # Remove original axes, and eliminate whitespace
    ax.set_axis_off()
    plt.subplots_adjust(left=-0.4, right=1.4, top=1.35, bottom=-0.4)
    return None


def __set_symmetrical_axis_limits(ax_min, ax_max, unit_axes=False):
    if abs(ax_min) > abs(ax_max):
        ax_max = -ax_min
    else:
        ax_min = -ax_max

    if unit_axes:
        if ax_max < 1:
            ax_max = 1
            ax_min = -1
    return ax_min, ax_max


def plot_arc3d(vector1, vector2, radius=0.2, fig=None, colour='C0'):
    """ Plot arc between two given vectors in 3D space. """

    """ Confirm correct input arguments """
    assert len(vector1) == 3
    assert len(vector2) == 3

    """ Calculate vector between two vector end points, and the resulting spherical angles for various points along 
        this vector. From this, derive points that lie along the arc between vector1 and vector2 """
    v = [i-j for i, j in zip(vector1, vector2)]
    v_points_direct = [(vector2[0]+v[0]*l, vector2[1]+v[1]*l, vector2[2]+v[2]*l) for l in np.linspace(0, 1)]
    v_phis = [atan2(v_point[1], v_point[0]) for v_point in v_points_direct]
    v_thetas = [acos(v_point[2]/np.linalg.norm(v_point)) for v_point in v_points_direct]

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


def plot_spatial_velocity(vcg, sv=None, qrs_limits=None, fig=None, legend=None, t_end=200, dt=2, filter_sv=True):
    """ Plot the spatial velocity and VCG elements, with limits (e.g. QRS limits) if provided. Note that if spatial
        velocity is not provided, default values will be used to calculate it - if anything else is desired,
        then spatial velocity must be calculated first and provided to the function.
    """

    vcg, legend = __plot_spatial_velocity_preprocess_inputs(vcg, legend)
    fig, ax, colours = __plot_spatial_velocity_prep_axes(vcg, fig)
    x_val, sv = __plot_spatial_velocity_get_plot_data(sv, vcg, t_end, dt, filter_sv)

    """ Plot spatial velocity and VCG components"""
    i_colour_init = get_i_colour(ax['sv'])
    i_colour = i_colour_init
    x_vcg_data = list(range(0, t_end + dt, dt))
    for (sim_x, sim_vcg, sim_sv, sim_label) in zip(x_val, vcg, sv, legend):
        __plot_spatial_velocity_plot_data(sim_x, sim_sv, x_vcg_data, sim_vcg, sim_label, colours[i_colour], ax)
        i_colour += 1

    """ Plot QRS limits, if provided """
    if qrs_limits is not None:
        # Cycle through each limit provided, e.g. QRS start, QRS end...
        for qrs_limit in qrs_limits:
            i_colour = i_colour_init
            assert len(qrs_limit) == len(vcg)

            # Plot limits for each given VCG
            for sim_qrs_limit in qrs_limit:
                __plot_spatial_velocity_plot_limits(sim_qrs_limit, ax, colours[i_colour])
                i_colour += 1

    """ Add legend (or legend for limits, if appropriate """
    if legend[0] is not None:
        labels = [line.get_label() for line in ax['sv'].get_lines()]
        plt.rc('text', usetex=True)
        ax['sv'].legend(labels)

    return fig, ax


def plot_spatial_velocity_multilimit(vcg, sv=None, qrs_limits=None, fig=None, legend=None, t_end=200, dt=2,
                                     filter_sv=True):
    """ Plot a single instance of a spatial velocity curve, but with multiple limits for QRS """

    """ Confirm VCG and limit data are correctly formatted """
    assert isinstance(vcg, np.ndarray)
    for qrs_limit in qrs_limits:
        assert len(qrs_limit) == len(qrs_limits[0])

    fig, ax, colours = __plot_spatial_velocity_prep_axes(vcg, fig)
    vcg, legend = __plot_spatial_velocity_preprocess_inputs(vcg, legend)
    x_val, sv = __plot_spatial_velocity_get_plot_data(sv, vcg, t_end, dt, filter_sv)

    """ Plot spatial velocity and VCG components"""
    x_vcg_data = list(range(0, t_end + dt, dt))
    __plot_spatial_velocity_plot_data(x_val[0], sv[0], x_vcg_data, vcg[0], None, None, ax)

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
            __plot_spatial_velocity_plot_limits(qrs_limit[i], ax, colours[i])
    ax['sv'].legend(handles=line_handles)
    return None


def __plot_spatial_velocity_preprocess_inputs(vcg, legend):
    """ Preprocess other inputs """
    if isinstance(vcg, np.ndarray):
        vcg = [vcg]
    if isinstance(legend, str):
        legend = [legend]
    elif legend is None:
        if len(vcg) > 1:
            legend = [str(i) for i in range(len(vcg))]
        else:
            legend = [None for _ in range(len(vcg))]
    return vcg, legend


def __plot_spatial_velocity_prep_axes(vcg, fig):
    """ Prepare figure and axes """
    if fig is None:
        fig = plt.figure()
        ax = dict()
        gs = gridspec.GridSpec(3, 3)
        ax['sv'] = fig.add_subplot(gs[:, :-1])
        ax['vcg_x'] = fig.add_subplot(gs[0, -1])
        ax['vcg_y'] = fig.add_subplot(gs[1, -1])
        ax['vcg_z'] = fig.add_subplot(gs[2, -1])
        plt.setp(ax['vcg_x'].get_xticklabels(), visible=False)
        plt.setp(ax['vcg_y'].get_xticklabels(), visible=False)
        gs.update(hspace=0.05)
        colours = common_analysis.get_plot_colours(len(vcg))
    else:
        ax = dict()
        ax_sv, ax_vcg_x, ax_vcg_y, ax_vcg_z = fig.get_axes()
        ax['sv'] = ax_sv
        ax['vcg_x'] = ax_vcg_x
        ax['vcg_y'] = ax_vcg_y
        ax['vcg_z'] = ax_vcg_z
        colours = common_analysis.get_plot_colours(len(ax_sv.lines) + len(vcg))
        """ If too many lines already exist on the plot, need to recolour them all to prevent cross-talk """
        if len(ax_sv.lines) + len(vcg) > 10:
            for key in ax:
                lines = ax[key].get_lines()
                i_vcg = 0
                for line in lines:
                    line.set_color(colours[i_vcg])
                    i_vcg += 1

    """ Add labels to axes """
    ax['sv'].set_xlabel('Time (ms)')
    ax['sv'].set_ylabel('Spatial velocity')
    ax['vcg_x'].set_ylabel('VCG (x)')
    ax['vcg_y'].set_ylabel('VCG (y)')
    ax['vcg_z'].set_ylabel('VCG (z)')
    ax['vcg_z'].set_xlabel('Time (ms)')

    return fig, ax, colours


def __plot_spatial_velocity_get_plot_data(sv, vcg, t_end, dt, filter_sv):
    """ Prepare spatial velocity """
    if sv is None:
        x_val, sv, _, _ = vcg_analysis.get_spatial_velocity(vcg=vcg, t_end=t_end, dt=dt, filter_sv=filter_sv)
    else:
        x_val = list()
        for sim_sv in sv:
            x_val.append([(i * dt) + 2 for i in range(len(sim_sv))])
    return x_val, sv


def __plot_spatial_velocity_plot_data(x_sv_data, sv_data, x_vcg_data, vcg_data, data_label, plot_colour, ax):
    ax['vcg_x'].plot(x_vcg_data, vcg_data[:, 0], color=plot_colour)
    ax['vcg_y'].plot(x_vcg_data, vcg_data[:, 1], color=plot_colour)
    ax['vcg_z'].plot(x_vcg_data, vcg_data[:, 2], color=plot_colour)
    ax['sv'].plot(x_sv_data, sv_data, color=plot_colour, label=data_label)
    return None


def __plot_spatial_velocity_plot_limits(qrs_limit, ax, limit_colour):
    for key in ax:
        ax[key].axvspan(qrs_limit, qrs_limit+0.1, color=limit_colour, alpha=0.5)
    return None


def get_i_colour(axis_handle):
    """ Get index appropriate to colour value to plot on a figure (will be 0 if brand new figure) """
    if axis_handle is None:
        return 0
    else:
        if len(axis_handle.lines) == 0:
            return 0
        else:
            return len(axis_handle.lines)-1


def plot_metric_change(metrics, metrics_phi, metrics_rho, metrics_z, metric_name, metrics_lv=None,
                       labels=None, scattermarkers=None, linemarkers=None, colours=None, linestyles=None,
                       layout=None, axis_match=True, no_labels=False):
    """ Function to plot all the various figures for trend analysis in one go. """
    plt.rc('text', usetex=True)

    """ Underlying constants (volumes, sizes, labels, etc.) """
    # Create volume variables (nb: percent of whole mesh)
    vol_lv_phi = [0.0, 2.657, 8.667, 14.808]
    vol_lv_rho = [0.0, 3.602, 7.243, 10.964, 14.808]
    vol_lv_z = [0.0, 6.183, 10.897, 14.808]
    vol_lv_size = [0.0, 0.294, 4.062, 14.808]
    vol_lv_other = [5.333]

    vol_septum_phi = [0.0, 6.926, 11.771, 17.019, 21.139]
    vol_septum_rho = [0.0, 5.105, 10.275, 15.586, 21.139]
    vol_septum_z = [0.0, 8.840, 15.818, 21.139]
    vol_septum_size = [0.0, 0.672, 6.531, 21.139]
    vol_septum_other = [8.944]

    volume_lv = vol_lv_phi + vol_lv_rho + vol_lv_z + vol_lv_size + vol_lv_other
    volume_septum = vol_septum_phi + vol_septum_rho + vol_septum_z + vol_septum_size + vol_septum_other

    # Create area variables (in cm^2)
    area_lv_phi = [0.0, 37.365, 85.575, 129.895]
    area_lv_rho = [0.0, 109.697, 115.906, 122.457, 129.895]
    area_lv_z = [0.0, 57.847, 97.439, 129.895]
    area_lv_size = [0.0, 10.140, 57.898, 129.895]
    area_lv_other = [76.501]

    area_septum_phi = [0.0, 56.066, 88.603, 122.337, 149.588]
    area_septum_rho = [0.0, 126.344, 133.363, 141.091, 149.588]
    area_septum_z = [0.0, 72.398, 114.937, 149.588]
    area_septum_size = [0.0, 17.053, 72.104, 149.588]
    area_septum_other = [97.174]

    area_lv = area_lv_phi + area_lv_rho + area_lv_z + area_lv_size + area_lv_other
    area_septum = area_septum_phi + area_septum_rho + area_septum_z + area_septum_size + area_septum_other
    area_lv_norm = [i/area_septum_phi[-1] for i in area_lv]
    area_septum_norm = [i/area_septum_phi[-1] for i in area_septum]

    legend_phi_lv = ['None', r'$1.4\pi/2$ \textrightarrow $1.6\pi/2$', r'$1.2\pi/2$ \textrightarrow $1.8\pi/2$',
                     r'$\pi/2$ \textrightarrow $\pi$']
    legend_phi_septum = ['None', r'-0.25 \textrightarrow 0.25', r'-0.50 \textrightarrow 0.25',
                         r'-0.75 \textrightarrow 0.75', r'-1.00 \textrightarrow 1.00']
    legend_rho = ['None', r'0.4 \textrightarrow 0.6', r'0.3 \textrightarrow 0.7', r'0.2 \textrightarrow 0.8',
                  r'0.1 \textrightarrow 0.9']
    legend_z = ['None', r'0.5 \textrightarrow 0.7', r'0.4 \textrightarrow 0.8', r'0.3 \textrightarrow 0.9']

    """ Assert correct data has been passed (insofar that it is of the right length!) """
    metrics = __set_metric_to_metrics(metrics)
    metrics_phi = __set_metric_to_metrics(metrics_phi)
    metrics_rho = __set_metric_to_metrics(metrics_rho)
    metrics_z = __set_metric_to_metrics(metrics_z)

    if metrics_lv is None:
        metrics_lv = [True, False]
    if labels is None:
        labels = ['LV', 'Septum']
    volumes = [volume_lv if metric_lv else volume_septum for metric_lv in metrics_lv]
    for metric, volume in zip(metrics, volumes):
        assert len(metric) == len(volume)
    areas = [area_lv_norm if metric_lv else area_septum_norm for metric_lv in metrics_lv]
    for metric, area in zip(metrics, areas):
        assert len(metric) == len(area)

    if scattermarkers is None:
        scattermarkers = ['+', 'o', 'D', 'v', '^', 's', '*', 'x']
    assert len(scattermarkers) >= len(metrics)
    if linemarkers is None:
        linemarkers = ['.' for _ in range(len(metrics_rho))]
    else:
        assert len(linemarkers) >= len(metrics_rho)
    if linestyles is None:
        linestyles = ['-' for _ in range(len(metrics_rho))]
    else:
        assert len(linestyles) >= len(metrics_rho)
    if colours is None:
        colours = common_analysis.get_plot_colours(len(metrics_rho))
    else:
        assert len(colours) >= len(metrics_rho)

    """ Set up figures and axes """
    keys = ['volume', 'area', 'phi_lv', 'phi_septum', 'rho', 'z']
    if (layout is None) or (layout == 'combined'):
        fig = plt.figure()
        fig.suptitle(metric_name)
        gs = gridspec.GridSpec(4, 3)
        ax = dict()
        ax['volume'] = fig.add_subplot(gs[:2, :2])
        ax['area'] = fig.add_subplot(gs[2:, :2])
        ax['phi_lv'] = fig.add_subplot(gs[0, 2])
        ax['phi_septum'] = fig.add_subplot(gs[1, 2])
        ax['rho'] = fig.add_subplot(gs[2, 2])
        ax['z'] = fig.add_subplot(gs[3, 2])
        # plt.setp(ax['y'].get_yticklabels(), visible=False)
        gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.09, hspace=0.25)
    elif layout == 'figures':
        fig = {key: plt.figure() for key in keys}
        ax = {key: fig[key].add_subplot(1, 1, 1) for key in keys}
        if not no_labels:
            for key in keys:
                ax[key].set_ylabel(metric_name)
    else:
        print("Unrecognised layout command.")
        return None, None

    """ Plot data on axes """
    # Volume
    for (metric, volume, label, colour, scattermarker) in zip(metrics, volumes, labels, colours, scattermarkers):
        ax['volume'].plot(volume, metric, label=label, linestyle='None', color=colour, marker=scattermarker,
                          markersize=10, markeredgewidth=3, markerfacecolor='none')
    # ax['volume'].plot(volume_lv, metric_lv, '+', label='LV', markersize=10, markeredgewidth=3, color='C0')
    # ax['volume'].plot(volume_septum, metric_septum, 'o', markersize=10, markeredgewidth=3, label='Septum',
    #                       markerfacecolor='none', color='C1')
    if no_labels:
        plt.setp(ax['volume'].get_xticklabels(), visible=False)
        plt.setp(ax['volume'].get_yticklabels(), visible=False)
        ax['volume'].set_title(r'Volume of scar (\% of mesh)')
    else:
        ax['volume'].legend()
        ax['volume'].set_xlabel(r'Volume of scar (\% of mesh)')

    # Area
    for (metric, area, label, colour, scattermarker) in zip(metrics, areas, labels, colours, scattermarkers):
        ax['area'].plot(area, metric, label=label, linestyle='None', color=colour, marker=scattermarker,
                          markersize=10, markeredgewidth=3, markerfacecolor='none')
    # ax['area'].plot(area_lv_norm, metric_lv, '+', label='LV', markersize=10, markeredgewidth=3, color='C0')
    # ax['area'].plot(area_septum_norm, metric_septum, 'o', markersize=10, markeredgewidth=3, label='Septum',
    #                 markerfacecolor='none', color='C1')
    if no_labels:
        plt.setp(ax['area'].get_xticklabels(), visible=False)
        plt.setp(ax['area'].get_yticklabels(), visible=False)
        ax['area'].set_title(r'Surface Area of scar (normalised)')
    else:
        ax['area'].set_xlabel(r'Surface Area of scar (normalised)')

    # Phi (LV)
    for (metric, label, colour, marker, linestyle, metric_lv) in zip(metrics_phi, labels, colours, linemarkers,
                                                                     linestyles, metrics_lv):
        if metric_lv:
            ax['phi_lv'].plot(metric, label=label, linestyle=linestyle, color=colour, marker=marker, linewidth=3)
        else:
            ax['phi_septum'].plot(metric, label=label, linestyle=linestyle, color=colour, marker=marker,
                                  linewidth=3)
    # ax['phi_lv'].plot(metric_phi_lv, 'o-', label='LV', linewidth=3, color='C0')
    ax['phi_lv'].set_xticks(list(range(len(legend_phi_lv))))
    if no_labels:
        plt.setp(ax['phi_lv'].get_xticklabels(), visible=False)
        plt.setp(ax['phi_lv'].get_yticklabels(), visible=False)
        ax['phi_lv'].set_title(r'$\phi$')
    else:
        ax['phi_lv'].set_xticklabels(legend_phi_lv)
        ax['phi_lv'].set_xlabel(r'$\phi$')

    # Phi (septum)
    # ax['phi_septum'].plot(metric_phi_septum, 'o-', label='Septum', linewidth=3, color='C1')
    ax['phi_septum'].set_xticks(list(range(len(legend_phi_septum))))
    if no_labels:
        plt.setp(ax['phi_septum'].get_xticklabels(), visible=False)
        plt.setp(ax['phi_septum'].get_yticklabels(), visible=False)
        ax['phi_septum'].set_title(r'$\phi$')
    else:
        ax['phi_septum'].set_xticklabels(legend_phi_septum)
        ax['phi_septum'].set_xlabel(r'$\phi$')

    # Rho
    for (metric, label, colour, marker, linestyle, metric_lv) in zip(metrics_rho, labels, colours, linemarkers,
                                                                     linestyles, metrics_lv):
        ax['rho'].plot(metric, label=label, linestyle=linestyle, color=colour, marker=marker, linewidth=3)
    # ax['rho'].plot(metric_rho_lv, 'o-', label='LV', linewidth=3, color='C0')
    # ax['rho'].plot(metric_rho_septum, 'o-', label='Septum', linewidth=3, color='C1')
    ax['rho'].set_xticks(list(range(len(legend_rho))))
    if no_labels:
        plt.setp(ax['rho'].get_xticklabels(), visible=False)
        plt.setp(ax['rho'].get_yticklabels(), visible=False)
        ax['rho'].set_title(r'$\rho$')
    else:
        ax['rho'].set_xticklabels(legend_rho)
        ax['rho'].set_xlabel(r'$\rho$')

    # z
    for (metric, label, colour, marker, linestyle, metric_lv) in zip(metrics_z, labels, colours, linemarkers,
                                                                     linestyles, metrics_lv):
        ax['z'].plot(metric, label=label, linestyle=linestyle, color=colour, marker=marker, linewidth=3)
    # ax['z'].plot(metric_z_lv, 'o-', label='LV', linewidth=3, color='C0')
    # ax['z'].plot(metric_z_septum, 'o-', label='Septum', linewidth=3, color='C1')
    ax['z'].set_xticks(list(range(len(legend_z))))
    if no_labels:
        plt.setp(ax['z'].get_xticklabels(), visible=False)
        plt.setp(ax['z'].get_yticklabels(), visible=False)
        ax['z'].set_title(r'$z$')
    else:
        ax['z'].set_xticklabels(legend_z)
        ax['z'].set_xlabel(r'$z$')

    if axis_match:
        ax_limits = ax['volume'].get_ylim()
        for key in keys:
            ax[key].set_ylim(ax_limits)

    return fig, ax


def __set_metric_to_metrics(metric):
    """ Function to change single list of metrics to list of one entry if required (so loops work correctly) """
    if not isinstance(metric[0], list):
        return [metric]
    else:
        return metric


def plot_metric_change_barplot(metrics_cont, metrics_lv, metrics_sept, metric_labels, layout=None):
    """ Plots a bar chart for the observed metrics. """

    """ Conduct initial checks, and set up values appropriate to plotting """
    assert len(metrics_cont) == len(metrics_lv)
    assert len(metrics_cont) == len(metrics_sept)
    assert len(metrics_cont) == len(metric_labels)

    if layout is None:
        layout = 'combined'

    if layout == 'combined':
        fig = plt.figure()
        gs = gridspec.GridSpec(1, len(metrics_cont))
        axes = list()
        for i, metric_label in enumerate(metric_labels):
            axes.append(fig.add_subplot(gs[i]))
    elif layout == 'fig':
        fig = [plt.figure() for _ in range(len(metrics_cont))]
        axes = [fig_i.add_subplot(1, 1, 1) for fig_i in fig]
    else:
        print("Invalid argument given for layout")
        return None, None
    # fig, ax = plt.subplots()

    # index = np.arange(len(metrics_cont))
    index = [0, 1, 2]
    bar_width = 1.2
    opacity = 0.8

    for ax, metric_cont, metric_lv, metric_sept, label in zip(axes, metrics_cont, metrics_lv, metrics_sept,
                                                              metric_labels):
        ax.bar(index[0], metric_cont, label='Control', alpha=opacity, color='C2', width=bar_width)
        ax.bar(index[1], metric_lv, label='LV Scar', alpha=opacity, color='C0', width=bar_width)
        ax.bar(index[2], metric_sept, label='Septum Scar', alpha=opacity, color='C1', width=bar_width)
        ax.set_title(label)
        ax.set_xticks([])
    axes[-1].legend()

    # """ Plot bar charts """
    # plt.bar(index-bar_width, metrics_cont, bar_width, label='Control')
    # plt.bar(index, metrics_lv, bar_width, label='LV Scar')
    # plt.bar(index+bar_width, metrics_sept, label='Septum Scar')
    #
    # """ Add labels """
    # ax.set_ylabel('Fractional Change')
    # ax.legend()
    # ax.set_xticklabels(index, metric_labels)

    return fig, axes


def plot_density_effect(metrics, metric_name, metric_labels=None, density_labels=None, linestyles=None, colours=None,
                        markers=None):
    """ Plot the effect of density on metrics. """
    preamble = {
        'text.usetex': True,
        'text.latex.preamble': [r'\usepackage{amsmath}']
    }
    plt.rcParams.update(preamble)
    # plt.rc('text', usetex=True)

    """ Process input arguments. """
    if not isinstance(metrics[0], list):
        metrics = [metrics]
    if metric_labels is None:
        if len(metrics) == 2:
            warnings.warn("Assuming metrics passed in order [LV, Septum].")
            metric_labels = ['LV', 'Septum']
        else:
            metric_labels = [None for _ in range(len(metrics))]
    else:
        assert len(metrics) == len(metric_labels)
    if linestyles is None:
        linestyles = ['-' for _ in range(len(metrics))]
    else:
        assert len(metrics) == len(linestyles)
    if colours is None:
        colours = common_analysis.get_plot_colours(len(metrics))
    else:
        assert len(metrics) == len(colours)
    if markers is None:
        markers = ['o' for _ in range(len(metrics))]
    else:
        assert len(metrics) == len(markers)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (metric, label, linestyle, colour, marker) in zip(metrics, metric_labels, linestyles, colours, markers):
        ax.plot(metric, linestyle=linestyle, marker=marker, color=colour, label=label, linewidth=3)

    if density_labels is None:
        density_labels = ['None',
                          r'\begingroup\addtolength{\jot}{-2mm}\begin{align*}'
                          r'p_\mathrm{low}&=0.2\\'
                          r'p_\mathrm{BZ}&=0.25\\'
                          r'p_\mathrm{dense}&=0.3'
                          r'\end{align*}\endgroup',
                          r'\begingroup\addtolength{\jot}{-2mm}\begin{align*}'
                          r'p_\mathrm{low}&=0.4\\'
                          r'p_\mathrm{BZ}&=0.5\\'
                          r'p_\mathrm{dense}&=0.6'
                          r'\end{align*}\endgroup',
                          r'\begingroup\addtolength{\jot}{-2mm}\begin{align*}'
                          r'p_\mathrm{low}&=0.6\\'
                          r'p_\mathrm{BZ}&=0.75\\'
                          r'p_\mathrm{dense}&=0.9'
                          r'\end{align*}\endgroup']

    """ Set axis labels and ticks """
    ax.set_ylabel(metric_name)
    ax.set_xticks(list(range(len(density_labels))))
    ax.set_xticklabels(density_labels)
    ax.legend()

    return fig