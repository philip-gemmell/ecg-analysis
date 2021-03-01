import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional


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
        cmap = cm.get_cmap(colourmap, lut=10)
        return [cmap(i) for i in range(n)]
    else:
        cmap = cm.get_cmap(colourmap, lut=n)
        return [cmap(i) for i in range(n)]


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
        basic_linestyles = ['-', '--', '-.', ':']
        return [basic_linestyles[i] for i in range(n)]
    elif n < 15:
        lines = list()
        lines.append('-')
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


def write_colourmap_to_xml(start_data: float,
                           end_data: float,
                           start_highlight: float,
                           end_highlight: float,
                           opacity_data: float = 1,
                           opacity_highlight: float = 1,
                           n_tags: int = 20,
                           colourmap: str = 'viridis',
                           outfile: str = 'colourmap.xml') -> None:
    """
    Create a Paraview friendly colourmap useful for highlighting a particular range

    Creates a colourmap that is entirely gray, save for a specified region of interest that will vary according to the
    specified colourmap

    Input parameters (required):
    ----------------------------
    start_data                      start value for overall data (can't just use data for region of interest -
                                    Paraview will scale)
    end_data                        end value for overall data
    start_highlight                 start value for region of interest
    end_highlight                   end value for region of interest

    Input parameters (optional):
    ----------------------------

    opacity_data        1.0                 overall opacity to use for all data
    opacity_highlight   1.0                 opacity for region of interest
    colourmap           'viridis'           colourmap to use
    outfile             'colourmap.xml'     filename to save .xml file under

    Output parameters:
    ------------------
    None
    """

    cmap = get_plot_colours(n_tags, colourmap=colourmap)

    # Get values for x, depending on start and end values
    x_offset = 0.2      # Value to provide safespace round x values
    x_maintain = 0.01   # Value to maintain safespace round n_tag values
    cmap_x_data = np.linspace(start_data, end_data, 20)
    cmap_x_data = np.delete(cmap_x_data, np.where(np.logical_and(cmap_x_data > start_highlight-x_offset,
                                                                 cmap_x_data < end_highlight+x_offset)))
    cmap_x_highlight = np.linspace(start_highlight-x_offset, end_highlight+x_offset, n_tags)

    # Extract colourmap name from given value for outfile
    if outfile.endswith('.xml'):
        name = outfile[:-4]
    else:
        name = outfile[:]
        outfile = outfile+'.xml'

    # Write to file
    with open(outfile, 'w') as pFile:
        # pFile.write('<ColorMaps>\n'.format(name))
        pFile.write('\t<ColorMap name="{}" space="RGB">\n'.format(name))

        # Write non-highlighted data values
        for x in cmap_x_data:
            pFile.write('\t\t<Point x="{}" o="{}" r="0.5" g="0.5" b="0.5"/>\n'.format(x, opacity_data))
        pFile.write('\t\t<Point x="{}" o="{}" r="0.5" g="0.5" b="0.5"/>\n'.format(start_highlight-3*x_offset,
                                                                                  opacity_data))
        pFile.write('\t\t<Point x="{}" o="{}" r="0.5" g="0.5" b="0.5"/>\n'.format(end_highlight+3*x_offset,
                                                                                  opacity_data))

        # Write highlighted data values
        for (rgb, x) in zip(cmap, cmap_x_highlight):
            pFile.write('\t\t<Point x="{}" o="{}" r="{}" g="{}" b="{}"/>\n'.format(x-x_maintain, opacity_highlight,
                                                                                   rgb[0], rgb[1], rgb[2]))
            pFile.write('\t\t<Point x="{}" o="{}" r="{}" g="{}" b="{}"/>\n'.format(x, opacity_highlight,
                                                                                   rgb[0], rgb[1], rgb[2]))
            pFile.write('\t\t<Point x="{}" o="{}" r="{}" g="{}" b="{}"/>\n'.format(x+x_maintain, opacity_highlight,
                                                                                   rgb[0], rgb[1], rgb[2]))

        pFile.write('\t</ColorMap>\n')
        pFile.write('</ColorMaps>')

    return None


def set_axis_limits(data: List[np.ndarray],
                    ax,
                    unit_min: bool = True,
                    axis_limits: Optional[Union[List[float], float]] = None,
                    pad_percent: float = 0.01) -> None:
    """Set axis limits (not automatic for line collections, so needs to be done manually)

    Parameters
    ----------
    data : list of np.ndarray
        Data that has been plotted
    ax
        Handles to the axes that need to be adjusted
    unit_min : bool, optional
        Whether to have the axes set to, as a minimum, unit length
    axis_limits : list of float or float, optional
        Min/max values for axes, either as one value (i.e. min=-max), or two separate values. Same axis limits will
        be applied to all dimensions
    pad_percent : float, optional
        Percentage 'padding' to add to the ranges, to try and ensure that the edges of linewidths are not cut off,
        default=0.01
    """
    assert 0 < pad_percent < 0.1, "pad_percent is set to 'unusual' values..."

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
    pad_value = (ax_max-ax_min)*pad_percent
    ax.set_xlim(ax_min-pad_value, ax_max+pad_value)
    ax.set_ylim(ax_min-pad_value, ax_max+pad_value)
    if len(data) == 3:
        ax.set_zlim(ax_min, ax_max)
    ax.set_aspect('equal', adjustable='box')
    return None


def add_colourbar(limits: List[float],
                  colourmap: str,
                  n_elements: int) -> None:
    """Add arbitrary colourbar to a figure, for instances when an automatic colorbar isn't available

    Parameters
    ----------
    limits : list of float
        Numerical limits to apply
    colourmap : str
        Colourmap to be used
    n_elements : int
        Number of entries to be made in the colourmap index

    Notes
    -----
    This is useful for instances suchs as when LineCollections are used to plot line that changes colour during the
    plotting process, as LineCollections do not enable an automatic colorbar to be added to the plot. This function
    adds a dummy colorbar to replace that.
    """

    cmap = plt.get_cmap(colourmap, n_elements)
    # noinspection PyUnresolvedReferences
    norm = mpl.colors.Normalize(vmin=limits[0], vmax=limits[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(np.ndarray([]))
    plt.colorbar(sm)
    return None
