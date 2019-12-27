import numpy as np
import math


def convert_time_to_index(qrs_start=None, qrs_end=None, t_start=0, t_end=200, dt=2):
    """
    Return indices of QRS start and end points. NB: indices returned match Matlab output

    Input parameters:
    -----------------

    qrs_start   None    Start time to convert to index
    qrs_end     None    End time to convert to index
    t_start     0       Start time of overall data
    t_end       200     End time of overall data
    dt          2       Interval between time points

    Output parameters:
    ------------------

    i_qrs_start         Index of start time
    i_qrs_end           Index of end time

    """
    if qrs_start is None:
        qrs_start = t_start
    if qrs_end is None:
        qrs_end = t_end
    x_val = np.array(range(t_start, t_end + dt, dt))
    i_qrs_start = np.where(x_val >= qrs_start)[0][0]
    i_qrs_end = np.where(x_val > qrs_end)[0][0]-1

    return i_qrs_start, i_qrs_end


def convert_index_to_time(idx, t_start=0, t_end=200, dt=2):
    """
    Return 'real' time for a given index

    Required input parameters:
    --------------------------

    idx     Index to convert

    Optional input parameters:
    --------------------------

    t_start     0       Start time for overall data
    t_end       200     End time for overall data
    dt          2       Interval between time points
    """
    x_val = np.array(range(t_start, t_end+dt, dt))
    return x_val[idx]


def write_colourmap_to_xml(start_data, end_data, start_highlight, end_highlight, opacity_data=1, opacity_highlight=1,
                           n_tags=20, colourmap='viridis', outfile='colourmap.xml'):
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
        pFile.write('<ColorMaps>\n'.format(name))
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


def asin2(x, y):
    """
    Function to return the inverse sin function across the range (-pi, pi], rather than (-pi/2, pi/2]

    Input parameters:
    -----------------

    x       x coordinate of the point in 2D space
    y       y coordinate of the point in 2D space

    Output parameters:
    ------------------

    theta   angle corresponding to point in 2D space in radial coordinates, within range (-pi, pi]
    """
    r = math.sqrt(x**2+y**2)
    if x >= 0:
        return math.asin(y/r)
    else:
        if y >= 0:
            return math.pi-math.asin(y/r)
        else:
            return -math.pi-math.asin(y/r)


def acos2(x, y):
    """
    Function to return the inverse cos function across the range (-pi, pi], rather than (0, pi]

    Input parameters:
    -----------------

    x       x coordinate of the point in 2D space
    y       y coordinate of the point in 2D space

    Output parameters:
    ------------------

    theta   angle corresponding to point in 2D space in radial coordinates, within range (-pi, pi]


    """
    r = math.sqrt(x**2+y**2)
    if y >= 0:
        return math.acos(x/r)
    else:
        return -math.acos(x/r)


def find_list_fraction(input_list, fraction=0.5, interpolate=True):
    """
    Find index corresponding to certain fractional length within a list, e.g. halfway along, a third along

    If only looking for an interval halfway along the list, uses a simpler method that is computationally faster

    Input parameters (required):
    ----------------------------

    input_list      List to find the fractional value of

    Input parameters (optional):
    ----------------------------

    fraction        0.5     Fraction of length of list to return the value of
    interpolate     True    If fraction does not precisely specify a particular entry in the list, whether to return the
                            values on either side, or whether to interpolate between the two values (with weighting
                            given to how close the fraction is to one value or the other)
    """

    if fraction == 0.5:
        middle = float(len(input_list)) / 2
        if middle % 2 != 0:
            return input_list[int(middle - .5)]
        else:
            if interpolate:
                return np.mean((input_list[int(middle)], input_list[int(middle - 1)]), axis=0)
            else:
                return input_list[int(middle - 1)], input_list[int(middle)]

    fraction_list = np.linspace(0, 1, len(input_list))
    fraction_bounds = 0.1
    fraction_idx = np.where((fraction_list >= fraction-fraction_bounds) &
                            (fraction_list <= fraction+fraction_bounds))[0]
    while len(fraction_idx) > 2:
        fraction_bounds /= 10
        fraction_idx = np.where((fraction_list >= fraction - fraction_bounds) &
                                (fraction_list <= fraction + fraction_bounds))[0]
        if len(fraction_idx) < 1:
            fraction_bounds *= 2
            fraction_idx = np.where((fraction_list >= fraction - fraction_bounds) &
                                    (fraction_list <= fraction + fraction_bounds))[0]

    if len(fraction_idx) == 1:
        return input_list[fraction_idx[0]]
    else:
        if interpolate:
            # Interpolate between two values, based on:
            # l = l_a * f_a + l_b * f_b
            # where l_a is value of list at fraction_idx[0] and l_b is value of list at fraction_idx[1]
            # f_a = (-1/(b-a))*(x-a) + 1
            # f_b = (1/(b-a))*(x-a)
            # where a=fraction_idx[0], b=fraction_idx[1], and x is the actually desired fraction
            a = fraction_list[fraction_idx[0]]
            b = fraction_list[fraction_idx[1]]
            gradient = 1/(b-a)
            f_a = -gradient*(fraction-a)+1
            f_b = gradient*(fraction-a)

            # Return different answers, depending on whether the input list is a list of lists or not
            if isinstance(input_list[0], list):
                return [i*f_a+j*f_b for i, j in zip(input_list[fraction_idx[0]], input_list[fraction_idx[1]])]
            else:
                return input_list[fraction_idx[0]]*f_a + input_list[fraction_idx[1]]*f_b
        else:
            return tuple(np.array(input_list)[fraction_idx])
