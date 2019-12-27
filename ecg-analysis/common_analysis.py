import numpy as np
import matplotlib.cm as cm
from scipy import signal


def filter_egm(egm, sample_freq=500, freq_filter=40, order=2, filter_type='low'):
    """
    Filter EGM data (low pass)

    Filter a given set of EGM data using a Butterworth filter, designed to have a specific passband for desired
    frequencies.

    Input parameters (required):
    ----------------------------

    egm                     data to filter

    Input parameters (optional):
    ----------------------------

    sample_rate     500     sampling rate of data (Hz)
    freq_filter     40      cut-off frequency for filter
    order           2       order of the Butterworth filter
    filter_type     'low'   type of filter ('low', 'high', 'band')

    Output parameters:
    ------------------
    filter_out              Output filtered data
    """

    # Define filter window (expressed as a fraction of the Nyquist frequency, which is half the sampling rate)
    window = freq_filter/(sample_freq*0.5)

    [b, a] = signal.butter(order, window, filter_type)
    filter_out = signal.filtfilt(b, a, egm)

    return filter_out


def get_plot_colours(n=10, colourmap=None):
    """
    Return iterable list of RGB colour values that can be used for custom plotting functions

    Returns a list of RGB colours values, potentially according to a specified colourmap. If n is low enough, will use
    the custom 'tab10' colourmap by default, which will use alternating colours as much as possible to maximise
    visibility. If n is too big, then the default setting is 'viridis', which should provide a gradation of colour from
    first to last.

    Input parameters:
    -----------------

    n           10      Number of distinct colours required
    colourmap   None    Matplotlib colourmap to base the end result on

    Output parameters:
    ------------------

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


def recursive_len(item):
    """ Return the total number of elements with a potentially nested list """

    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1
