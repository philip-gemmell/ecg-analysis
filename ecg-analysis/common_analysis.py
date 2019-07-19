from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colours
import matplotlib.cm as cm
import shelve


def filter_egm(egm, sample_freq=2.0, freq_filter=40, order=2, filter_type='low'):
    """ Filters EGM data (low pass) """

    """ egm             data to filter
        sample_rate     sampling rate of data
        freq_filter     cut-off frequency for filter
        order           order of the Butterworth filter
        filter_type     type of filter ('low', 'high', 'band')
    """

    # Define filter window (expressed as a fraction of the Nyquist frequency, which is half the sampling rate)
    window = freq_filter/(sample_freq*0.5)

    [b, a] = signal.butter(order, window, filter_type)
    filter_out = signal.filtfilt(b, a, egm)

    return filter_out


def convert_time_to_index(qrs_start=None, qrs_end=None, t_start=0, t_end=200, dt=2, matlab_match=False):
    """ Returns indices of QRS start and end points. NB: indices returned match Matlab output """
    if qrs_start is None:
        qrs_start = t_start
    if qrs_end is None:
        qrs_end = t_end
    x_val = np.array(range(t_start, t_end + dt, dt))
    i_qrs_start = np.where(x_val >= qrs_start)[0][0]
    i_qrs_end = np.where(x_val > qrs_end)[0][0]-1

    return i_qrs_start, i_qrs_end


def convert_index_to_time(idx, t_start=0, t_end=200, dt=2):
    """ Returns 'real' time for a given index """
    x_val = np.array(range(t_start, t_end+dt, dt))
    return x_val[idx]


def get_plot_colours(n):
    """ Returns colour values to be used when plotting """

    if n < 11:
        # cmap = plt.get_cmap('tab10')
        cmap = cm.get_cmap('tab10')
        return [cmap(i) for i in np.linspace(0, 1, 10)]
    else:
        # cmap = plt.get_cmap('viridis', n)
        cmap = cm.get_cmap('viridis', n)
        return [cmap(i) for i in np.linspace(0, 1, n)]

    # values = range(n)
    # c_norm = mpl_colours.Normalize(vmin=0, vmax=values[-1])
    # scalar_cmap = cm.ScalarMappable(norm=c_norm, cmap=cmap)
    #
    # return [scalar_cmap.to_rgba(values[i]) for i in range(n)]


def save_workspace(shelf_name):
    my_shelf = shelve.open(shelf_name, 'n')   # 'n' for new

    for key in dir():
        try:
            my_shelf[key] = globals()[key]
        except (TypeError, KeyError):
            # __builtins__, my_shelf, and imported modules can not be shelved.
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()

    return None


def load_workspace(shelf_name):
    my_shelf = shelve.open(shelf_name)
    for key in my_shelf:
        globals()[key] = my_shelf[key]
    my_shelf.close()

    return None
