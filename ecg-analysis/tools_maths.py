import math
import numpy as np
from scipy import signal


def asin2(x: float, y: float) -> float:
    """Function to return the inverse sin function across the range (-pi, pi], rather than (-pi/2, pi/2]

    Parameters
    ----------
    x : float
        x coordinate of the point in 2D space
    y : float
        y coordinate of the point in 2D space

    Returns
    -------
    theta : float
        Angle corresponding to point in 2D space in radial coordinates, within range (-pi, pi]
    """
    r = math.sqrt(x**2+y**2)
    if x >= 0:
        return math.asin(y/r)
    else:
        if y >= 0:
            return math.pi-math.asin(y/r)
        else:
            return -math.pi-math.asin(y/r)


def acos2(x: float, y: float) -> float:
    """Function to return the inverse cos function across the range (-pi, pi], rather than (0, pi]

    Parameters
    ----------
    x : float
        x coordinate of the point in 2D space
    y : float
        y coordinate of the point in 2D space

    Returns
    -------
    theta : float
        Angle corresponding to point in 2D space in radial coordinates, within range (-pi, pi]
    """
    r = math.sqrt(x**2+y**2)
    if y >= 0:
        return math.acos(x/r)
    else:
        return -math.acos(x/r)


def filter_egm(egm: np.ndarray,
               sample_freq: float = 500,
               freq_filter: float = 40,
               order: int = 2,
               filter_type: str = 'low') -> np.ndarray:
    """
    Filter EGM data (low pass)

    Filter a given set of EGM data using a Butterworth filter, designed to have a specific passband for desired
    frequencies.

    Parameters
    ----------
    egm : list
        Data to filter
    sample_freq : int or float
        Sampling rate of data (Hz), default=500
    freq_filter : int or float
        Cut-off frequency for filter, default=40
    order : int
        Order of the Butterworth filter, default=2
    filter_type : {'low', 'high', 'band'}
        Type of filter to use, default='low'

    Returns
    -------
    filter_out : np.ndarray
        Output filtered data
    """

    # Define filter window (expressed as a fraction of the Nyquist frequency, which is half the sampling rate)
    window = freq_filter/(sample_freq*0.5)

    [b, a] = signal.butter(order, window, filter_type)
    filter_out = signal.filtfilt(b, a, egm)

    return filter_out


def normalise_signal(data: np.ndarray) -> np.ndarray:
    """Returns a normalised signal, such that the maximum value in the signal is 1, or the minimum is -1

    Parameters
    ----------
    data : np.ndarray
        Signal to be normalised

    Returns
    -------
    normalised_data : np.ndarray
        Normalised signal
    """

    return np.divide(np.absolute(data), np.amax(np.absolute(data)))
