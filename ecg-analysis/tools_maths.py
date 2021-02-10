import math
import numpy as np
import pandas as pd
from typing import Union
from scipy import signal
from sklearn import preprocessing


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


def filter_butterworth(data: Union[np.ndarray, pd.DataFrame],
                       sample_freq: float = 500.0,
                       freq_filter: float = 40,
                       order: int = 2,
                       filter_type: str = 'low') -> Union[np.ndarray, pd.DataFrame]:
    """Filter data using Butterworth filter

    Filter a given set of data using a Butterworth filter, designed to have a specific passband for desired frequencies.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Data to filter
    sample_freq : int or float
        Sampling rate of data (Hz), default=500. If data passed as dataframe, the sample_freq will be calculated from
        the dataframe index.
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
    if isinstance(data, pd.DataFrame):
        dt = np.mean(np.diff(data.index))/1000
        sample_freq = 1/dt
    window = freq_filter/(sample_freq*0.5)

    [b, a] = signal.butter(order, window, filter_type)
    if isinstance(data, pd.DataFrame):
        data_filtered = pd.DataFrame(columns=data.columns, index=data.index)
        for col in data_filtered:
            data_filtered.loc[:, col] = signal.filtfilt(b, a, data[col])
        return data_filtered
    else:
        return signal.filtfilt(b, a, data)


def filter_savitzkygolay(data: pd.DataFrame,
                         window_length: int = 50,
                         order: int = 2,
                         deriv: int = 0,
                         delta: float = 1.0):
    """Filter EGM data using a Savitzky-Golay filter

    Filter a given set of data using a Savitzky-Golay filter, designed to smooth data using a convolution process
    fitting to a low-degree polynomial within a given window. Default values are either taken from scipy
    documentation (not all options are provided here), or adapted to match Hermans et al.

    Parameters
    ----------
    data : pd.DataFrame
        Data to filter
    window_length : float, optional
        The length of the filter window in milliseconds. When passed to the scipy filter, will be converted to a
        positive odd integer (i.e. the number of coefficients). Default=50
    order : int, optional
        The order of the polynomial used to fit the samples. polyorder must be less than window_length. Default=2
    deriv : int, optional
        The order of the derivative to compute. This must be a nonnegative integer. The default is 0, which means to
        filter the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied. This is only used if deriv > 0. Default=1.0

    Returns
    -------
    data : pd.DataFrame
        Output filtered data

    References
    ----------
    The development and validation of an easy to use automatic QT-interval algorithm
        Hermans BJM, Vink AS, Bennis FC, Filippini LH, Meijborg VMF, Wilde AAM, Pison L, Postema PG, Delhaas T
        PLoS ONE, 12(9), 1–14 (2017)
        https://doi.org/10.1371/journal.pone.0184352
    """
    i_window = np.where(data.index-data.index[0] > window_length)[0][0]
    if (i_window % 2) == 0:
        i_window += 1

    data_filtered = pd.DataFrame(index=data.index, columns=data.columns)
    for col in data_filtered:
        data_filtered.loc[:, col] = signal.savgol_filter(data[col], window_length=i_window, polyorder=order,
                                                         deriv=deriv, delta=delta)

    return data_filtered


def normalise_signal(data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
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

    if isinstance(data, np.ndarray):
        return np.divide(np.absolute(data), np.amax(np.absolute(data)))
    elif isinstance(data, pd.DataFrame):
        columns = data.keys()
        index = data.index
        minmax = preprocessing.MaxAbsScaler()
        data = minmax.fit_transform(data.values)
        return pd.DataFrame(data, columns=columns, index=index)
