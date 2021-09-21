import numpy as np
import pandas as pd
import math
from math import sin, cos, acos, atan2
import warnings
from typing import Union, List, Tuple, Optional, Iterable

import tools_maths
import tools_python
import set_midwallFibrosis as smF


def get_vcg_from_ecg(ecgs: Union[List[pd.DataFrame], pd.DataFrame]) -> List[pd.DataFrame]:
    """Convert ECG data to vectorcardiogram (VCG) data using the Kors matrix method

    Parameters
    ----------
    ecgs : list of pd.DataFrame or pd.DataFrame
        List of ECG dataframe data, or ECG dataframe data directly, with dict keys corresponding to ECG outputs

    Returns
    -------
    vcgs: list of pd.DataFrame
        List of VCG output data

    References
    ----------
    Kors JA, van Herpen G, Sittig AC, van Bemmel JH.
        Reconstruction of the Frank vectorcardiogram from standard electrocardiographic leads: diagnostic comparison
        of different methods
        Eur Heart J. 1990 Dec;11(12):1083-92.
    """

    kors = np.array([[0.38, -0.07, 0.11],
                     [-0.07, 0.93, -0.23],
                     [-0.13, 0.06, -0.43],
                     [0.05, -0.02, -0.06],
                     [-0.01, -0.05, -0.14],
                     [0.14, 0.06, -0.20],
                     [0.06, -0.17, -0.11],
                     [0.54, 0.13, 0.31]])

    if isinstance(ecgs, pd.DataFrame):
        ecgs = [ecgs]

    vcgs = list()
    for ecg in ecgs:
        ecg_matrix = np.array([ecg['LI'], ecg['LII'], ecg['V1'], ecg['V2'], ecg['V3'],
                               ecg['V4'], ecg['V5'], ecg['V6']])
        vcg = pd.DataFrame(np.dot(ecg_matrix.transpose(), kors), index=ecg.index, columns=['x', 'y', 'z'])
        vcgs.append(vcg)

    return vcgs


def get_qrs_start_end(vcgs: Union[List[pd.DataFrame], pd.DataFrame],
                      velocity_offset: int = 2,
                      low_p: float = 40,
                      order: int = 2,
                      threshold_frac_start: float = 0.22,
                      threshold_frac_end: float = 0.54,
                      filter_sv: bool = True,
                      qrs_window: float = 180,
                      ecgs: Union[List[pd.DataFrame], pd.DataFrame, None] = None) -> Tuple[List[float], List[float],
                                                                                           List[float]]:
    """Calculate the extent of the VCG QRS complex on the basis of max derivative

    TODO: Check whether i_qrs_start variable is needed, or can be simplified using DataFrame function

    Calculate the start and end points, and hence duration, of the QRS complex of a list of VCGs. It does this by
    finding the time at which the spatial velocity of the VCG exceeds a threshold value (the start time), then searches
    backwards from the end of the VCG to find when this threshold is exceeded (the end time); the start and end
    thresholds do not necessarily have to be the same.

    Parameters
    ----------
    vcgs : list of pd.DataFrame or pd.DataFrame
        List of VCG data to get QRS start and end points for
    velocity_offset : int, optional
        Offset between values in VCG over which to calculate spatial velocity, i.e. 1 will use neighbouring values to
        calculate the gradient/velocity. Default=2
    low_p : float, optional
        Low frequency for bandpass filter, default=40
    order : int, optional
        Order for Butterworth filter, default=2
    threshold_frac_start : float, optional
        Fraction of maximum spatial velocity to trigger start of QRS detection, default=0.15
    threshold_frac_end : float, optional
        Fraction of maximum spatial velocity to trigger end of QRS detection, default=0.15
    filter_sv : bool, optional
        Whether or not to apply filtering to spatial velocity prior to finding the start/end points for the threshold
    qrs_window : float, optional
        Default size of 'window' in which to search for end of QRS complex, default=180ms
    ecgs : list of pd.DataFrame or pd.DataFrame, optional
        ECG data associated with VCG data. Only used if having trouble establishing QRS start, in which case will be
        used to plot ECG data to allow user to determine whether or not the QRS is occurring at the start of the
        simulation, or whether there is a more deep-seated issue with the data.

    Returns
    -------
    qrs_start : list of float
        List of start time of QRS complexes of provided VCGs
    qrs_end : list of float
        List of end time of QRS complex of provided VCGs
    qrs_duration : list of float
        List of duration of QRS complex of provided VCGs
    """

    if isinstance(vcgs, pd.DataFrame):
        vcgs = [vcgs]
    if ecgs is not None:
        if isinstance(ecgs, pd.DataFrame):
            ecgs = [ecgs]
        assert len(ecgs) == len(vcgs)
    assert 0 < threshold_frac_start < 1, "threshold_frac_start must be between 0 and 1"
    assert 0 < threshold_frac_end < 1, "threshold_frac_end must be between 0 and 1"

    sv = get_spatial_velocity(vcgs=vcgs, velocity_offset=velocity_offset, filter_sv=filter_sv, low_p=low_p, order=order)

    qrs_start = list()
    qrs_end = list()
    qrs_duration = list()
    for i_sim, sim_sv in enumerate(sv):
        # Determine threshold for QRS complex, then find start of QRS complex. Iteratively remove more of the plot if
        # the 'start' is found to be 0 (implies it is still getting confused by the preceding wave,
        # but inspection flag added to allow if just very early QRS).
        sim_sv_orig = sim_sv.copy()
        # noinspection PyArgumentList
        threshold_start = sim_sv.max().values * threshold_frac_start
        i_qrs_start = np.where(sim_sv > threshold_start)[0][0]
        while i_qrs_start == 0:
            sim_sv = sim_sv.iloc[1:, :]
            threshold_start = sim_sv.max().values * threshold_frac_start

            i_qrs_start = np.where(sim_sv > threshold_start)[0][0]
            if sim_sv.index[0] > 50:
                import matplotlib.pyplot as plt
                # Figure won't plot if using the Qt5Agg backend, for some reason (see
                # https://github.com/matplotlib/matplotlib/issues/9206 for discussion, but that says it's fixed).
                # Unable to change backend in any meaningful way to resolve this dispute, so forced to use a block on
                # the plt.show() command
                import ecg_plot as ep
                _ = ep.plot(ecgs[i_sim])
                fig, ax = plt.subplots(1, 1)
                ax.plot(sim_sv_orig)
                ax.set_xlabel('Time')
                ax.set_ylabel('Spatial Velocity')
                ax.axhline(sim_sv.max().values * threshold_frac_start,
                           label='Threshold={}'.format(threshold_frac_start))
                ax.legend()
                print("Hack applied here - need to close the figure window in order to continue with the program.")
                plt.show(block=True)
                qrs_early = ''
                while not (qrs_early.lower() == 'y' or qrs_early.lower() == 'n' or ',' in qrs_early):
                    qrs_early = input('More than 50ms of trace removed - do you wish to :'
                                      '\n\tset start to 0 (y), '
                                      '\n\tset values to NaN (n),'
                                      '\n\tspecify a cut-off and new threshold (ms to remove, threshold_value)?')
                if qrs_early.lower() == 'n':
                    warnings.warn('QRS unable to be calculated - setting to NaN.')
                    i_qrs_start = np.nan
                elif qrs_early.lower() == 'y':
                    sim_sv = sim_sv_orig.copy()
                    i_qrs_start = 0
                    break
                elif ',' in qrs_early:
                    qrs_early = qrs_early.split(',')
                    assert len(qrs_early) == 2
                    qrs_early = [float(qrs_early_temp) for qrs_early_temp in qrs_early]
                    assert sim_sv_orig.index[0] <= qrs_early[0] < sim_sv_orig.index[-1]
                    # noinspection PyArgumentList
                    assert sim_sv_orig.min().values <= qrs_early[1] <= sim_sv_orig.max().values
                    sim_sv = sim_sv_orig.loc[qrs_early[0]:, :]
                    i_qrs_start = np.where(sim_sv > qrs_early[1])[0][0]

        if np.isnan(i_qrs_start):
            qrs_start.append(np.nan)
            qrs_end.append(np.nan)
            qrs_duration.append(np.nan)
        else:
            qrs_start.append(sim_sv.index[i_qrs_start])

            # noinspection PyArgumentList
            threshold_end = sim_sv.max().values * threshold_frac_end
            try:
                # Set window for QRS complex, then find when threshold_end is exceeded (searching backwards from end
                # of window)
                sim_sv_temp = sim_sv.loc[qrs_start[-1]:qrs_start[-1]+qrs_window, :]
                qrs_end_temp = sim_sv_temp[sim_sv_temp > threshold_end]
                qrs_end_temp.dropna(axis=0, inplace=True)
                qrs_end.append(qrs_end_temp.index[-1])
                # i_qrs_end = len(sim_sv) - (np.where(np.flip(sim_sv.values) > threshold_end)[0][0] - 1)
                # qrs_end.append(sim_sv.index[i_qrs_end])
            except IndexError:
                # Figure won't plot if using the Qt5Agg backend, for some reason (see
                # https://github.com/matplotlib/matplotlib/issues/9206 for discussion, but that says it's fixed).
                # Unable to change backend in any meaningful way to resolve this dispute, so forced to use a block on
                # the plt.show() command
                import matplotlib.pyplot as plt
                import ecg_plot as ep
                _ = ep.plot(ecgs[i_sim])
                fig, ax = plt.subplots(1, 1)
                ax.plot(sim_sv)
                ax.set_xlabel('Time')
                ax.set_ylabel('Spatial Velocity')
                ax.axhline(threshold_start, color='k', linestyle='--')
                ax.axhline(threshold_end, color='k', linestyle='-')
                ax.axvline(sim_sv.index[i_qrs_start], color='k', linestyle='--')
                print("Hack applied here - need to close the figure window in order to continue with the program.")
                plt.show(block=True)

                qrs_late = ''
                while not (qrs_late.lower() == 'n'):
                    qrs_late = input('Struggling to find QRS end - do you wish to:'
                                     '\n\tset values to NaN (n),')
                if qrs_late.lower() == 'n':
                    warnings.warn('QRS unable to be calculated - setting to NaN.')
                    qrs_end.append(np.nan)

            qrs_duration.append(qrs_end[-1] - qrs_start[-1])

            if not np.isnan(qrs_end[-1]):
                assert qrs_start[-1] < qrs_end[-1], "qrs_start {} >= qrs_end[-1] {}".format(qrs_start[-1], qrs_end[-1])
                assert qrs_end[-1] <= sim_sv_orig.index[-1], "i_qrs_end >= len(sv)"

    return qrs_start, qrs_end, qrs_duration


def get_spatial_velocity(vcgs: Union[List[pd.DataFrame], pd.DataFrame],
                         velocity_offset: int = 2,
                         filter_sv: bool = True,
                         low_p: float = 40,
                         order: int = 2) -> List[pd.DataFrame]:
    """Calculate spatial velocity

    Calculate the spatial velocity of a VCG, in terms of calculating the gradient of the VCG in each of its x,
    y and z components, before combining these components in a Euclidian norm. Will then find the point at which the
    spatial velocity exceeds a threshold value, and the point at which it declines below another threshold value.

    Parameters
    ----------
    vcgs : list of pd.DataFrame or pd.DataFrame
        VCG data to analyse
    velocity_offset : int, optional
        Offset between values in VCG over which to calculate spatial velocity, i.e. 1 will use neighbouring values to
        calculate the gradient/velocity. Default=2
    filter_sv : bool, optional
        Whether or not to apply filtering to spatial velocity, default=True
    low_p : float, optional
        Low frequency for bandpass filter, default=40
    order : int, optional
        Order for Butterworth filter, default=2

    Returns
    -------
    sv : list of pd.DataFrame
        Spatial velocity data, filtered according to input parameters

    Notes
    -----
    Calculation of spatial velocity based on [1]_, [2]_, [3]_

    References
    ----------
    .. [1] Kors JA, van Herpen G, "Methodology of QT-interval measurement in the modular ECG analysis system (MEANS)"
           Ann Noninvasive Electrocardiol. 2009 Jan;14 Suppl 1:S48-53. doi: 10.1111/j.1542-474X.2008.00261.x.
    .. [2] Xue JQ, "Robust QT Interval Estimation—From Algorithm to Validation"
           Ann Noninvasive Electrocardiol. 2009 Jan;14 Suppl 1:S35-41. doi: 10.1111/j.1542-474X.2008.00264.x.
    .. [3] Sörnmo L, "A model-based approach to QRS delineation"
           Comput Biomed Res. 1987 Dec;20(6):526-42.
    """
    if isinstance(vcgs, pd.DataFrame):
        vcgs = [vcgs]

    sv = list()
    for vcg in vcgs:
        # Compute spatial velocity of VCG
        dvcg = np.divide(vcg.values[velocity_offset:] - vcg.values[:-velocity_offset],
                         vcg.index.values[velocity_offset:, None]-vcg.index.values[:-velocity_offset, None])

        # Calculates Euclidean distance based on spatial velocity in x, y and z directions, i.e. will calculate
        # sqrt(x^2+y^2+z^2) to get total spatial velocity

        # Calculate the time appropriate to the spatial velocity - cuts time off from each side equally, with a bias
        # towards cutting the tail values. If len(vcg)=n, then:
        # If velocity_offset=1 => len(sv)=n-1 => time_sv=time_vcg[:-1]
        # If velocity_offset=2 => len(sv)=n-2 => time_sv=time_vcg[1:-1]
        pre_cut = math.floor(velocity_offset/2)
        end_cut = velocity_offset-pre_cut

        sim_sv = pd.DataFrame(np.linalg.norm(dvcg, axis=1), index=vcg.index.values[pre_cut:-end_cut], columns=['sv'])
        if filter_sv:
            sv.append(tools_maths.filter_butterworth(sim_sv, freq_filter=low_p, order=order))
        else:
            sv.append(sim_sv)
    return sv


def get_vcg_area(vcgs: Union[List[pd.DataFrame], pd.DataFrame],
                 limits_start: Optional[List[float]] = None,
                 limits_end: Optional[List[float]] = None,
                 method: str = 'pythag',
                 matlab_match: bool = False) -> List[float]:
    """Calculate area under VCG curve for a given section (e.g. QRS complex).

    Calculate the area under the VCG between two intervals (usually QRS start and QRS end). This is calculated in two
    ways: a 'Pythagorean' method, wherein the area under each of the VCG(x), VCG(y) and VCG(z) curves are calculated,
    then combined in a Euclidean norm, or a '3D' method, wherein the area of the arc traced in 3D space between
    successive timepoints is calculated, then summed.

    Parameters
    ----------
    vcgs : pd.DataFrame or list of pd.DataFrame
        VCG data from which to get area
    limits_start : list of float, optional
        Start times (NOT INDICES) for where to calculate area under curve from, default=0
    limits_end : list of float, optional
        End times (NOT INDICES) for where to calculate are under curve until, default=end
    method : {'pythag', '3d'}, optional
        Which method to use to calculate the area under the VCG curve, default='pythag'
    matlab_match : bool, optional
        Whether to alter the calculation for start and end indices to match the original Matlab output, from which this
        module is based, default=False

    Returns
    -------
    qrs_area_3d : list of float
        Values for the area under the curve (as defined by the 3D method) between the provided limits for each of the
        VCGs
    qrs_area_pythag : list of float
        Values for the area under the curve (as defined by the Pythagorean method) between the provided limits for each
        of the VCGs
    qrs_area_components : list of list of float
        Areas under the individual x, y, z curves of the VCG, for each of the supplied VCGs
    """

    assert method in ['pythag', '3d'], "Unsuitable method requested"

    if isinstance(vcgs, pd.DataFrame):
        vcgs = [vcgs]

    if limits_start is None:
        limits_start = [0 for _ in range(len(vcgs))]
    if limits_end is None:
        limits_end = [vcg.index[-1] for vcg in vcgs]
    for limit_start, limit_end in zip(limits_start, limits_end):
        assert limit_start < limit_end, "limit_start >= limit_end"

    vcg_areas = list()
    for vcg, limit_start, limit_end in zip(vcgs, limits_start, limits_end):
        # Recalculate indices for start and end points of QRS, and extract relevant data
        i_limit_start = np.where(vcg.index == limit_start)[0][0]
        i_limit_end = np.where(vcg.index == limit_end)[0][0]
        if matlab_match:
            vcg_limited = vcg.iloc[i_limit_start - 1:i_limit_end + 1]
        else:
            vcg_limited = vcg.iloc[i_limit_start:i_limit_end + 1]

        if method == 'pythag':
            # Calculate area under x,y,z curves by trapezium rule, then combine
            dt = np.mean(np.diff(vcg_limited.index))
            qrs_area_temp = np.trapz(vcg_limited, dx=float(dt), axis=0)
            vcg_areas.append(np.linalg.norm(qrs_area_temp))
        elif method == '3d':
            # Calculate the area under the curve in 3d space wrt to the origin.
            sim_triangles = np.array([(i, j, (0, 0, 0)) for i, j in zip(vcg_limited[:-1], vcg_limited[1:])])
            vcg_areas.append(sum([smF.simplex_volume(vertices=sim_triangle) for sim_triangle in sim_triangles]))
        else:
            raise Exception("Improper method executed")

    return vcg_areas


def get_azimuth_elevation(vcgs: Union[List[pd.DataFrame], pd.DataFrame],
                          t_start: Optional[List[float]] = None,
                          t_end: Optional[List[float]] = None) -> Tuple[List[Iterable[float]], List[Iterable[float]]]:
    """Calculate azimuth and elevation angles for a specified section of the VCG.

    Will calculate the azimuth and elevation angles for the VCG at each recorded point, potentially within specified
    limits (e.g. start/end of QRS)

    Parameters
    ----------
    vcgs : pd.DataFrame or list of pd.DataFrame
        VCG data to calculate
    t_start : list of float, optional
        Start time from which to calculate the angles, default=0
    t_end : list of float, optional
        End time until which to calculate the angles, default=end

    Returns
    -------
    azimuth : list of list of float
        List (one entry for each passed VCG) of azimuth angles (in radians) for the dipole for every time point during
        the specified range
    elevation : list of list of float
        List (one entry for each passed VCG) of elevation angles (in radians) for the dipole for every time point during
        the specified range
    """
    if isinstance(vcgs, pd.DataFrame):
        vcgs = [vcgs]
    assert len(vcgs) == len(t_start)
    assert len(vcgs) == len(t_end)

    azimuth = list()
    elevation = list()
    for (vcg, sim_t_start, sim_t_end) in zip(vcgs, t_start, t_end):
        theta, phi, _ = get_single_vcg_azimuth_elevation(vcg, sim_t_start, sim_t_end, weighted=False)

        azimuth.append(theta)
        elevation.append(phi)

    return azimuth, elevation


def get_weighted_dipole_angles(vcgs: Union[List[pd.DataFrame], pd.DataFrame],
                               t_start: Optional[List[float]] = None,
                               t_end: Optional[List[float]] = None) \
        -> Tuple[List[float], List[float], List[List[float]]]:
    """
    Calculate metrics relating to the angles of the weighted dipole of the VCG. Usually used with QRS limits.

    Calculates the weighted averages of both the azimuth and the elevation (inclination above the xy-plane) for a
    given section of the VCG. Based on these weighted averages of the angles, the unit weighted dipole for that
    section of the VCG is returned as well.

    Parameters
    ----------
    vcgs : pd.DataFrame or list of pd.DataFrame
        VCG data to calculate
    t_start : list of float, optional
        Start time from which to calculate the angles, default=0
    t_end : list of float, optional
        End time until which to calculate the angles, default=end

    Returns
    -------
    waa : list of float
        List of Weighted Average Azimuth angles (in radians) for each given VCG
    wae : list of float
        List of Weighted Average Elevation (above xy-plane) angles (in radians) for each given VCG
    uwd : list of list of float
        x, y, z coordinates for the unit mean weighted dipole for the given (section of) VCGs
    """

    if isinstance(vcgs, pd.DataFrame):
        vcgs = [vcgs]
    assert len(vcgs) == len(t_start)
    assert len(vcgs) == len(t_end)

    weighted_average_azimuth = list()
    weighted_average_elev = list()
    unit_weighted_dipole = list()
    for (vcg, sim_t_start, sim_t_end) in zip(vcgs, t_start, t_end):
        # Calculate dipole at all points
        theta, phi, dipole_magnitude = get_single_vcg_azimuth_elevation(vcg, sim_t_start, sim_t_end, weighted=True)

        wae = sum(phi) / sum(dipole_magnitude)
        waa = sum(theta) / sum(dipole_magnitude)

        weighted_average_elev.append(wae)
        weighted_average_azimuth.append(waa)
        unit_weighted_dipole.append([sin(wae) * cos(waa), cos(wae), sin(wae) * sin(waa)])

    return weighted_average_azimuth, weighted_average_elev, unit_weighted_dipole


def get_single_vcg_azimuth_elevation(vcg: pd.DataFrame,
                                     t_start: float,
                                     t_end: float,
                                     weighted: bool = True) \
        -> Tuple[List[float], List[float], np.ndarray]:
    """Get the azimuth and elevation data for a single VCG trace, along with the average dipole magnitude.

    Returns the azimuth and elevation angles for a single given VCG trace. Can analyse only a segment of the
    VCG if required, and can weight the angles according to the dipole magnitude. Primarily designed as a helper
    function for get_azimuth_elevation and get_weighted_dipole_angles.

    Parameters
    ----------
    vcg : pd.DataFrame
        VCG data to calculate
    t_start : float
        Start time from which to calculate the angles
    t_end : float
        End time until which to calculate the angles
    weighted : bool, optional
        Whether or not to weight the returned angles by the magnitude of the dipole at the same moment, default=True

    Returns
    -------
    theta : list of float
        List of the azimuth angles for the VCG dipole, potentially weighted according to the dipole magnitude at the
        associated time
    phi : list of float
        List of the elevation above xy-plane angles for the VCG dipole, potentially weighted according to the dipole
        magnitude at the associated time
    dipole_magnitude : np.ndarray
        Array containing the dipole magnitude at all points throughout the VCG
    """
    sim_vcg = vcg.loc[t_start:t_end]
    dipole_magnitude = np.linalg.norm(sim_vcg, axis=1)

    # Calculate azimuth (theta, ranges (-pi,pi]) and elevation (phi, ranges (0, pi]), potentially weighted or not.
    if weighted:
        theta = [atan2(sim_vcg_t[2], sim_vcg_t[0])*dipole_magnitude_t for (sim_vcg_t, dipole_magnitude_t) in
                 zip(sim_vcg, dipole_magnitude)]
        phi = [acos(sim_vcg_t[1]/dipole_magnitude_t)*dipole_magnitude_t for (sim_vcg_t, dipole_magnitude_t) in
               zip(sim_vcg, dipole_magnitude)]
    else:
        theta = [atan2(sim_vcg_t[2], sim_vcg_t[0]) for (sim_vcg_t, dipole_magnitude_t) in
                 zip(sim_vcg, dipole_magnitude)]
        phi = [acos(sim_vcg_t[1]/dipole_magnitude_t) for (sim_vcg_t, dipole_magnitude_t) in
               zip(sim_vcg, dipole_magnitude)]

    return theta, phi, dipole_magnitude


def get_dipole_magnitudes(vcgs: Union[List[pd.DataFrame], pd.DataFrame],
                          t_start: Union[float, List[float]] = 0,
                          t_end: Union[float, List[float]] = -1) \
        -> Tuple[List[np.ndarray], List[float], List[float], List[List[float]], List]:
    """Calculates metrics relating to the magnitude of the weighted dipole of the VCG

    Returns the mean weighted dipole, maximum dipole magnitude,(x,y.z) components of the maximum dipole and the time
    at which the maximum dipole occurs

    Parameters
    ----------
    vcgs : pd.DataFrame or list of pd.DataFrame
        VCG data to calculate
    t_start : list of float, optional
        Start time from which to calculate the magnitude, default=0 (for any other value to be recognisable,
        time variable must be given)
    t_end : list of float, optional
        End time until which to calculate the magnitudes, default=end (for any other value to be recognisable,
        time variable must be given)

    Returns
    -------
    dipole_magnitude : list of np.ndarray
        Magnitude time courses for each VCG
    weighted_magnitude : list of float
        Mean magnitude of the VCG
    max_dipole_magnitude : list of float
        Maximum magnitude of the VCG
    max_dipole_components : list of list of float
        x, y, z components of the dipole at is maximum value
    max_dipole_time : list of float
        Time at which the maximum magnitude of the VCG occurs
    """

    # Check input arguments are in the correct format
    if isinstance(vcgs, pd.DataFrame):
        vcgs = [vcgs]

    t_start = tools_python.convert_input_to_list(t_start, n_list=len(vcgs), default_entry=t_start)
    t_end = tools_python.convert_input_to_list(t_end, n_list=len(vcgs), default_entry=t_end)

    dipole_magnitude = list()
    weighted_magnitude = list()
    max_dipole_magnitude = list()
    max_dipole_components = list()
    max_dipole_time = list()
    for (vcg, sim_t_start, sim_t_end) in zip(vcgs, t_start, t_end):
        # Calculate dipole at all points
        sim_vcg_qrs = vcg.loc[sim_t_start:sim_t_end]
        sim_dipole_magnitude = np.linalg.norm(sim_vcg_qrs, axis=1)

        dipole_magnitude.append(sim_dipole_magnitude)
        weighted_magnitude.append(sum(sim_dipole_magnitude) / len(sim_vcg_qrs))
        max_dipole_magnitude.append(max(sim_dipole_magnitude))
        i_max = np.where(sim_dipole_magnitude == max(sim_dipole_magnitude))
        assert len(i_max) == 1
        max_dipole_components.append(sim_vcg_qrs[i_max[0]])
        max_dipole_time.append(sim_vcg_qrs.index[i_max])

    return dipole_magnitude, weighted_magnitude, max_dipole_magnitude, max_dipole_components, max_dipole_time


def calculate_delta_dipole_angle(azimuth1: List[float],
                                 elevation1: List[float],
                                 azimuth2: List[float],
                                 elevation2: List[float],
                                 convert_to_degrees: bool = False) -> List[float]:
    """
    Calculates the angular difference between two VCGs based on difference in azimuthal and elevation angles.

    Useful for calculating difference between weighted averages.

    Parameters
    ----------
    azimuth1 : list of float
        Azimuth angles for the first dipole
    elevation1 : list of float
        Elevation angles for the first dipole
    azimuth2 : list of float
        Azimuth angles for the second dipole
    elevation2 : list of float
        Elevation angles for the second dipole
    convert_to_degrees : bool, optional
        Whether to convert the angle from radians to degrees, default=False

    Returns
    -------
    dt : list of float
        List of angles between a series of dipoles, either in radians (default) or degrees depending on input argument
    """

    dt = list()
    for az1, ele1, az2, ele2 in zip(azimuth1, elevation1, azimuth2, elevation2):
        dot_product = (sin(ele1) * cos(az1) * sin(ele2) * cos(az2)) + \
                      (cos(ele1) * cos(ele2)) + \
                      (sin(ele1) * sin(az1) * sin(ele2) * sin(az2))
        if abs(dot_product) > 1:
            warnings.warn("abs(dot_product) > 1: dot_product = {}".format(dot_product))
            assert abs(dot_product)-1 < 0.000001
            if dot_product > 1:
                dot_product = 1
            else:
                dot_product = -1

        dt.append(acos(dot_product))

    if convert_to_degrees:
        return [dt_i*180/math.pi for dt_i in dt]
    else:
        return dt


def compare_dipole_angles(vcg1: pd.DataFrame,
                          vcg2: pd.DataFrame,
                          t_start1: float = 0,
                          t_end1: Optional[float] = None,
                          t_start2: float = 0,
                          t_end2: Optional[float] = None,
                          n_compare: int = 10,
                          convert_to_degrees: bool = False,
                          matlab_match: bool = False) -> List[float]:
    """
    Calculates the angular differences between two VCGs at multiple points during their evolution

    To compensate for the fact that the two VCG traces may not be of the same length, the comparison does not occur
    at every moment of the VCG; rather, the dipoles are calculated for certain fractional points during the VCG.

    Parameters
    ----------
    vcg1 : pd.DataFrame
        First VCG trace to consider
    vcg2 : pd.DataFrame
        Second VCG trace to consider
    t_start1 : float, optional
        Time from which to consider the data from the first VCG trace, default=0
    t_end1 : float, optional
        Time until which to consider the data from the first VCG trace, default=end
    t_start2 : float, optional
        Time from which to consider the data from the second VCG trace, default=0
    t_end2 : float, optional
        Time until which to consider the data from the second VCG trace, default=end
    n_compare : int, optional
        Number of points during the VCGs at which to calculate the dipole angle. If set to -1, will calculate at
        every point during the VCG, but requires VCG traces to be the same length, default=10
    convert_to_degrees : bool, optional
        Whether to convert the angles from radians to degrees, default=False
    matlab_match : bool, optional
        Whether to extract the data segment to match Matlab output or to use simpler Python, default=False

    Returns
    -------
    dt : list of float
        Angle between two given VCGs at n points during the VCG, where n is given as input
    """

    # Calculate indices for the two VCG traces that correspond to the time points to be compared
    i_start1, i_end1 = tools_python.deprecated_convert_time_to_index(t_start1, t_end1)
    i_start2, i_end2 = tools_python.deprecated_convert_time_to_index(t_start2, t_end2)

    if n_compare == -1:
        assert len(vcg1) == len(vcg2)
        idx_list1 = range(len(vcg1))
        idx_list2 = range(len(vcg2))
    else:
        if matlab_match:
            i_start1 -= 1
            i_end1 -= 1
            i_start2 -= 1
            i_end2 -= 1
            idx_list1 = [int(round(i_start1 + i*(i_end1-i_start1) / 10)) for i in range(1, n_compare+1)]
            idx_list2 = [int(round(i_start2 + i*(i_end2-i_start2) / 10)) for i in range(1, n_compare+1)]
        else:
            idx_list1 = [int(round(i)) for i in np.linspace(start=i_start1, stop=i_end1, num=n_compare)]
            idx_list2 = [int(round(i)) for i in np.linspace(start=i_start2, stop=i_end2, num=n_compare)]

    # Calculate the dot product and magnitudes of vectors. If the fraction of the two is slightly greater than 1 or less
    # than -1, give a warning and correct accordingly.
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
        return [dt_i*180/math.pi for dt_i in dt]
    else:
        return dt
