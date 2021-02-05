import numpy as np
import math
from math import sin, cos, acos, atan2
import warnings
from typing import Union, List, Tuple, Optional, Iterable

import common_analysis
import set_midwallFibrosis as smF


def convert_ecg_to_vcg(ecg: Union[List[dict], dict]) -> List[np.ndarray]:
    """
    Convert ECG data to vectorcardiogram (VCG) data using the Kors matrix method

    Parameters
    ----------
    ecg : list of dict or list
        List of ECG dict data, or ECG dict data directly, with dict keys corresponding to ECG outputs

    Returns
    -------
    vcg: list of np.ndarray
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

    if isinstance(ecg, dict):
        ecg = [ecg]

    vcg = list()
    for sim_ecg in ecg:
        ecg_matrix = np.array([sim_ecg['LI'], sim_ecg['LII'], sim_ecg['V1'], sim_ecg['V2'], sim_ecg['V3'],
                               sim_ecg['V4'], sim_ecg['V5'], sim_ecg['V6']])
        vcg.append(np.dot(ecg_matrix.transpose(), kors))

    return vcg


def get_qrs_start_end(vcg: Union[list, np.ndarray],
                      time: List = None,
                      dt: float = 2,
                      velocity_offset: int = 2,
                      low_p: float = 40,
                      order: int = 2,
                      threshold_frac_start: float = 0.22,
                      threshold_frac_end: float = 0.54,
                      filter_sv: bool = True,
                      t_end: float = 200,
                      matlab_match: bool = False) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate the extent of the VCG QRS complex on the basis of max derivative

    Calculate the start and end points, and hence duration, of the QRS complex of a list of VCGs. It does this by
    finding the time at which the spatial velocity of the VCG exceeds a threshold value (the start time), then searches
    backwards from the end of the VCG to find when this threshold is exceeded (the end time); the start and end
    thresholds do not necessarily have to be the same.

    Parameters
    ----------
    vcg : list of np.ndarray
        List of VCG data to get QRS start and end points for
    time : list of float, optional
        Time variable for the VCG data; provided instead of dt and t_end, default=None
    dt : float, optional
        Time interval between successive data points in the VCG data, default=2ms
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
    t_end : float, optional
        End time of simulation, default=200
    matlab_match : bool, optional
        Apply fudge factor to match Matlab results, default=False

    Returns
    -------
    qrs_start : list of float
        List of start time of QRS complexes of provided VCGs
    qrs_end : list of float
        List of end time of QRS complex of provided VCGs
    qrs_duration : list of float
        List of duration of QRS complex of provided VCGs
    """

    if isinstance(vcg, np.ndarray):
        vcg = [vcg]
    assert 0 < threshold_frac_start < 1, "threshold_frac_start must be between 0 and 1"
    assert 0 < threshold_frac_end < 1, "threshold_frac_end must be between 0 and 1"

    # Create indices to track (1) which colour to plot, and (2) which of the current set of VCGs is currently under
    # consideration
    i_vcg = 0
    sv_time, sv, threshold_start, threshold_end = get_spatial_velocity(vcg=vcg, time=time,
                                                                       velocity_offset=velocity_offset, t_end=t_end,
                                                                       dt=dt,
                                                                       threshold_frac_start=threshold_frac_start,
                                                                       threshold_frac_end=threshold_frac_end,
                                                                       matlab_match=matlab_match, filter_sv=filter_sv,
                                                                       low_p=low_p, order=order)
    qrs_start = list()
    qrs_end = list()
    qrs_duration = list()
    for (sim_sv, sim_time, sim_threshold_start, sim_threshold_end) in zip(sv, sv_time, threshold_start, threshold_end):
        if matlab_match:
            i_qrs_start = np.where(sim_sv > sim_threshold_start)[0][0] + 2
        else:
            i_qrs_start = np.where(sim_sv > sim_threshold_start)[0][0]

        # Find end of QRS complex where it reduces below threshold (searching backwards from end). Fudge factors are
        # added to ensure uniformity with Matlab results
        i_qrs_end = len(sim_sv) - (np.where(np.flip(sim_sv) > sim_threshold_end)[0][0] - 1)
        assert i_qrs_start < i_qrs_end
        assert i_qrs_end <= len(sim_sv)

        qrs_start_temp = sim_time[i_qrs_start]
        qrs_end_temp = sim_time[i_qrs_end]

        qrs_start.append(qrs_start_temp)
        qrs_end.append(qrs_end_temp)
        qrs_duration.append(qrs_end_temp - qrs_start_temp)

        i_vcg += 1

    return qrs_start, qrs_end, qrs_duration


def get_spatial_velocity(vcg: Union[List[np.ndarray], np.ndarray],
                         time: Union[List[np.ndarray], np.ndarray] = None,
                         velocity_offset: int = 2,
                         t_end: Union[float, List[float]] = 200,
                         dt: Union[float, List[float]] = 2,
                         threshold_frac_start: float = 0.22,
                         threshold_frac_end: float = 0.54,
                         matlab_match: bool = False,
                         filter_sv: bool = True,
                         low_p: float = 40,
                         order: int = 2) -> Tuple[List[float], List[List[float]], List[float], List[float]]:
    """
    Calculate spatial velocity

    Calculate the spatial velocity of a VCG, in terms of calculating the gradient of the VCG in each of its x,
    y and z components, before combining these components in a Euclidian norm. Will then find the point at which the
    spatial velocity exceeds a threshold value, and the point at which it declines below another threshold value.

    Parameters
    ----------
    vcg : list of np.ndarray or np.ndarray
        VCG data to analyse
    time : list of np.ndarray or np.ndarray, optional
        Time data associated with the VCG data. Provided instead of dt/t_end data, default=None (will calculate based on
        dt and t_end)
    velocity_offset : int, optional
        Offset between values in VCG over which to calculate spatial velocity, i.e. 1 will use neighbouring values to
        calculate the gradient/velocity. Default=2
    t_end : float, optional
        End time of simulation, default=200
    dt : float, optional
        Time interval between successive data points in the VCG data, default=2ms
    threshold_frac_start : float, optional
        Fraction of maximum spatial velocity to trigger start of QRS detection, default=0.15
    threshold_frac_end : float, optional
        Fraction of maximum spatial velocity to trigger end of QRS detection, default=0.15
    matlab_match : bool, optional
        Apply fudge factor to match Matlab results, default=False
    filter_sv : bool, optional
        Whether or not to apply filtering to spatial velocity prior to finding the start/end points for the
        threshold, default=True
    low_p : float, optional
        Low frequency for bandpass filter, default=40
    order : int, optional
        Order for Butterworth filter, default=2

    Returns
    -------
    sv_time : list of float
        x-values against which to measure the spatial velocity, i.e. corresponds to the time for the measurement of
        the spatial velocity points
    sv : list of list of float
        Spatial velocity data, filtered according to input parameters
    threshold_start_full: list of float
        Absolute values for the threshold for a given spatial velocity trace, rather than the relative values
        originally input
    threshold_end_full: list of float
        Absolute values for the threshold for a given spatial velocity trace, rather than the relative values
        originally input

    References
    ----------
    Calculation of spatial velocity based on:
    Kors JA, van Herpen G.
        Methodology of QT-interval measurement in the modular ECG analysis system (MEANS)
        Ann Noninvasive Electrocardiol. 2009 Jan;14 Suppl 1:S48-53. doi: 10.1111/j.1542-474X.2008.00261.x.
    Xue JQ
        Robust QT Interval Estimation—From Algorithm to Validation
        Ann Noninvasive Electrocardiol. 2009 Jan;14 Suppl 1:S35-41. doi: 10.1111/j.1542-474X.2008.00264.x.
    Sörnmo L
        A model-based approach to QRS delineation
        Comput Biomed Res. 1987 Dec;20(6):526-42.
    """
    if isinstance(vcg, np.ndarray):
        vcg = [vcg]
    assert 0 < threshold_frac_start < 1, "threshold_frac_start must be between 0 and 1"
    assert 0 < threshold_frac_end < 1, "threshold_frac_end must be between 0 and 1"

    # Prepare time variables (time, dt and t_end), depending on input
    time, dt, t_end = common_analysis.get_time(time=time, dt=dt, t_end=t_end, n_vcg=len(vcg),
                                               len_vcg=[len(sim_vcg) for sim_vcg in vcg])

    sv = list()
    sv_time = list()
    threshold_start_full = list()
    threshold_end_full = list()
    for (sim_vcg, sim_time, sim_dt, sim_t_end) in zip(vcg, time, dt, t_end):
        # Compute spatial velocity of VCG
        dvcg = np.divide(sim_vcg[velocity_offset:] - sim_vcg[:-velocity_offset],
                         sim_time[velocity_offset:, np.newaxis]-sim_time[:-velocity_offset, np.newaxis])

        # Calculates Euclidean distance based on spatial velocity in x, y and z directions, i.e. will calculate
        # sqrt(x^2+y^2+z^2) to get total spatial velocity
        sim_sv = np.linalg.norm(dvcg, axis=1)

        # Determine threshold for QRS complex, then find start of QRS complex. Iteratively remove more of the plot if
        # the 'start' is found to be 0 (implies it is still getting confused by the preceding wave). Alternatively, just
        # cut off the first 10ms of the beat (original Matlab method)
        sample_freq = 1000/sim_dt
        if matlab_match:
            sim_sv = sim_sv[5:]
            sim_time = sim_time[5:-velocity_offset]
            if filter_sv:
                sim_sv = common_analysis.filter_egm(sim_sv, sample_freq, low_p, order)
            threshold_start = max(sim_sv)*threshold_frac_start
            threshold_end = max(sim_sv)*threshold_frac_end
        else:
            sim_time = sim_time[:-velocity_offset]
            threshold_start = max(sim_sv)*threshold_frac_start
            if filter_sv:
                sv_filtered = common_analysis.filter_egm(sim_sv, sample_freq, low_p, order)
            else:
                sv_filtered = sim_sv
            i_qrs_start = np.where(sv_filtered > threshold_start)[0][0]
            sim_sv_orig = sim_sv
            sim_time_orig = sim_time
            while i_qrs_start == 0:
                sim_sv = sim_sv[1:]
                sim_time = sim_time[1:]
                threshold_start = max(sim_sv) * threshold_frac_start

                if filter_sv:
                    sv_filtered = common_analysis.filter_egm(sim_sv, sample_freq, low_p, order)
                else:
                    sv_filtered = sim_sv
                i_qrs_start = np.where(sv_filtered > threshold_start)[0][0]
                if sim_time[0] > 50:
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(sim_time_orig, sim_sv_orig, linewidth=3)
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Spatial Velocity')
                    ax.axhline(max(sim_sv) * threshold_frac_start, label='Threshold={}'.format(threshold_frac_start))
                    ax.legend()
                    raise Exception('More than 50ms of trace removed - try changing threshold_frac_start')
            threshold_end = max(sim_sv) * threshold_frac_end
            sim_sv = sv_filtered
        sv.append(sim_sv)
        sv_time.append(sim_time)
        threshold_start_full.append(threshold_start)
        threshold_end_full.append(threshold_end)

    return sv_time, sv, threshold_start_full, threshold_end_full


def get_qrs_area(vcg: Union[List[np.ndarray], np.ndarray], qrs_start: Optional[List[float]] = None,
                 qrs_end: Optional[List[float]] = None, dt: float = 2, t_end: float = 200, matlab_match: bool = False) \
        -> Tuple[List[Union[float, int]], List[Union[float, np.ndarray]], List[float]]:
    """
    Calculate area under QRS complex on VCG.

    Calculate the area under the VCG between two intervals (usually QRS start and QRS end). This is calculated in two
    ways: a 'Pythagorean' method, wherein the area under each of the VCG(x), VCG(y) and VCG(z) curves are calculated,
    then combined in a Euclidean norm, or a '3D' method, wherein the area of the arc traced in 3D space between
    successive timepoints is calculated, then summed.

    Parameters
    ----------
    vcg : np.ndarray or list of np.ndarray
        VCG data from which to get area
    qrs_start : list of float, optional
        Start times (NOT INDICES) for where to calculate area under curve from, default=start
    qrs_end : list of float, optional
        End times (NOT INDICES) for where to calculate are under curve until, default=end
    dt: float, optional
        Time interval between recorded points, default=2
    t_end : float, optional
        End time of VCG data, default=200
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

    qrs_area_3d = list()
    qrs_area_pythag = list()
    qrs_area_components = list()
    for sim_vcg, sim_qrs_start, sim_qrs_end in zip(vcg, qrs_start, qrs_end):
        # Recalculate indices for start and end points of QRS, and extract relevant data
        i_qrs_start, i_qrs_end = common_analysis.convert_time_to_index_deprecated(sim_qrs_start, sim_qrs_end, t_end=t_end, dt=dt)
        if matlab_match:
            sim_vcg_qrs = sim_vcg[i_qrs_start - 1:i_qrs_end + 1]
        else:
            sim_vcg_qrs = sim_vcg[i_qrs_start:i_qrs_end + 1]

        # Calculate area under x,y,z curves by trapezium rule, then combine
        qrs_area_temp = np.trapz(sim_vcg_qrs, dx=dt, axis=0)
        qrs_area_components.append(qrs_area_temp)
        qrs_area_pythag.append(np.linalg.norm(qrs_area_temp))

        # Calculate the area under the curve in 3d space wrt to the origin.
        sim_triangles = np.array([(i, j, (0, 0, 0)) for i, j in zip(sim_vcg_qrs[:-1], sim_vcg_qrs[1:])])
        qrs_area_3d.append(sum([smF.simplex_volume(vertices=sim_triangle) for sim_triangle in sim_triangles]))

    return qrs_area_3d, qrs_area_pythag, qrs_area_components


def get_azimuth_elevation(vcg: Union[List[np.ndarray], np.ndarray],
                          t_start: Optional[List[float]] = None,
                          t_end: Optional[List[float]] = None) -> Tuple[List[Iterable[float]], List[Iterable[float]]]:
    """
    Calculate azimuth and elevation angles for a specified section of the VCG.

    Will calculate the azimuth and elevation angles for the VCG at each recorded point, potentially within specified
    limits (e.g. start/end of QRS)

    Parameters
    ----------
    vcg : np.ndarray or list of np.ndarray
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
    if isinstance(vcg, np.ndarray):
        vcg = [vcg]
    assert len(vcg) == len(t_start)
    assert len(vcg) == len(t_end)

    azimuth = list()
    elevation = list()
    for (sim_vcg, sim_t_start, sim_t_end) in zip(vcg, t_start, t_end):
        theta, phi, _ = get_single_vcg_azimuth_elevation(sim_vcg, sim_t_start, sim_t_end, weighted=False)

        azimuth.append(theta)
        elevation.append(phi)

    return azimuth, elevation


def get_weighted_dipole_angles(vcg: Union[List[np.ndarray], np.ndarray],
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
    vcg : np.ndarray or list of np.ndarray
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

    if isinstance(vcg, np.ndarray):
        vcg = [vcg]
    assert len(vcg) == len(t_start)
    assert len(vcg) == len(t_end)

    weighted_average_azimuth = list()
    weighted_average_elev = list()
    unit_weighted_dipole = list()
    for (sim_vcg, sim_t_start, sim_t_end) in zip(vcg, t_start, t_end):
        # Calculate dipole at all points
        theta, phi, dipole_magnitude = get_single_vcg_azimuth_elevation(sim_vcg, sim_t_start, sim_t_end, weighted=True)

        wae = sum(phi) / sum(dipole_magnitude)
        waa = sum(theta) / sum(dipole_magnitude)

        weighted_average_elev.append(wae)
        weighted_average_azimuth.append(waa)
        unit_weighted_dipole.append([sin(wae) * cos(waa), cos(wae), sin(wae) * sin(waa)])

    return weighted_average_azimuth, weighted_average_elev, unit_weighted_dipole


def get_single_vcg_azimuth_elevation(vcg: np.ndarray,
                                     t_start: float,
                                     t_end: float,
                                     weighted: bool = True,
                                     matlab_match: bool = False) \
        -> Tuple[List[float], List[float], np.ndarray]:
    """
    Get the azimuth and elevation data for a single VCG trace, along with the average dipole magnitude.

    Returns the azimuth and elevation angles for a single given VCG trace. Can analyse only a segment of the
    VCG if required, and can weight the angles according to the dipole magnitude. Primarily designed as a helper
    function for get_azimuth_elevation and get_weighted_dipole_angles.

    Parameters
    ----------
    vcg : np.ndarray or list of np.ndarray
        VCG data to calculate
    t_start : float
        Start time from which to calculate the angles
    t_end : float
        End time until which to calculate the angles
    weighted : bool, optional
        Whether or not to weight the returned angles by the magnitude of the dipole at the same moment, default=True
    matlab_match : bool, optional
        Whether or not to match the original Matlab function's output, regarding how the section of the VCG is
        extracted, default=False

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
    i_start, i_end = common_analysis.convert_time_to_index_deprecated(t_start, t_end)
    if matlab_match:
        sim_vcg = vcg[i_start - 1:i_end]
    else:
        sim_vcg = vcg[i_start:i_end + 1]
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


def get_dipole_magnitudes(vcg: Union[List[np.ndarray], np.ndarray],
                          time: Optional[List[float]] = None,
                          t_start: Union[float, List[float]] = 0,
                          t_end: Union[float, List[float]] = -1,
                          dt: Optional[List[float]] = None,
                          matlab_match: bool = False) -> Tuple[List[List[float]], List[float], List[float],
                                                               List[List[float]], List]:
    """Calculates metrics relating to the magnitude of the weighted dipole of the VCG

    Returns the mean weighted dipole, maximum dipole magnitude,(x,y.z) components of the maximum dipole and the time
    at which the maximum dipole occurs

    Parameters
    ----------
    vcg : np.ndarray or list of np.ndarray
        VCG data to calculate
    time : list of float, optional
        Time data for VCG data
    t_start : list of float, optional
        Start time from which to calculate the magnitude, default=0 (for any other value to be recognisable,
        time variable must be given)
    t_end : list of float, optional
        End time until which to calculate the magnitudes, default=end (for any other value to be recognisable,
        time variable must be given)
    dt : list of float, default=None
        Time interval for recorded data, only used if time is not given, default=None
    matlab_match : bool, optional
        Whether or not to match the original Matlab function's output, regarding how the section of the VCG is
        extracted, default=False

    Returns
    -------
    dipole_magnitude : list of list of float
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
    if isinstance(vcg, np.ndarray):
        vcg = [vcg]

    if isinstance(time, list):
        if not hasattr(time[0], '__iter__'):
            time = [time]
        else:
            assert len(vcg) == len(time), "Time and VCG variables mismatched"
    else:
        time = [None for _ in range(len(vcg))]

    t_start = common_analysis.convert_input_to_list(t_start, n_list=len(vcg), default_entry=t_start)
    # if t_start is not None:
    #     assert len(vcg) == len(t_start)
    # else:
    #     t_start = [None for _ in range(len(vcg))]

    t_end = common_analysis.convert_input_to_list(t_end, n_list=len(vcg), default_entry=t_end)
    # if t_end is not None:
    #     assert len(vcg) == len(t_end)
    # else:
    #     t_end = [None for _ in range(len(vcg))]

    dt = common_analysis.convert_input_to_list(dt, n_list=len(vcg))

    dipole_magnitude = list()
    weighted_magnitude = list()
    max_dipole_magnitude = list()
    max_dipole_components = list()
    max_dipole_time = list()
    for (sim_vcg, sim_time, sim_t_start, sim_t_end, sim_dt) in zip(vcg, time, t_start, t_end, dt):
        # Calculate dipole at all points
        # i_start, i_end = common_analysis.convert_time_to_index_deprecated(sim_t_start, sim_t_end, time=sim_time)
        i_start = common_analysis.convert_time_to_index(sim_t_start, time=sim_time,
                                                        t_start=sim_t_start, t_end=sim_t_end, dt=sim_dt)
        i_end = common_analysis.convert_time_to_index(sim_t_end, time=sim_time,
                                                      t_start=sim_t_start, t_end=sim_t_end, dt=sim_dt)
        if matlab_match:
            sim_vcg_qrs = sim_vcg[i_start-1:i_end]
        else:
            sim_vcg_qrs = sim_vcg[i_start:i_end+1]
        sim_dipole_magnitude = np.linalg.norm(sim_vcg_qrs, axis=1)

        dipole_magnitude.append(sim_dipole_magnitude)
        weighted_magnitude.append(sum(sim_dipole_magnitude) / len(sim_vcg_qrs))
        max_dipole_magnitude.append(max(sim_dipole_magnitude))
        i_max = np.where(sim_dipole_magnitude == max(sim_dipole_magnitude))
        assert len(i_max) == 1
        max_dipole_components.append(sim_vcg_qrs[i_max[0]])
        max_dipole_time.append(common_analysis.convert_index_to_time(i_max[0], time=sim_time,
                                                                     t_start=sim_t_start, t_end=sim_t_end))

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


def compare_dipole_angles(vcg1: np.ndarray,
                          vcg2: np.ndarray,
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
    vcg1 : np.ndarray
        First VCG trace to consider
    vcg2 : np.ndarray
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
    i_start1, i_end1 = common_analysis.convert_time_to_index_deprecated(t_start1, t_end1)
    i_start2, i_end2 = common_analysis.convert_time_to_index_deprecated(t_start2, t_end2)

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
