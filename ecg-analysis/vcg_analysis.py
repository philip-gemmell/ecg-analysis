import numpy as np
import math
from math import sin, cos, acos, atan2
import warnings
from typing import Union, List, Tuple

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


def get_qrs_start_end(vcg: Union[list, np.ndarray], dt: float = 2, velocity_offset: int = 2, low_p: float = 40,
                      order: int = 2, threshold_frac_start: float = 0.15, threshold_frac_end: float = 0.15,
                      filter_sv: bool = True, t_end: float = 200, matlab_match: bool = False) -> Tuple[List[float],
                                                                                                       List[float],
                                                                                                       List[float]]:
    """
    Calculate the extent of the VCG QRS complex on the basis of max derivative

    Calculate the start and end points, and hence duration, of the QRS complex of a list of VCGs. It does this by
    finding the time at which the spatial velocity of the VCG exceeds a threshold value (the start time), then searches
    backwards from the end of the VCG to find when this threshold is exceeded (the end time).

    Parameters
    ----------
    vcg : list of np.ndarray
        List of VCG data to get QRS start and end points for
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
    x_val, sv, threshold_start, threshold_end = get_spatial_velocity(vcg=vcg, velocity_offset=velocity_offset,
                                                                     t_end=t_end, dt=dt,
                                                                     threshold_frac_start=threshold_frac_start,
                                                                     threshold_frac_end=threshold_frac_end,
                                                                     matlab_match=matlab_match, filter_sv=filter_sv,
                                                                     low_p=low_p, order=order)
    qrs_start = list()
    qrs_end = list()
    qrs_duration = list()
    for (sim_sv, sim_x, sim_threshold_start, sim_threshold_end) in zip(sv, x_val, threshold_start, threshold_end):
        if matlab_match:
            i_qrs_start = np.where(sim_sv > sim_threshold_start)[0][0] + 2
        else:
            i_qrs_start = np.where(sim_sv > sim_threshold_start)[0][0]

        # Find end of QRS complex where it reduces below threshold (searching backwards from end). Fudge factors are
        # added to ensure uniformity with Matlab results
        i_qrs_end = len(sim_sv) - (np.where(np.flip(sim_sv) > sim_threshold_end)[0][0] - 1)
        assert i_qrs_start < i_qrs_end
        assert i_qrs_end <= len(sim_sv)

        qrs_start_temp = sim_x[i_qrs_start]
        qrs_end_temp = sim_x[i_qrs_end]

        qrs_start.append(qrs_start_temp)
        qrs_end.append(qrs_end_temp)
        qrs_duration.append(qrs_end_temp - qrs_start_temp)

        i_vcg += 1

    return qrs_start, qrs_end, qrs_duration


def get_spatial_velocity(vcg: Union[List[np.ndarray], np.ndarray], velocity_offset: int = 2, t_end: float = 200,
                         dt: float = 2, threshold_frac_start: float = 0.15, threshold_frac_end: float = 0.15,
                         matlab_match: bool = False, filter_sv: bool = True, low_p: float = 40, order: int = 2) -> \
        Tuple[List[float], List[List[float]], List[float], List[float]]:
    """
    Calculate spatial velocity

    Calculate the spatial velocity of a VCG, in terms of calculating the gradient of the VCG in each of its x,
    y and z components, before combining these components in a Euclidian norm. Will then find the point at which the
    spatial velocity exceeds a threshold value, and the point at which it declines below another threshold value.

    Parameters
    ----------
    vcg : list of np.ndarray or np.ndarray
        VCG data to analyse
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
        Whether or not to apply filtering to spatial velocity prior to finding the start/end points for the threshold
    low_p : float, optional
        Low frequency for bandpass filter, default=40
    order : int, optional
        Order for Butterworth filter, default=2

    Returns
    -------
    x_val : list of float
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
    """
    if isinstance(vcg, np.ndarray):
        vcg = [vcg]
    assert 0 < threshold_frac_start < 1, "threshold_frac_start must be between 0 and 1"
    assert 0 < threshold_frac_end < 1, "threshold_frac_end must be between 0 and 1"

    sv = list()
    x_val = list()
    threshold_start_full = list()
    threshold_end_full = list()
    for sim_vcg in vcg:
        # Compute spatial velocity of VCG
        dvcg = ((sim_vcg[velocity_offset:] - sim_vcg[:-velocity_offset]) / 2) * dt

        # Calculates Euclidean distance based on spatial velocity in x, y and z directions
        sim_sv = np.linalg.norm(dvcg, axis=1)

        # Determine threshold for QRS complex, then find start of QRS complex. Iteratively remove more of the plot if
        # the 'start' is found to be 0 (implies it is still getting confused by the preceding wave). Alternatively, just
        # cut off the first 10ms of the beat (original Matlab method)
        sample_freq = 1000/dt
        if matlab_match:
            sim_sv = sim_sv[5:]
            sim_x = list(range(velocity_offset, t_end, dt))[5:]
            if filter_sv:
                sim_sv = common_analysis.filter_egm(sim_sv, sample_freq, low_p, order)
            threshold_start = max(sim_sv)*threshold_frac_start
            threshold_end = max(sim_sv)*threshold_frac_end
        else:
            sim_x = list(range(velocity_offset, t_end, dt))
            threshold_start = max(sim_sv)*threshold_frac_start
            if filter_sv:
                sv_filtered = common_analysis.filter_egm(sim_sv, sample_freq, low_p, order)
            else:
                sv_filtered = sim_sv
            i_qrs_start = np.where(sv_filtered > threshold_start)[0][0]
            while i_qrs_start == 0:
                sim_sv = sim_sv[1:]
                sim_x = sim_x[1:]
                threshold_start = max(sim_sv) * threshold_frac_start

                if filter_sv:
                    sv_filtered = common_analysis.filter_egm(sim_sv, sample_freq, low_p, order)
                else:
                    sv_filtered = sim_sv
                i_qrs_start = np.where(sv_filtered > threshold_start)[0][0]
                if sim_x[0] > 50:
                    raise Exception('More than 50ms of trace removed - try changing threshold_frac_start')
            threshold_end = max(sim_sv) * threshold_frac_end
            sim_sv = sv_filtered
        sv.append(sim_sv)
        x_val.append(sim_x)
        threshold_start_full.append(threshold_start)
        threshold_end_full.append(threshold_end)

    return x_val, sv, threshold_start_full, threshold_end_full


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

    qrs_area_3d = list()
    qrs_area_pythag = list()
    qrs_area_components = list()
    for sim_vcg, sim_qrs_start, sim_qrs_end in zip(vcg, qrs_start, qrs_end):
        # Recalculate indices for start and end points of QRS, and extract relevant data
        i_qrs_start, i_qrs_end = common_analysis.convert_time_to_index(sim_qrs_start, sim_qrs_end, t_end=t_end, dt=dt)
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


def get_azimuth_elevation(vcg, t_start=None, t_end=None):
    """ Calculate azimuth and elevation angles for a specified section of the VCG. """
    if isinstance(vcg, np.ndarray):
        vcg = [vcg]
    assert len(vcg) == len(t_start)
    assert len(vcg) == len(t_end)

    azimuth = list()
    elevation = list()
    for (sim_vcg, sim_t_start, sim_t_end) in zip(vcg, t_start, t_end):
        theta, phi, _ = get_simvcg_azimuth_elevation(sim_vcg, sim_t_start, sim_t_end, weighted=False)

        azimuth.append(theta)
        elevation.append(phi)

    return azimuth, elevation


def get_weighted_dipole_angles(vcg, t_start=None, t_end=None):
    """ Calculate metrics relating to the angles of the weighted dipole of the VCG. Usually used with QRS limits. """
    # WAA (Weighted Angle Azimuth): Weighted average angle of azimuth
    # WAE (Weighted Angle Elevation: Weighted average angle of elevation, inclination above xy-plane.

    if isinstance(vcg, np.ndarray):
        vcg = [vcg]
    assert len(vcg) == len(t_start)
    assert len(vcg) == len(t_end)

    weighted_average_azimuth = list()
    weighted_average_elev = list()
    unit_weighted_dipole = list()
    for (sim_vcg, sim_t_start, sim_t_end) in zip(vcg, t_start, t_end):
        # Calculate dipole at all points
        theta, phi, dipole_magnitude = get_simvcg_azimuth_elevation(sim_vcg, sim_t_start, sim_t_end, weighted=True)

        wae = sum(phi) / sum(dipole_magnitude)
        waa = sum(theta) / sum(dipole_magnitude)

        weighted_average_elev.append(wae)
        weighted_average_azimuth.append(waa)
        unit_weighted_dipole.append([sin(wae) * cos(waa), cos(wae), sin(wae) * sin(waa)])

    return weighted_average_azimuth, weighted_average_elev, unit_weighted_dipole


def get_simvcg_azimuth_elevation(vcg, t_start, t_end, weighted=True, matlab_match=False):
    """ Helper function to get azimuth and elevation data for a single VCG trace. """
    i_start, i_end = common_analysis.convert_time_to_index(t_start, t_end)
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


def get_dipole_magnitudes(vcg, t_start=None, t_end=None, matlab_match=False):
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
        # Calculate dipole at all points
        i_start, i_end = common_analysis.convert_time_to_index(sim_t_start, sim_t_end)
        if matlab_match:
            sim_vcg_qrs = sim_vcg[i_start-1:i_end]
        else:
            sim_vcg_qrs = sim_vcg[i_start:i_end+1]
        dipole_magnitude = np.linalg.norm(sim_vcg_qrs, axis=1)

        weighted_magnitude.append(sum(dipole_magnitude)/len(sim_vcg_qrs))
        max_dipole_magnitude.append(max(dipole_magnitude))
        i_max = np.where(dipole_magnitude == max(dipole_magnitude))
        max_dipole_components.append(sim_vcg_qrs[i_max])
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


def compare_dipole_angles(vcg1, vcg2, t_start1=0, t_end1=None, t_start2=0, t_end2=None, n_compare=10,
                          convert_to_degrees=False, matlab_match=False):
    """ Calculates the angular differences between two VCGs at multiple points during their evolution """

    # Calculate indices for the two VCG traces that correspond to the time points to be compared
    i_start1, i_end1 = common_analysis.convert_time_to_index(t_start1, t_end1)
    i_start2, i_end2 = common_analysis.convert_time_to_index(t_start2, t_end2)

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
