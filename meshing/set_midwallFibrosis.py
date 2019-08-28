import pandas as pd
import math
import random
import numpy as np
from scipy.spatial import distance
import open_mesh


def get_centroid_uvc(uvc, elem):
    """ Constructs centroid based UVC """

    elem_val = elem.values

    """ Remove region data, then reshape data """
    elem_val_noregion = np.delete(elem_val, 4, axis=1)
    elem_val_flat = elem_val_noregion.reshape(elem_val_noregion.size)

    """ Extract centroid values, check for sign flips, then calculate means """
    z = uvc.loc[elem_val_flat, 0].values.reshape(elem_val_noregion.shape)

    rho = uvc.loc[elem_val_flat, 1].values.reshape(elem_val_noregion.shape)
    rho_high = rho > 0.9
    rho_high = rho_high.any(axis=1)
    rho_low = rho < 0.1
    rho_low = rho_low.any(axis=1)
    rho_highlow = np.vstack([rho_high, rho_low]).transpose()
    rho_highlow = rho_highlow.all(axis=1)
    rho[rho_highlow] = 1

    phi = uvc.loc[elem_val_flat, 2].values.reshape(elem_val_noregion.shape)
    phi_high = phi > math.pi-0.3
    phi_high = phi_high.any(axis=1)
    phi_low = phi < -(math.pi-0.3)
    phi_low = phi_low.any(axis=1)
    phi_highlow = np.vstack([phi_high, phi_low]).transpose()
    phi_highlow = phi_highlow.all(axis=1)
    phi[phi_highlow] = math.pi

    v = uvc.loc[elem_val_flat, 3].values.reshape(elem_val_noregion.shape)

    z = np.mean(z, axis=1)
    rho = np.mean(rho, axis=1)
    rho[rho > 1] = 1
    phi = np.mean(phi, axis=1)
    v = np.mean(v, axis=1)
    v[v != -1] = 1

    print("Centroids of elements calculated.")

    return np.vstack([z, rho, phi, v]).transpose()


def set_scar(centroid_uvc, elem, lon, phi_bounds=None, rho_bounds=None, z_bounds=None, lv_scar=True, p_low=0.6,
             p_bz=0.75, p_dense=0.9):
    """ Set scar to a provided mesh (mesh already extracted, centroid UVC determined """

    """ Set default bounds """
    phi_bounds, rho_bounds, z_bounds = get_scar_bounds(phi_bounds, rho_bounds, z_bounds, lv_scar)

    lv_check = centroid_uvc[:, 3] == -1

    """ Find boundary zone scar """
    within_z = (centroid_uvc[:, 0] > z_bounds[0]) & (centroid_uvc[:, 0] < z_bounds[1])
    within_rho = (centroid_uvc[:, 1] > rho_bounds[0]) & (centroid_uvc[:, 1] < rho_bounds[1])
    within_phi = (centroid_uvc[:, 2] > phi_bounds[0]) & (centroid_uvc[:, 2] < phi_bounds[1])
    within_scar = lv_check & within_z & within_rho & within_phi

    """ Find intermediate scar """
    rho_intermediate = get_bz_and_dense_scar(rho_bounds, (1/12))
    phi_intermediate = get_bz_and_dense_scar(phi_bounds, (1/12))
    within_rho_intermediate = (centroid_uvc[:, 1] > rho_intermediate[0]) & (centroid_uvc[:, 1] < rho_intermediate[1])
    within_phi_intermediate = (centroid_uvc[:, 2] > phi_intermediate[0]) & (centroid_uvc[:, 2] < phi_intermediate[1])
    within_intermediate = within_scar & within_rho_intermediate & within_phi_intermediate

    """ Find dense scar """
    rho_dense = get_bz_and_dense_scar(rho_bounds, (1/6))
    phi_dense = get_bz_and_dense_scar(phi_bounds, (1/6))
    within_rho_dense = (centroid_uvc[:, 1] > rho_dense[0]) & (centroid_uvc[:, 1] < rho_dense[1])
    within_phi_dense = (centroid_uvc[:, 2] > phi_dense[0]) & (centroid_uvc[:, 2] < phi_dense[1])
    within_dense = within_intermediate & within_rho_dense & within_phi_dense

    """ Apply data to elem data """
    elem.loc[within_scar, 5] = 200
    elem.loc[within_intermediate, 5] = 201
    elem.loc[within_dense, 5] = 202

    """ Apply loss of fibre direction to the lon data (based on random number generation) """
    rand_num = np.array([random.random() for _ in range(centroid_uvc.shape[0])])
    nofibre_scar = (rand_num < p_low) & within_scar
    nofibre_intermediate = (rand_num < p_bz) & within_intermediate
    nofibre_dense = (rand_num < p_dense) & within_dense
    lon.loc[nofibre_scar, :] = 0
    lon.loc[nofibre_intermediate, :] = 0
    lon.loc[nofibre_dense, :] = 0

    print("Scar set.")

    return elem, lon


def check_scar_from_file(file_pts, file_elem, scar_label=None, convert_to_centroid_uvc=True):
    """ Confirm parameters of scar in a given file.  """
    pts, elem, _ = open_mesh.get_mesh(file_pts=file_pts, file_elem=file_elem, file_lon=None)
    phi_bounds, rho_bounds, z_bounds = check_scar_from_var(pts, elem,
                                                           scar_label=scar_label,
                                                           convert_to_centroid_uvc=convert_to_centroid_uvc)

    return phi_bounds, rho_bounds, z_bounds


def check_scar_from_var(pts, elem, scar_label=None, convert_to_centroid_uvc=True):
    """ Confirm parameters of scar from a given set of pts, elem and lon variables. """

    """ Confirm whether scar_label exists """
    if scar_label == 'scar':
        scar_label = 200
    elif scar_label == 'intermediate':
        scar_label = 201
    elif scar_label == 'dense':
        scar_label = 202
    elif scar_label is None:
        scar_label = 200

    if convert_to_centroid_uvc:
        pts_check = get_centroid_uvc(uvc=pts, elem=elem)
        pts_check = pd.DataFrame(pts_check)
    else:
        pts_check = pts

    """ Extract elements that are classified as scar, then extract the relevant UVC coordinates from the pts
        variable """
    if convert_to_centroid_uvc:
        elem_scar_bool = elem.loc[:, 5] == scar_label
        pts_check = pts_check[elem_scar_bool]

        z_scar = np.unique(pts_check.loc[:, 0].values)
        rho_scar = np.unique(pts_check.loc[:, 1].values)
        phi_scar = np.unique(pts_check.loc[:, 2].values)
        v_scar = np.unique(pts_check.loc[:, 3].values)
    else:
        elem_scar = elem[elem.loc[:, 5] == scar_label].values
        elem_scar = np.delete(elem_scar, 4, axis=1)
        elem_scar = elem_scar.reshape(elem_scar.size)

        z_scar = np.unique(pts_check.loc[elem_scar, 0].values)
        rho_scar = np.unique(pts_check.loc[elem_scar, 1].values)
        phi_scar = np.unique(pts_check.loc[elem_scar, 2].values)
        v_scar = np.unique(pts_check.loc[elem_scar, 3].values)

    assert len(v_scar) == 1  # Confirm all elements of the scar are in the left ventricle

    phi_bounds = [min(phi_scar), max(phi_scar)]
    rho_bounds = [min(rho_scar), max(rho_scar)]
    z_bounds = [min(z_scar), max(z_scar)]

    return phi_bounds, rho_bounds, z_bounds


def get_mesh_volume_from_file(file_pts, file_elem, scar_label=None):
    """ Calculate volume of mesh. If label provided, will calculate volume that is labelled as that.
        NB: Do not use UVC pts file, but rather base mesh file (../torso_final_ref_smooth_noAir_myo.pts) """

    pts, elem, _ = open_mesh.get_mesh(file_pts=file_pts, file_elem=file_elem, file_lon=None)

    return get_mesh_volume_from_var(pts, elem, scar_label)


def get_mesh_volume_from_var(pts, elem, scar_label=None):
    """ Calculate volume of mesh. If label provided, will calculate volume that is labelled as that. If multiple
        labels given, will calculate total volume for those labels combined.
        NB: Do not use UVC pts file, but rather base mesh data. """

    """ Extract data only relating to scar_label """
    if scar_label is not None:
        if not isinstance(scar_label, list):
            scar_label = [scar_label]
        scar_mask = elem.loc[:, 5].isin(scar_label)
        elem_val = elem[scar_mask].values
    else:
        elem_val = elem.values
    elem_val = np.delete(elem_val, 4, axis=1)
    elem_val_flat = elem_val.reshape(elem_val.size)

    """ Extract x, y, z data, then reshape to required format """
    xyz = pts.loc[elem_val_flat, :].values
    shape_xyz_new = (int(xyz.shape[0]/4), 4, 3)
    try:
        xyz_reshape = np.reshape(xyz, shape_xyz_new)
    except ValueError:
        print("Unable to reshape. Maybe used a UVC pts file instead of a xyz pts file...?")
        return None

    """ Try parallelising the process of calculating the volume of the individual elements """
    import multiprocessing

    try:
        cpus = multiprocessing.cpu_count()-2
    except NotImplementedError:
        cpus = 6
    with multiprocessing.Pool(processes=cpus) as pool:
        volume = pool.map(__simplex_volume_vertices, xyz_reshape)
    return sum(volume)


def __simplex_volume_vertices(vertices):
    return simplex_volume(vertices=vertices)


def simplex_volume(*, vertices=None, sides=None) -> float:
    """
    Return the volume of the simplex with given vertices or sides.

    If vertices are given they must be in a NumPy array with shape (N+1, N):
    the position vectors of the N+1 vertices in N dimensions. If the sides
    are given, they must be the compressed pairwise distance matrix as
    returned from scipy.spatial.distance.pdist.

    Raises a ValueError if the vertices do not form a simplex (for example,
    because they are coplanar, colinear or coincident).

    Warning: this algorithm has not been tested for numerical stability.

    Originally from
    https://codereview.stackexchange.com/questions/77593/calculating-the-volume-of-a-tetrahedron
    """

    # Implements http://mathworld.wolfram.com/Cayley-MengerDeterminant.html

    if (vertices is None) == (sides is None):
        raise ValueError("Exactly one of vertices and sides must be given")

    # β_ij = |v_i - v_k|²
    if sides is None:
        vertices = np.asarray(vertices, dtype=float)
        sq_dists = distance.pdist(vertices, metric='sqeuclidean')

    else:
        sides = np.asarray(sides, dtype=float)
        if not distance.is_valid_y(sides):
            raise ValueError("Invalid number or type of side lengths")

        sq_dists = sides ** 2

    # Add border while compressed
    num_verts = distance.num_obs_y(sq_dists)
    bordered = np.concatenate((np.ones(num_verts), sq_dists))

    # Make matrix and find volume
    sq_dists_mat = distance.squareform(bordered)

    coeff = - (-2) ** (num_verts-1) * math.factorial(num_verts-1) ** 2
    vol_square = np.linalg.det(sq_dists_mat) / coeff

    if vol_square <= 0:
        raise ValueError('Provided vertices do not form a tetrahedron')

    return np.sqrt(vol_square)


def get_surface_area(file_surf, file_pts, convert_to_centroid_uvc=False):
    """ Will calculate the surface area for a given surface file """
    pts, _, _ = open_mesh.get_mesh(file_pts=file_pts)
    surf = pd.read_csv(file_surf, skiprows=1, delimiter=' ', usecols=(1, 2, 3), header=None)

    surf_val = surf.values
    surf_val_flat = surf_val.reshape(surf_val.size)

    # if convert_to_centroid_uvc:
    #     pts_check = get_centroid_uvc(uvc=pts, elem=elem)
    #     pts_check = pd.DataFrame(pts_check)
    # else:
    #     pts_check = pts

    """ Extract x, y, z data, then reshape to required format """
    xyz = pts.loc[surf_val_flat, :].values
    shape_xyz_new = (int(xyz.shape[0] / 3), 3, 3)
    try:
        xyz_reshape = np.reshape(xyz, shape_xyz_new)
    except ValueError:
        print("Unable to reshape. Maybe used a UVC pts file instead of a xyz pts file...?")
        return None
    area = sum([simplex_volume(vertices=surf_pts) for surf_pts in xyz_reshape])
    # area = sum([get_triangle_area(vertices=surf_pts) for surf_pts in xyz_reshape])
    return area


def get_triangle_area(vertices):
    """ Calculate the area of a triangle via the cross product """

    ab = vertices[0]-vertices[2]
    ac = vertices[0]-vertices[1]
    cross_product = np.cross(ab, ac)
    return np.linalg.norm(cross_product)/2


def get_bz_and_dense_scar(bounds, bound_reduction):
    """ Will calculate new bounds for a given bound_reduction fraction, e.g. will look for bounds that are 1/6 less
        on either side of the original bounds"""
    lower_bound = (bounds[1]-bounds[0])*bound_reduction + bounds[0]
    upper_bound = (bounds[1]-bounds[0])*(1-bound_reduction) + bounds[0]
    return lower_bound, upper_bound


def retag_elements(elem, orig_tag=200, new_tag=22):
    """ Reset scar in a given elem variable """
    elem.loc[elem[5] == orig_tag, 5] = new_tag
    print("Scar reset.")

    return elem


def get_scar_bounds(phi_bounds=None, rho_bounds=None, z_bounds=None, lv_scar=True):
    """ Set default bounds """
    if phi_bounds is None:
        if lv_scar:
            phi_bounds = [math.pi/2, math.pi]
        else:
            phi_bounds = [-1, 1]
    if rho_bounds is None:
        rho_bounds = [0.1, 0.9]
    if z_bounds is None:
        z_bounds = [0.3, 0.9]

    return phi_bounds, rho_bounds, z_bounds


def set_scar_to_meshfile(file_uvc=None, file_elem=None, file_lon=None, lv_scar=True, phi_bounds=None,
                         rho_bounds=None, z_bounds=None, p_low=0.6, p_bz=0.75, p_dense=0.9, output_folder=None):
    """ Set scar in a given series of mesh files """

    """ Extract original mesh details """
    if file_uvc is None:
        file_uvc = '/home/pg16/Documents/ecg-scar/meshing/meshes/COMBINED_COORDS_Z_RHO_PHI_V.pts'
    if file_elem is None:
        file_elem = '/home/pg16/Documents/ecg-scar/meshing/meshes/torso_final_ref_smooth_noAir_myoFastEndo.elem'
    if file_lon is None:
        file_lon = '/home/pg16/Documents/ecg-scar/meshing/meshes/torso_final_ref_smooth_noAir_myoFIBRES.lon'
    uvc, elem, lon = open_mesh.get_mesh(file_root=None, file_pts=file_uvc, file_elem=file_elem, file_lon=file_lon)

    """ Reset scar in elem """
    elem = retag_elements(elem, orig_tag=200, new_tag=22)

    """ Determine centroid UVC model """
    centroid_uvc = get_centroid_uvc(uvc=uvc, elem=elem)

    """ Set default bounds """
    phi_bounds, rho_bounds, z_bounds = get_scar_bounds(phi_bounds, rho_bounds, z_bounds, lv_scar)

    """ Define scar within torso model """
    elem, lon = set_scar(centroid_uvc, elem, lon, phi_bounds=phi_bounds, rho_bounds=rho_bounds, z_bounds=z_bounds,
                         lv_scar=lv_scar, p_low=p_low, p_bz=p_bz, p_dense=p_dense)

    """ Derive filenames"""
    if output_folder is None:
        output_folder = ''
    else:
        if output_folder[-1] != '/':
            output_folder += '/'
    if lv_scar:
        output_root = output_folder+'myoScarLV'
    else:
        output_root = output_folder+'myoScarSeptum'
    output_root = '{}_phi{:.5f}-{:.5f}_rho{:.5f}-{:.5f}_z{:.5f}-{:.5f}'.format(output_root, phi_bounds[0],
                                                                               phi_bounds[1], rho_bounds[0],
                                                                               rho_bounds[1], z_bounds[0], z_bounds[1])

    """ Write data to files (don't bother writing .pts file!) """
    open_mesh.write_mesh(output_root, None, elem, lon)

    return None
