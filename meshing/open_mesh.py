import pandas as pd
import glob

""" Add carputils functions (if not already present) """
try:
    from carputils.carpio import igb
except ImportError:
    import sys
    sys.path.append('/home/pg16/software/carputils/')
    from carputils.carpio import igb


def get_mesh(file_root=None, file_pts=None, file_elem=None, file_lon=None):
    """ Function to extract data from .pts, .elem, .lon files """

    """ Establish file names (guessing if required), then read. """
    if file_pts is None:
        file_pts = look_for_file(file_root, '.pts')
    if file_pts is not None:
        pts = pd.read_csv(file_pts, sep=' ', skiprows=1, header=None)
        print("Successfully read {}".format(file_pts))
    else:
        pts = None

    if file_elem is None:
        file_elem = look_for_file(file_root, '.elem')
    if file_elem is not None:
        elem = pd.read_csv(file_elem, sep=' ', skiprows=1, usecols=(1, 2, 3, 4, 5), header=None)
        print("Successfully read {}".format(file_elem))
    else:
        elem = None

    if file_lon is None:
        file_lon = look_for_file(file_root, '.lon')
    if file_lon is not None:
        lon = pd.read_csv(file_lon, sep=' ', skiprows=1, header=None)
        print("Successfully read {}".format(file_elem))
    else:
        lon = None

        # file_elem = glob.glob(file_root+'*.elem')
        # if len(file_elem) > 1:
        #     raise ValueError('Too many matching .elem files')
        # elif len(file_elem) == 0:
        #     raise ValueError('No matching .elem files')
        # file_elem = file_elem[0]

    # if file_lon is None:
    #     file_lon = glob.glob(file_root+'*.lon')
    #     if len(file_lon) > 1:
    #         raise ValueError('Too many matching .lon files')
    #     elif len(file_lon) == 0:
    #         raise ValueError('No matching .lon files')
    #     file_lon = file_lon[0]
    #
    # """ Read files """
    # elem = pd.read_csv(file_elem, sep=' ', skiprows=1, usecols=(1, 2, 3, 4, 5), header=None)
    # print("Successfully read {}".format(file_elem))
    # lon = pd.read_csv(file_lon, sep=' ', skiprows=1, header=None)
    # print("Successfully read {}".format(file_lon))

    return pts, elem, lon


def look_for_file(file_root=None, file_type=None):
    if file_type is None:
        raise ValueError('Need to provide either pts, elem or lon')
    elif file_type[0] != '.':
        file_type = '.'+file_type

    if file_root is not None:
        filename = glob.glob(file_root+'*'+file_type)
        if len(filename) > 1:
            raise ValueError('Too many matching'+file_type+' files')
        elif len(filename) == 0:
            print("No "+file_type+" file found.")
            filename = None
        else:
            filename = filename[0]
    else:
        print("No "+file_type+" file found.")
        filename = None

    return filename


def write_mesh(file_root, pts=None, elem=None, lon=None, precision_pts=None, precision_lon=None):
    """ Write pts, elem and lon data to file """

    """ Ensure *something* is being written! """
    assert ((pts is not None) or (elem is not None) or (lon is not None)), "No data given to write to file."

    """ Adapt precision to default formats """
    if precision_pts is None:
        precision_pts = '%.12g'
    if precision_lon is None:
        precision_lon = '%.5g'

    """ Basic error checking on output file name """
    if file_root[-1] == '.':
        file_root = file_root[:-1]

    if pts is not None:
        with open(file_root+'.pts', 'w') as pFile:
            pFile.write('{}\n'.format(len(pts)))
        pts.to_csv(file_root+'.pts', sep=' ', header=False, index=False, mode='a', float_format=precision_pts)
        print("pts data written to file {}.".format(file_root+'.pts'))

    if elem is not None:
        with open(file_root+'.elem', 'w') as pFile:
            pFile.write('{}\n'.format(len(elem)))
        elem.insert(loc=0, value='Tt', column=0)
        elem.to_csv(file_root+'.elem', sep=' ', header=False, index=False, mode='a')
        print("elem data written to file {}.".format(file_root+'.elem'))
        del elem[0]     # Remove added column to prevent cross-talk problems later

    if lon is not None:
        with open(file_root+'.lon', 'w') as pFile:
            pFile.write('1\n')
        lon.to_csv(file_root+'.lon', sep=' ', header=False, index=False, mode='a', float_format=precision_lon)
        print("lon data written to file {}.".format(file_root+'.lon'))

    return None
