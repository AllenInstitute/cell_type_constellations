from anndata._io.specs import read_elem
import h5py
import pandas as pd


def get_coords_from_h5ad(
        h5ad_path,
        coord_key):
    """
    Extract a set of coordinates from obsm in and h5ad file.
    Convert them into a KD Tree and return

    Parameters
    ----------
    h5ad_path:
        the path to the h5ad file
    coord_key:
        the key (within obsm) of the coordinate array being extracted

    Returns
    -------
    kd_tree:
        a scipy.spatial.cKDTree built off of the corresponding
        coordinates
    """
    with h5py.File(h5ad_path, 'r') as src:
        obsm = read_elem(src['obsm'])
        if coord_key not in obsm.keys():
            raise KeyError(f'key {coord_key} not in obsm')
        coords = obsm[coord_key][()]

    if isinstance(coords, pd.DataFrame):
        coords = coords.to_numpy()

    return coords
