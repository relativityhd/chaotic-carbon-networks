import numpy as np
import xarray as xr


def calc_similarity_matrix_v4(x):
    """Calculate similarity matrix using Pearson correlation coefficient
    Should be the fastest version, bad scalable
    """
    assert len(x.dims) == 3, "x must have 3 dimensions"
    assert "time" in x.dims, "x must have time dimension"
    assert "lat" in x.dims, "x must have lat dimension"
    assert "lon" in x.dims, "x must have lon dimension"

    x_stacked = x.stack(vertex=("lat", "lon"))

    similarity_matrix = np.abs(np.corrcoef(x_stacked, rowvar=False))
    similarity_matrix = xr.DataArray(similarity_matrix, dims=("vertex_i", "vertex_j"))

    # similarity_matrix = similarity_matrix.dropna("vertex_i", how="all").dropna("vertex_j", how="all")

    # Set diagonal to 0
    similarity_matrix = similarity_matrix.where(~np.eye(len(similarity_matrix), dtype=bool), 0)

    similarity_matrix.attrs = x.attrs
    similarity_matrix.attrs = {
        "long_name": f"Similarity matrix of {x.attrs.get('long_name', 'unknown')}",
        "var_desc": "Similarity",
        "units": "Similarity",
        "valid_range": (0, 1),
        "actual_range": (similarity_matrix.min().item(), similarity_matrix.max().item()),
    }

    return similarity_matrix


def calc_adjacency_matrix(x, rr=0.05):
    similarity_matrix = calc_similarity_matrix_v4(x)
    eps = np.nanquantile(similarity_matrix.values, 1 - rr)
    print(eps)
    adjacency_matrix = (similarity_matrix > eps).astype(int)
    adjacency_matrix.attrs = {
        "long_name": f"Adjacency matrix of {x.attrs.get('long_name', 'unknown')}",
        "var_desc": "Connected",
        "units": "c",
        "valid_range": (0, 1),
        "actual_range": (0, 1),
    }
    return adjacency_matrix
