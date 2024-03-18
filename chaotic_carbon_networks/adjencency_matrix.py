import numpy as np
import xarray as xr
import pandas as pd
from rich import print

# TODO: Calculate similarity matrix based on chaotic measures


def calc_similarity_matrix_v4(x):
    """Calculate similarity matrix using Pearson correlation coefficient
    Should be the fastest version, bad scalable
    """
    assert len(x.dims) == 3, "x must have 3 dimensions"
    assert "time" in x.dims, "x must have time dimension"
    assert "lat" in x.dims, "x must have lat dimension"
    assert "lon" in x.dims, "x must have lon dimension"

    x_stacked = x.stack(vertex=("lat", "lon")).dropna(dim="vertex", how="all")
    vertex_coords = x_stacked.coords["vertex"].values

    similarity_matrix = np.abs(np.corrcoef(x_stacked, rowvar=False))
    similarity_matrix = xr.DataArray(
        similarity_matrix,
        dims=("vertex", "vertex_other"),
        coords={
            "vertex": pd.MultiIndex.from_tuples(vertex_coords, names=("lat", "lon")),
            "vertex_other": pd.MultiIndex.from_tuples(vertex_coords, names=("lat_other", "lon_other")),
        },
    )

    # Set diagonal to 0
    similarity_matrix = similarity_matrix.where(~np.eye(len(similarity_matrix), dtype=bool), 0)

    similarity_matrix.attrs = {
        "long_name": "Similarity matrix",
        "var_desc": "Similarity",
        "valid_range": (0, 1),
        "actual_range": (similarity_matrix.min().item(), similarity_matrix.max().item()),
    }

    return similarity_matrix


def calc_adjacency_matrix(similarity_matrix, rr=0.05):
    assert len(similarity_matrix.dims) == 2, "similarity_matrix must have 2 dimensions"
    assert "vertex" in similarity_matrix.dims, "similarity_matrix must have vertex dimension"
    assert "vertex_other" in similarity_matrix.dims, "similarity_matrix must have vertex_other dimension"

    eps = np.nanquantile(similarity_matrix.values, 1 - rr)
    print(f"Using a threshold of {eps} for the adjacency matrix")
    adjacency_matrix = (similarity_matrix > eps).astype(int)
    adjacency_matrix.attrs = {
        "long_name": f"Adjacency matrix",
        "var_desc": "Connected",
        "valid_range": (0, 1),
        "actual_range": (0, 1),
    }
    return adjacency_matrix


def calc_link_length(adj_mtx: xr.DataArray):
    assert len(adj_mtx.dims) == 2, "adj_mtx must have 2 dimensions"
    assert "vertex" in adj_mtx.dims, "adj_mtx must have vertex dimension"
    assert "vertex_other" in adj_mtx.dims, "adj_mtx must have vertex_other dimension"

    lats_i = adj_mtx.coords["lat"] * np.pi / 180
    lons_i = adj_mtx.coords["lon"] * np.pi / 180
    lats_j = adj_mtx.coords["lat_other"] * np.pi / 180
    lons_j = adj_mtx.coords["lon_other"] * np.pi / 180

    d_lat = lats_i - lats_j
    d_lon = lons_i - lons_j

    # Distance between two points on a sphere
    a = np.sin(d_lat / 2) ** 2 + np.cos(lats_i) * np.cos(lats_j) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6371.0088
    d = R * c

    ll_adj_mtx = adj_mtx * d
    ll_adj_mtx.attrs = {
        "long_name": f"Link lengths of {adj_mtx.attrs['long_name']}",
        "units": "km",
        "var_desc": "Link length",
        "valid_range": (0, d.max().item()),
        "actual_range": (ll_adj_mtx.min().item(), ll_adj_mtx.max().item()),
    }
    return ll_adj_mtx
