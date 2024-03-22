import numpy as np
import xarray as xr
import pandas as pd
from rich import print, progress
import h3

# Own rust library
from fast_mutual_information import mind

from chaotic_carbon_networks.hex import hexgrid

# TODO: Calculate similarity matrix based on chaotic measures


def calc_mutual_information(x: xr.DataArray):
    """Calculate mutual information between time series"""
    is_hex = "hex_res" in x.attrs
    if is_hex:
        assert len(x.dims) == 2, "x must have 2 dimensions"
        assert "time" in x.dims, "x must have time dimension"
        assert "vertex" in x.dims, "x must have vertex dimension"
        vertex_coords = x.coords["vertex"].values

        mutual_information = mind(x.values, x.values, 32)
        mutual_information = xr.DataArray(
            mutual_information,
            dims=("vertex", "vertex_other"),
            coords={
                "vertex": vertex_coords.copy(),
                "vertex_other": vertex_coords.copy(),
            },
        )
    else:
        assert len(x.dims) == 3, "x must have 3 dimensions"
        assert "time" in x.dims, "x must have time dimension"
        assert "lat" in x.dims, "x must have lat dimension"
        assert "lon" in x.dims, "x must have lon dimension"

        x_stacked = x.stack(vertex=("lat", "lon")).dropna(dim="vertex", how="all")
        vertex_coords = x_stacked.coords["vertex"].values
        vertex_multiindex = pd.MultiIndex.from_tuples(vertex_coords, names=("lat", "lon"))
        vertex_other_multiindex = pd.MultiIndex.from_tuples(vertex_coords, names=("lat_other", "lon_other"))

        mutual_information = mind(x_stacked.values, x_stacked.values, 32)
        mutual_information = xr.DataArray(
            mutual_information,
            dims=("vertex", "vertex_other"),
            coords={
                "vertex": vertex_multiindex,
                "vertex_other": vertex_other_multiindex,
            },
        )

    # Set diagonal to 0
    mutual_information = mutual_information.where(~np.eye(len(mutual_information), dtype=bool), 0)

    mutual_information.attrs = {
        "long_name": "Mutual information",
        "var_desc": "Mutual information",
        "valid_range": (0, np.inf),
        "actual_range": (mutual_information.min().item(), mutual_information.max().item()),
    }

    if is_hex:
        mutual_information.attrs["hex_res"] = x.attrs["hex_res"]

    return mutual_information


def calc_similarity_matrix_with_lag(x, lag=0):
    # TODO: negative lag?
    """Calculate similarity matrix using Pearson correlation coefficient
    Should be the fastest version, bad scalable
    """
    if lag == 0:
        return calc_similarity_matrix_v4(x)
    is_hex = "hex_res" in x.attrs
    if is_hex:
        assert len(x.dims) == 2, "x must have 2 dimensions"
        assert "time" in x.dims, "x must have time dimension"
        assert "vertex" in x.dims, "x must have vertex dimension"

        # Working with negative lag because we want to shift the time series to the left
        y = x.shift(time=-lag)

        vertex_coords = x.coords["vertex"].values

        similarity_matrix = np.abs(np.corrcoef(x.values[:-lag], y.values[:-lag], rowvar=False))
        similarity_matrix = similarity_matrix[: len(vertex_coords), len(vertex_coords) :]

        similarity_matrix = xr.DataArray(
            similarity_matrix,
            dims=("vertex", "vertex_other"),
            coords={
                "vertex": vertex_coords.copy(),
                "vertex_other": vertex_coords.copy(),
            },
        )
    else:
        assert len(x.dims) == 3, "x must have 3 dimensions"
        assert "time" in x.dims, "x must have time dimension"
        assert "lat" in x.dims, "x must have lat dimension"
        assert "lon" in x.dims, "x must have lon dimension"

        x_stacked = x.stack(vertex=("lat", "lon")).dropna(dim="vertex", how="all")

        y_stacked = x_stacked.shift(time=-lag)

        vertex_coords = x_stacked.coords["vertex"].values
        vertex_multiindex = pd.MultiIndex.from_tuples(vertex_coords, names=("lat", "lon"))
        vertex_other_multiindex = pd.MultiIndex.from_tuples(vertex_coords, names=("lat_other", "lon_other"))

        similarity_matrix = np.abs(np.corrcoef(x_stacked.values[:-lag], y_stacked.values[:-lag], rowvar=False))
        similarity_matrix = similarity_matrix[: len(vertex_coords), len(vertex_coords) :]

        similarity_matrix = xr.DataArray(
            similarity_matrix,
            dims=("vertex", "vertex_other"),
            coords={
                "vertex": vertex_multiindex,
                "vertex_other": vertex_other_multiindex,
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

    if is_hex:
        similarity_matrix.attrs["hex_res"] = x.attrs["hex_res"]

    return similarity_matrix


def calc_lagged_similarity_matrix(x, tau_min: int = None, tau_max: int = None):
    similarities = []
    if not tau_min:
        tau_min = int(len(x.time) / 40)
    if not tau_max:
        tau_max = int(len(x.time) / 10)
    print(f"Calculating similarity matrix for lags from {tau_min} to {tau_max}")
    for tau in progress.track(range(tau_min, tau_max + 1), total=tau_max - tau_min + 1):
        similarity = calc_similarity_matrix_with_lag(x, lag=tau)
        similarity = similarity.expand_dims(tau=[tau])
        similarities.append(similarity)

    similarities = xr.concat(similarities, dim="tau")
    similarity_matrix = similarities.max(dim="tau") / similarities.std(dim="tau")
    similarity_matrix.attrs = similarities.attrs
    return similarity_matrix


def calc_similarity_matrix_v4(x):
    """Calculate similarity matrix using Pearson correlation coefficient
    Should be the fastest version, bad scalable
    """
    is_hex = "hex_res" in x.attrs
    if is_hex:
        assert len(x.dims) == 2, "x must have 2 dimensions"
        assert "time" in x.dims, "x must have time dimension"
        assert "vertex" in x.dims, "x must have vertex dimension"
        vertex_coords = x.coords["vertex"].values

        similarity_matrix = np.abs(np.corrcoef(x, rowvar=False))
        similarity_matrix = xr.DataArray(
            similarity_matrix,
            dims=("vertex", "vertex_other"),
            coords={
                "vertex": vertex_coords.copy(),
                "vertex_other": vertex_coords.copy(),
            },
        )
    else:
        assert len(x.dims) == 3, "x must have 3 dimensions"
        assert "time" in x.dims, "x must have time dimension"
        assert "lat" in x.dims, "x must have lat dimension"
        assert "lon" in x.dims, "x must have lon dimension"

        x_stacked = x.stack(vertex=("lat", "lon")).dropna(dim="vertex", how="all")
        vertex_coords = x_stacked.coords["vertex"].values
        vertex_multiindex = pd.MultiIndex.from_tuples(vertex_coords, names=("lat", "lon"))
        vertex_other_multiindex = pd.MultiIndex.from_tuples(vertex_coords, names=("lat_other", "lon_other"))

        similarity_matrix = np.abs(np.corrcoef(x_stacked, rowvar=False))
        similarity_matrix = xr.DataArray(
            similarity_matrix,
            dims=("vertex", "vertex_other"),
            coords={
                "vertex": vertex_multiindex,
                "vertex_other": vertex_other_multiindex,
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

    if is_hex:
        similarity_matrix.attrs["hex_res"] = x.attrs["hex_res"]

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

    if "hex_res" in similarity_matrix.attrs:
        adjacency_matrix.attrs["hex_res"] = similarity_matrix.attrs["hex_res"]

    return adjacency_matrix


def calc_link_length(adj_mtx: xr.DataArray):
    assert len(adj_mtx.dims) == 2, "adj_mtx must have 2 dimensions"
    assert "vertex" in adj_mtx.dims, "adj_mtx must have vertex dimension"
    assert "vertex_other" in adj_mtx.dims, "adj_mtx must have vertex_other dimension"

    is_hex = "hex_res" in adj_mtx.attrs

    if is_hex:
        latlon_coords = [h3.h3_to_geo(str(hex(h))[2:]) for h in adj_mtx.vertex.values]
        lats_i = xr.DataArray([lat for lat, lon in latlon_coords], dims="vertex", coords={"vertex": adj_mtx.vertex})
        lons_i = xr.DataArray([lon for lat, lon in latlon_coords], dims="vertex", coords={"vertex": adj_mtx.vertex})
        latlon_coords_other = [h3.h3_to_geo(str(hex(h))[2:]) for h in adj_mtx.vertex_other.values]
        lats_j = xr.DataArray(
            [lat for lat, lon in latlon_coords_other],
            dims="vertex_other",
            coords={"vertex_other": adj_mtx.vertex_other},
        )
        lons_j = xr.DataArray(
            [lon for lat, lon in latlon_coords_other],
            dims="vertex_other",
            coords={"vertex_other": adj_mtx.vertex_other},
        )

        lats_i *= np.pi / 180
        lons_i *= np.pi / 180
        lats_j *= np.pi / 180
        lons_j *= np.pi / 180
    else:
        lats_i = adj_mtx.coords["lat"] * np.pi / 180
        lons_i = adj_mtx.coords["lon"] * np.pi / 180
        lats_j = adj_mtx.coords["lat_other"] * np.pi / 180
        lons_j = adj_mtx.coords["lon_other"] * np.pi / 180

    d_lat = lats_i - lats_j
    d_lon = lons_i - lons_j

    # Distance between two points on a sphere
    a = np.sin(d_lat / 2) ** 2 + np.cos(lats_i) * np.cos(lats_j) * np.sin(d_lon / 2) ** 2

    # Clip to avoid numerical errors
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1 - a, 0, 1)))
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

    if is_hex:
        ll_adj_mtx.attrs["hex_res"] = adj_mtx.attrs["hex_res"]

    return ll_adj_mtx
