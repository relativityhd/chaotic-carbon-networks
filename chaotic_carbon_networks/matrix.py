import numpy as np
import xarray as xr
import pandas as pd
from rich import print, progress
import h3
from typing import Callable, Optional

# Own rust library
from chaotic_carbon_networks.rust_chaotic_carbon_networks import mind, lapend

from chaotic_carbon_networks.hex import hexgrid


def assert_dims(x: xr.DataArray):
    is_hex = "hex_res" in x.attrs
    if is_hex:
        assert len(x.dims) == 2, "x must have 2 dimensions"
        assert "time" in x.dims, "x must have time dimension"
        assert "vertex" in x.dims, "x must have vertex dimension"
    else:
        assert len(x.dims) == 3, "x must have 3 dimensions"
        assert "time" in x.dims, "x must have time dimension"
        assert "lat" in x.dims, "x must have lat dimension"
        assert "lon" in x.dims, "x must have lon dimension"


def get_coords(x: xr.DataArray, is_hex: bool):
    vertex_coords: np.ndarray = x.coords["vertex"].values
    if is_hex:
        return vertex_coords.copy()
    vertex_multiindex = pd.MultiIndex.from_tuples(vertex_coords, names=("lat", "lon"))
    return vertex_multiindex


def xrmatrix_from_func(x: xr.DataArray, y: xr.DataArray, f: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    """Wraps a matrix-generation function to xarray

    Args:
        x (xr.DataArray): The DataArray of shape [t, v] or [t, lat, lon]
        y (xr.DataArray): The DataArray of shape [t, v] or [t, lat, lon]
        f (Callable): A function which generates a matrix of shape [v, v]
    """
    assert_dims(x)
    assert_dims(y)

    x_is_hex = "hex_res" in x.attrs
    y_is_hex = "hex_res" in y.attrs
    if not x_is_hex:
        x = x.stack(vertex=("lat", "lon")).dropna(dim="vertex", how="all")
    if not y_is_hex:
        y = y.stack(vertex=("lat", "lon")).dropna(dim="vertex", how="all")

    vcoords = get_coords(x, x_is_hex)
    vocoords = get_coords(y, y_is_hex)

    m = f(x.values, y.values)
    m = xr.DataArray(
        m,
        dims=("vertex", "vertex_other"),
        coords={
            "vertex": vcoords,
            "vertex_other": vocoords,
        },
    )
    if x_is_hex:
        m.coords["vertex"].attrs["hex_res"] = x.attrs["hex_res"]
    if y_is_hex:
        m.coords["vertex_other"].attrs["hex_res"] = x.attrs["hex_res"]
    return m


def mutual_information_matrix(x: xr.DataArray, y: xr.DataArray = None, bins=64):
    # Set y to x if y is none
    if y is None:
        y = x

    def f(x, y):
        return mind(x, y, bins)

    m = xrmatrix_from_func(x, y, f)
    if x.sizes == y.sizes:
        m = m.where(~np.eye(len(m), dtype=bool), 0)
    m.attrs = {
        "long_name": "Mutual Information Matrix",
        "valid_range": (0, np.inf),
        "actual_range": (m.min().item(), m.max().item()),
    }
    return m


def pearson_similarity_matrix(x: xr.DataArray):
    # TODO: Add y
    def f(x, y):
        return np.corrcoef(x, rowvar=False)

    m = xrmatrix_from_func(x, x, f)
    m = m.where(~np.eye(len(m), dtype=bool), 0)
    m.attrs = {
        "long_name": "Pearson Similarity Matrix",
        "valid_range": (0, 1),
        "actual_range": (m.min().item(), m.max().item()),
    }
    return m


def laged_pearson_similarity_matrix(x: xr.DataArray, y: xr.DataArray = None, tau_min: int = None, tau_max: int = None):
    # Set y to x if y is none
    if y is None:
        y = x

    if not tau_min:
        tau_min = int(len(x.time) / 40)
    if not tau_max:
        tau_max = int(len(x.time) / 10)
    print(f"Calculating similarity matrix for lags from {tau_min} to {tau_max}")

    def f(x, y):
        return lapend(x, tau_min, tau_max, y)

    m = xrmatrix_from_func(x, y, f)
    if x.sizes == y.sizes:
        m = m.where(~np.eye(len(m), dtype=bool), 0)
    m.attrs = {
        "long_name": "Lagged Pearson Similarity Matrix",
        "valid_range": (0, np.inf),
        "actual_range": (m.min().item(), m.max().item()),
    }
    return m


def adjacency_matrix(m: xr.DataArray, rr=0.05):
    assert len(m.dims) == 2, "m must have 2 dimensions"
    assert "vertex" in m.dims, "m must have vertex dimension"
    assert "vertex_other" in m.dims, "m must have vertex_other dimension"

    eps = np.nanquantile(m.values, 1 - rr)
    print(f"Using a threshold of {eps} for the adjacency matrix")
    adjacency_matrix = (m > eps).astype(int)
    adjacency_matrix.attrs = {
        "long_name": f"Adjacency Matrix",
        "valid_range": (0, 1),
        "actual_range": (0, 1),
    }

    return adjacency_matrix


def link_lengths_like(m: xr.DataArray):
    """Returns a Matrix with length between verticies

    Args:
        m (xr.DataArray): The Matrix

    Usage:

    ```py
    m = pearson_similarity_matrix(x)
    ll = link_lengths_like(m)
    m * ll # Length-Corrected Similarity
    ```
    """
    assert len(m.dims) == 2, "m must have 2 dimensions"
    assert "vertex" in m.dims, "m must have vertex dimension"
    assert "vertex_other" in m.dims, "m must have vertex_other dimension"

    x_hex = m.coords["vertex"].attrs.get("hex_res", False)
    y_hex = m.coords["vertex_other"].attrs.get("hex_res", False)

    if x_hex:
        latlon_coords = [h3.h3_to_geo(str(hex(h))[2:]) for h in m.vertex.values]
        lats_i = xr.DataArray([lat for lat, lon in latlon_coords], dims="vertex", coords={"vertex": m.vertex})
        lons_i = xr.DataArray([lon for lat, lon in latlon_coords], dims="vertex", coords={"vertex": m.vertex})
        lats_i *= np.pi / 180
        lons_i *= np.pi / 180
    else:
        lats_i = m.coords["lat"] * np.pi / 180
        lons_i = m.coords["lon"] * np.pi / 180

    if y_hex:
        latlon_coords_other = [h3.h3_to_geo(str(hex(h))[2:]) for h in m.vertex_other.values]
        lats_j = xr.DataArray(
            [lat for lat, lon in latlon_coords_other],
            dims="vertex_other",
            coords={"vertex_other": m.vertex_other},
        )
        lons_j = xr.DataArray(
            [lon for lat, lon in latlon_coords_other],
            dims="vertex_other",
            coords={"vertex_other": m.vertex_other},
        )
        lats_j *= np.pi / 180
        lons_j *= np.pi / 180
    else:
        lats_j = m.coords["lat_other"] * np.pi / 180
        lons_j = m.coords["lon_other"] * np.pi / 180

    d_lat = lats_i - lats_j
    d_lon = lons_i - lons_j

    # Distance between two points on a sphere
    a = np.sin(d_lat / 2) ** 2 + np.cos(lats_i) * np.cos(lats_j) * np.sin(d_lon / 2) ** 2

    # Clip to avoid numerical errors
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1 - a, 0, 1)))
    R = 6371.0088
    ll = R * c

    ll.attrs = {
        "long_name": f"Link lengths",
        "units": "km",
        "var_desc": "Link length",
        "valid_range": (0, ll.max().item()),
        "actual_range": (ll.min().item(), ll.max().item()),
    }

    if x_hex:
        ll.coords["vertex"].attrs["hex_res"] = x_hex
    if y_hex:
        ll.coords["vertex_other"].attrs["hex_res"] = y_hex

    return ll
