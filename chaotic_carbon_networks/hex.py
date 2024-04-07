import xarray as xr
import h3
import pandas as pd
from typing import Literal
import numpy as np
from numba import njit
import pickle
from shapely.geometry import Polygon

from chaotic_carbon_networks import ROOT

ResampleMethod = Literal["mean", "max", "min", "sum"]


DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "hex" / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)


def axis_is_hex(x, dim):
    if dim not in x.dims:
        return False
    return isinstance(x.coords[dim].attrs.get("hex_res", False), (int, np.int64))


def latlon_to_hex(x: xr.DataArray, hex_res: int = 2):
    lat_min, lat_max = x.lat.min().item(), x.lat.max().item()
    lon_min, lon_max = x.lon.min().item(), x.lon.max().item()
    lat_diff = round((lat_max - lat_min) / len(x.lat), 4)
    lon_diff = round((lon_max - lon_min) / len(x.lon), 4)
    cache_fname = (
        CACHE_DIR
        / f"latlon_to_hex_{hex_res:.4f}_{lat_min:.4f}_{lat_max:.4f}_{lat_diff:.4f}_{lon_min:.4f}_{lon_max:.4f}_{lon_diff:.4f}.pickle"
    )
    if cache_fname.exists():
        hex_coords = pickle.load(open(cache_fname, "rb"))
    else:
        hex_coords = [int(h3.geo_to_h3(v.lat.item(), v.lon.item(), hex_res), base=16) for v in x.vertex]
        pickle.dump(hex_coords, open(cache_fname, "wb"))
    x = x.drop_vars(["vertex", "lat", "lon"]).assign_coords(vertex=hex_coords)
    return x


def hex_to_latlon(x: xr.DataArray, final_res: int = None):
    latlon_coords = [h3.h3_to_geo(str(hex(h))[2:]) for h in x.vertex.values]
    if final_res is not None:
        latlon_coords = [(round(lat, final_res), round(lon, final_res)) for lat, lon in latlon_coords]
    mindex_obj = pd.MultiIndex.from_tuples(latlon_coords, names=("lat", "lon"))
    mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex_obj, "vertex")
    x = x.assign_coords(mindex_coords)
    return x


def h3_to_geom(x):
    h = str(hex(x.item()))[2:]
    b = h3.h3_to_geo_boundary(h, geo_json=True)

    # Check if polygon crosses the antimeridian
    lon_values = [coord[0] for coord in b]
    crosses_antimeridian = -90 > min(lon_values) and 90 < max(lon_values)
    if crosses_antimeridian:
        b = [(coords[0] + 360 if coords[0] < -90 else coords[0], coords[1]) for coords in b]

    return Polygon(b)  # if not crosses_antimeridian else None


def hexgrid(x: xr.DataArray, method: ResampleMethod = "mean", hex_res: int = 2):
    """Convert a DataArray to a hexagonal grid -> flattened DataArray with hexagonal coordinates.
    Workflow of this function:
    1. LatLon to hex
    2. Groupby hex
    3. Hex to LatLon

    Args:
        x (xr.DataArray): DataArray to convert
        method (ResampleMethod, optional): How the values of each hex-bin should be calculated. Defaults to "mean".
        hex_res (int, optional): Resolution of the hexgrid. 0 is largest. Defaults to 2.
        final_res (int, optional): Lat Lon Coordinate roundup. 0 means 1 deg resolution. 1 means 0.1 deg resolution. Defaults to 0.

    Returns:
        xr.DataArray: DataArray with hexagonal coordinates
    """
    assert len(x.dims) >= 2, "x must have at least 2 dimensions"
    assert "lat" in x.dims, "lat must be in x.dims"
    assert "lon" in x.dims, "lon must be in x.dims"

    # Stack lat and lon into a single dimension called vertex
    x_stacked = x.stack(vertex=("lat", "lon")).dropna(dim="vertex", how="all")
    # Convert lat and lon to hexagonal coordinates and assign them to the vertex dimension
    x_stacked = latlon_to_hex(x_stacked, hex_res=hex_res)

    # Aggregate the data based on the hexagonal coordinates
    if method == "sum":
        x_stacked = x_stacked.groupby("vertex").sum()
    elif method == "mean":
        x_stacked = x_stacked.groupby("vertex").mean()
    elif method == "max":
        x_stacked = x_stacked.groupby("vertex").max()
    elif method == "min":
        x_stacked = x_stacked.groupby("vertex").min()
    else:
        raise ValueError(f"Method {method} not supported")

    # x_stacked = hex_to_latlon(x_stacked, final_res=final_res)

    x_stacked.coords["vertex"].attrs["hex_res"] = hex_res

    return x_stacked


def filledgrid_from_hexgrid(x: xr.DataArray, res=1) -> xr.DataArray:
    """Creates a world-grid with lat/lon coordinates from a hexgrid. Should only be used for visualization purposes.

    Args:
        x (xr.DataArray): DataArray with hexagonal coordinates
        res (int, optional): Final resolution in degrees. Defaults to 1.

    Returns:
        _type_: _description_
    """
    assert len(x.dims) == 1, "x must have 1 dimension"
    assert "vertex" in x.dims, "vertex must be in x.dims"
    assert "hex_res" in x.coords["vertex"].attrs, "hex_res must be in x.coords['vertex'].attrs"

    hex_res = x.coords["vertex"].attrs["hex_res"]
    # hex_idx = np.array([int(h3.geo_to_h3(v.lat.item(), v.lon.item(), hex_res), base=16) for v in x.vertex])
    hex_coords = x.coords["vertex"].values
    x_vals = x.values

    @njit()
    def get_v(h: int):
        res = np.where(hex_coords == h)[0]
        return x_vals[res[0]] if len(res) > 0 else np.nan

    lats = np.arange(-90, 90, res)
    lons = np.arange(-180, 180, res)
    g = np.zeros((len(lats), len(lons)))
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            h = int(h3.geo_to_h3(lat, lon, hex_res), base=16)
            g[i, j] = get_v(h)
    da = xr.DataArray(g, coords={"lat": lats, "lon": lons}, dims=["lat", "lon"], attrs=x.attrs, name=x.name)

    return da
