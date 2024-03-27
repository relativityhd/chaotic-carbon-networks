import xarray as xr
import geopandas as gpd
import regionmask

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from pathlib import Path

from chaotic_carbon_networks.hex import filledgrid_from_hexgrid, axis_is_hex

CKWARGS = dict(
    transform=ccrs.PlateCarree(),  # remember to provide this!
    cbar_kwargs={"orientation": "horizontal", "shrink": 0.8, "aspect": 40},
    robust=True,
)


def plot_world(da: xr.DataArray):
    if axis_is_hex(da, "vertex"):
        da = filledgrid_from_hexgrid(da)

    assert "lat" in da.dims, "lat must be in da.dims"
    assert "lon" in da.dims, "lon must be in da.dims"

    f = da.plot(subplot_kws={"projection": ccrs.PlateCarree(central_longitude=0)}, **CKWARGS)
    f.axes.add_feature(cfeature.BORDERS)
    plt.gca().coastlines()
    plt.gca().gridlines(draw_labels=True)


def plot_world_to_axis(da: xr.DataArray, ax, cmap="viridis"):
    if axis_is_hex(da, "vertex"):
        da = filledgrid_from_hexgrid(da)

    assert "lat" in da.dims, "lat must be in da.dims"
    assert "lon" in da.dims, "lon must be in da.dims"

    f = da.plot(ax=ax, cmap=cmap, **CKWARGS)
    f.axes.add_feature(cfeature.COASTLINE, linewidth=2)
    f.axes.add_feature(cfeature.BORDERS, linewidth=2)
    f.axes.gridlines(draw_labels=True)


def plot_matrix(m: xr.DataArray):
    assert "vertex" in m.dims, "vertex must be in da.dims"
    assert "vertex_other" in m.dims, "vertex_other must be in da.dims"

    plt.imshow(m.values)
    plt.colorbar()
    plt.xlabel("vertex")
    plt.ylabel("vertex_other")
    plt.title(m.long_name)


def plot_matrix_to_axis(m: xr.DataArray, ax):
    assert "vertex" in m.dims, "vertex must be in da.dims"

    assert "vertex_other" in m.dims, "vertex_other must be in da.dims"
    ax.imshow(m, cmap="flare")
    ax.set_title(m.long_name)
    ax.set_xlabel("vertex")
    ax.set_ylabel("vertex_other")
