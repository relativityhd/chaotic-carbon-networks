import xarray as xr
import geopandas as gpd
import regionmask

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from pathlib import Path

from chaotic_carbon_networks.hex import filledgrid_from_hexgrid, hexgrid


# Load ocean-mask
oceans = gpd.read_file(Path(__file__).parent.parent / "data" / "ocean/ne_10m_ocean.shp")


def mask_oceans(da: xr.DataArray):
    mask = regionmask.mask_geopandas(oceans, da)
    return da.where(mask.isnull())


def mask_poles(da: xr.DataArray, deg: int = 60):
    """Only use data between 60°S and 60°N"""
    return da.where((da.lat > -deg) & (da.lat < deg))


CKWARGS = dict(
    transform=ccrs.PlateCarree(),  # remember to provide this!
    subplot_kws={"projection": ccrs.PlateCarree(central_longitude=0)},
    cbar_kwargs={"orientation": "horizontal", "shrink": 0.8, "aspect": 40},
    robust=True,
)


def plot_world(da: xr.DataArray):
    if "hex_res" in da.attrs:
        da = filledgrid_from_hexgrid(da)

    assert "lat" in da.dims, "lat must be in da.dims"
    assert "lon" in da.dims, "lon must be in da.dims"

    f = da.plot(**CKWARGS)
    f.axes.add_feature(cfeature.BORDERS)
    plt.gca().coastlines()
    plt.gca().gridlines(draw_labels=True)


def plot_matrix(da: xr.DataArray):
    assert "vertex" in da.dims, "vertex must be in da.dims"
    assert "vertex_other" in da.dims, "vertex_other must be in da.dims"

    plt.imshow(da.values)
    plt.colorbar()
    plt.xlabel("vertex")
    plt.ylabel("vertex_other")
    plt.title(da.long_name)
