import xarray as xr
import geopandas as gpd
import regionmask

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from pathlib import Path


# Load ocean-mask
oceans = gpd.read_file(Path(__file__).parent.parent / "data" / "ocean/ne_10m_ocean.shp")


def mask_oceans(da: xr.DataArray):
    mask = regionmask.mask_geopandas(oceans, da)
    return da.where(mask.isnull())


def mask_poles(da: xr.DataArray):
    """Only use data between 60°S and 60°N"""
    return da.where((da.lat > -60) & (da.lat < 60))


CKWARGS = dict(
    transform=ccrs.PlateCarree(),  # remember to provide this!
    subplot_kws={"projection": ccrs.PlateCarree(central_longitude=0)},
    cbar_kwargs={"orientation": "horizontal", "shrink": 0.8, "aspect": 40},
    robust=True,
)


def plot_world(da: xr.DataArray):
    f = da.plot(**CKWARGS)
    f.axes.add_feature(cfeature.BORDERS)
    plt.gca().coastlines()
    plt.gca().gridlines(draw_labels=True)
