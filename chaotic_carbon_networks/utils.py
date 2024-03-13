import xarray as xr
import geopandas as gpd
import regionmask
import rioxarray as rxr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from pathlib import Path
from rich import print


# Load ocean-mask
oceans = gpd.read_file(Path(__file__).parent.parent / "data" / "ocean/ne_10m_ocean.shp")


def mask_oceans(da: xr.DataArray):
    mask = regionmask.mask_geopandas(oceans, da)
    return da.where(mask.isnull())


DATA_DIR = Path(__file__).parent.parent / "data" / "population"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)


def mask_population(da: xr.DataArray, correct=False, force=False):
    cached = CACHE_DIR / "GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0_resampled.nc"
    if cached.exists() and not force:
        print(f"Loading cached data from {cached}")
        pop = xr.open_dataarray(cached)
    else:
        pop = rxr.open_rasterio("../data/population/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif").squeeze("band")
        pop = pop.rename({"x": "lon", "y": "lat"})
        pop = pop.interp_like(da, method="nearest")
        print(f"Saving cached data to {cached}")
        pop.to_netcdf(cached)

    if correct:
        da = da * pop
    da = da.where(pop > 10)
    return da


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
