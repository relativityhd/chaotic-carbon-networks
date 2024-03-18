from pathlib import Path
import xarray as xr
import rioxarray as rxr
from rich import print

DATA_DIR = Path(__file__).parent.parent / "data" / "population"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)


def mask_population(da: xr.DataArray, correct=False, force=False, threshold=0):
    resolution = da.lon[1] - da.lon[0]
    cached = CACHE_DIR / f"GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0_resampled_{resolution:.2f}deg.nc"
    if cached.exists() and not force:
        print(f"Loading cached population data from {cached}")
        pop = xr.open_dataarray(cached)
    else:
        pop = rxr.open_rasterio("../data/population/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif").squeeze("band")
        pop = pop.rename({"x": "lon", "y": "lat"})
        pop = pop.interp_like(da, method="nearest")
        print(f"Saving cached population data to {cached}")
        pop.to_netcdf(cached)

    if correct:
        da = da * pop
    da = da.where(pop > threshold)
    return da
