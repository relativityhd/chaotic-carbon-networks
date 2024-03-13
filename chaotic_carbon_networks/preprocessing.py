from pathlib import Path
from rich.progress import track
from rich import print
import xarray as xr
from typing import Literal


DATA_DIR = Path(__file__).parent.parent / "data" / "graced"
RAW_DIR = DATA_DIR / "original"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

ResampleMethod = Literal["mean", "max", "min", "sum"]
CorrectMethod = Literal["month", "week", "weekday"]


def get_cached_fname(resample: int = None, method: ResampleMethod = "mean", correctby: CorrectMethod = None):
    fname = "co2_concat"
    if resample:
        fname += f"_{resample}x{resample}_{method}ed"
    if correctby:
        fname += f"_{correctby}ly"
    fname += ".nc"
    return fname


def concat_data(
    resample: int = None, method: ResampleMethod = "mean", correctby: CorrectMethod = None, force=False
) -> xr.DataArray:
    fname = get_cached_fname(resample, method, correctby)
    cached = CACHE_DIR / fname
    if cached.exists() and not force:
        print(f"Loading cached data from {cached}")
        return xr.open_dataarray(cached)

    files = list(RAW_DIR.glob("*.nc"))
    arrays = []
    for file in track(files):
        da = xr.open_dataarray(file)
        if resample:
            da = da.coarsen(latitude=resample, longitude=resample, boundary="trim")
            if method == "mean":
                da = da.mean()
            elif method == "max":
                da = da.max()
            elif method == "min":
                da = da.min()
            elif method == "sum":
                da = da.sum()
        arrays.append(da)
    co2 = xr.concat(arrays, dim="nday")
    co2 = co2.rename({"latitude": "lat", "longitude": "lon", "nday": "time"})
    co2 = co2.sortby("time")

    # Make anomaly correction
    unit = "d kgC/h"
    name = "Carbon Dioxide Emissions Anomaly"
    if correctby == "month":
        co2 = co2.groupby("time.month") - co2.groupby("time.month").mean("time", keep_attrs=True)
    elif correctby == "week":
        co2 = co2.groupby("time.week") - co2.groupby("time.week").mean("time", keep_attrs=True)
    elif correctby == "weekday":
        co2 = co2.groupby("time.weekday") - co2.groupby("time.weekday").mean("time", keep_attrs=True)
    else:
        # No correction
        unit = "kgC/h"
        name = "Carbon Dioxide Emissions"
    co2.attrs = {"units": unit, "long_name": name}
    co2.name = fname.strip(".nc")

    print(f"Saving data to {cached}")
    co2.to_netcdf(cached)

    return co2
