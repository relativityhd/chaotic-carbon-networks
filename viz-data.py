import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi
import seaborn as sns

from rich import pretty, traceback, print

sns.set_theme(context="paper", style="whitegrid", palette="Set2", font_scale=1.5, rc={"figure.figsize": (9, 6)})
pretty.install()
traceback.install()

CKWARGS = dict(
    transform=ccrs.PlateCarree(),  # remember to provide this!
    subplot_kws={"projection": ccrs.PlateCarree(central_longitude=0)},
    cbar_kwargs={"orientation": "horizontal", "shrink": 0.8, "aspect": 40},
    robust=True,
)

co2 = xr.open_dataset("data/xco2_c3s_l3_v42_200301_201912_2x2.nc")["xco2"]
print(co2.sizes)
# co2 = co2.interpolate_na(dim="lat", method="linear")
# co2 = co2.interpolate_na(dim="lon", method="linear")


def interpolate_2d(da):
    # Create 2D coordinates grid
    X, Y = np.meshgrid(da["lon"], da["lat"])
    # Create mask for valid values
    mask = np.isfinite(da.values)
    # Interpolate over the grid
    Z = spi.griddata((X[mask], Y[mask]), da.values[mask], (X, Y), method="linear")
    Z = xr.DataArray(Z, coords=da.coords, dims=da.dims)
    Z.attrs = da.attrs
    return Z


# co2 = co2.where((co2.count(dim="time") / 203) > 0.6)

co2 = co2.dropna(dim="time", how="all")


# attrs = co2.attrs
# co2 = co2.groupby("time").apply(interpolate_2d)
# co2.attrs = attrs

# co2 = co2.interpolate_na(dim="time", method="nearest", use_coordinate=False)
print(co2.sizes)

f = ((co2.count(dim="time") / 203) > 0.4).astype(int).plot(**CKWARGS)
f.axes.add_feature(cfeature.BORDERS)
plt.gca().coastlines()
plt.gca().gridlines(draw_labels=True)
# Wait for user input to continue
plt.draw()
plt.waitforbuttonpress()

# Clear the plot
plt.clf()


for i in co2.time:
    da = co2.sel(time=i)

    f = da.plot(**CKWARGS)
    f.axes.add_feature(cfeature.BORDERS)
    plt.gca().coastlines()
    plt.gca().gridlines(draw_labels=True)
    # Wait for user input to continue
    plt.draw()
    plt.waitforbuttonpress()

    # Clear the plot
    plt.clf()
