import xarray as xr
import geopandas as gpd
import regionmask

import matplotlib.cm as cm
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from pathlib import Path

from chaotic_carbon_networks.hex import filledgrid_from_hexgrid, axis_is_hex, h3_to_geom

crs_epsg = ccrs.PlateCarree(central_longitude=0)

CKWARGS = dict(
    transform=crs_epsg,  # remember to provide this!
    cbar_kwargs={"orientation": "horizontal", "shrink": 0.9, "aspect": 40},
    robust=True,
)


def plot_world(da: xr.DataArray):
    if axis_is_hex(da, "vertex"):
        # Generate a figure with two axes, one for CartoPy, one for GeoPandas
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": crs_epsg})
        plot_world_to_axis(da, ax)
        return

    assert "lat" in da.dims, "lat must be in da.dims"
    assert "lon" in da.dims, "lon must be in da.dims"

    f = da.plot(subplot_kws={"projection": crs_epsg}, **CKWARGS)
    f.axes.add_feature(cfeature.BORDERS)
    plt.gca().coastlines()
    plt.gca().gridlines(draw_labels=True)


def plot_world_to_axis(da: xr.DataArray, ax, cmap="viridis", nocbar=False):
    if axis_is_hex(da, "vertex"):
        cmap = cm.get_cmap(cmap)
        norm = colors.Normalize(vmin=da.quantile(0.02), vmax=da.quantile(0.98))
        rgba = cmap(norm(da.values.tolist()))
        c = [colors.rgb2hex(c) for c in rgba]
        geoms = [h3_to_geom(h) for h in da.vertex]

        ax.add_geometries(geoms, crs=crs_epsg, facecolors=c, linewidth=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        ax.gridlines(draw_labels=True)
        ax.set_extent([-180, 180, -83, 83])
        title = None
        if "long_name" in da.attrs:
            title = da.long_name
            if "units" in da.attrs:
                title += f" [{da.attrs['units']}]"
        if not nocbar:
            plt.colorbar(
                cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                orientation="horizontal",
                shrink=0.9,
                extend="both",
                aspect=40,
                label=title,
            )
        return

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
