from typing import Literal
from pathlib import Path

import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rich import print, inspect, traceback, pretty
import scipy.interpolate as spi
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from chaotic_carbon_networks.preprocessing import concat_airs_data
from chaotic_carbon_networks.adjencency_matrix import (
    calc_adjacency_matrix,
    calc_link_length,
    calc_similarity_matrix_v4,
    calc_lagged_similarity_matrix,
    calc_mutual_information,
)
from chaotic_carbon_networks.utils import mask_oceans, plot_world, mask_poles
from chaotic_carbon_networks.population_masking import mask_population
from chaotic_carbon_networks.anomaly_correction import (
    anomaly_correction_week,
    anomaly_correction_month,
    anomaly_correction_spatial,
)
from chaotic_carbon_networks.measures import calc_degrees, calc_avg_link_length, calc_betweenness
from chaotic_carbon_networks.hex import hexgrid, filledgrid_from_hexgrid

sns.set_theme(context="paper", style="whitegrid", palette="Set2", font_scale=1.5, rc={"figure.figsize": (9, 6)})
pretty.install()
traceback.install()

CKWARGS = dict(
    transform=ccrs.PlateCarree(),  # remember to provide this!
    # subplot_kws={"projection": ccrs.PlateCarree(central_longitude=0)},
    cbar_kwargs={"orientation": "horizontal", "shrink": 0.8, "aspect": 40},
    robust=True,
)

FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def plot_world_to_axis(da: xr.DataArray, ax, cmap="viridis"):
    if "hex_res" in da.attrs:
        da = filledgrid_from_hexgrid(da)

    assert "lat" in da.dims, "lat must be in da.dims"
    assert "lon" in da.dims, "lon must be in da.dims"

    f = da.plot(ax=ax, cmap=cmap, **CKWARGS)
    f.axes.add_feature(cfeature.COASTLINE, linewidth=2)
    f.axes.add_feature(cfeature.BORDERS, linewidth=2)
    f.axes.gridlines(draw_labels=True)


ADJ_METHODS = Literal["similarity", "lagged_similarity", "mutual_information"]


def analyse_single_dataset(x: xr.DataArray, adj_method="similarity", saveto: str = None):
    """Expects dataset to be already aligned and corrected"""

    is_hex = "hex_res" in x.attrs

    if adj_method == "similarity":
        similarity_matrix = calc_similarity_matrix_v4(x)
    elif adj_method == "lagged_similarity":
        similarity_matrix = calc_lagged_similarity_matrix(x)
    elif adj_method == "mutual_information":
        similarity_matrix = calc_mutual_information(x)
    else:
        raise ValueError(f"adj_method must be one of {ADJ_METHODS}")

    adj_mtx = calc_adjacency_matrix(similarity_matrix)
    link_length = calc_link_length(adj_mtx)
    degrees = calc_degrees(adj_mtx, weighted=(not is_hex))
    avg_link_length = calc_avg_link_length(link_length)
    betweenness = calc_betweenness(adj_mtx, k=10)
    fig = plt.figure(layout="constrained", figsize=(40, 20))
    gs = GridSpec(4, 4, figure=fig)

    ax1 = fig.add_subplot(gs[:2, :2], projection=ccrs.PlateCarree())
    plot_world_to_axis(degrees, ax1, "plasma")

    ax2 = fig.add_subplot(gs[2:, 2:], projection=ccrs.PlateCarree())
    plot_world_to_axis(avg_link_length, ax2, "cividis")

    ax3 = fig.add_subplot(gs[2:, :2], projection=ccrs.PlateCarree())
    plot_world_to_axis(betweenness, ax3, "viridis")

    ax4 = fig.add_subplot(gs[0, 2:])
    if is_hex:
        x.mean(dim=["vertex"], keep_attrs=True).plot(ax=ax4)
    else:
        x.mean(dim=["lat", "lon"], keep_attrs=True).plot(ax=ax4)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.imshow(adj_mtx, cmap="flare")
    ax5.set_title(adj_mtx.long_name)
    ax5.set_xlabel("vertex")
    ax5.set_ylabel("vertex_other")

    ax6 = fig.add_subplot(gs[1, 3])
    similarity_matrix.where(similarity_matrix > similarity_matrix.quantile(0.02)).plot.hist(bins=100, ax=ax6)

    fig.suptitle(f"Full Network Analysis on {x.attrs['long_name']}")

    if saveto:
        fig.savefig(FIG_DIR / f"{saveto}.svg")
        fig.savefig(FIG_DIR / f"{saveto}.jpg")

    return fig
