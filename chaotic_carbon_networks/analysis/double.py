from typing import Literal

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import seaborn as sns

from chaotic_carbon_networks.matrix import (
    laged_pearson_similarity_matrix,
    mutual_information_matrix,
    adjacency_matrix,
    link_lengths_like,
    degrees,
    average_link_length,
)
from chaotic_carbon_networks.viz import plot_matrix_to_axis, plot_world_to_axis
from chaotic_carbon_networks import ROOT


FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


ADJ_METHODS = Literal["lagged_similarity", "mutual_information"]


def plot_meanovertime(x: xr.DataArray, y: xr.DataArray, ax):
    palette = sns.color_palette("Set2", 2)
    x_is_hex = len(x.dims) == 2
    y_is_hex = len(x.dims) == 2
    if x_is_hex:
        x.mean(dim="vertex").plot(ax=ax, label=x.long_name, c=palette[0])
    else:
        x.mean(dim=["lat", "lon"]).plot(ax=ax, label=x.long_name, c=palette[0])
    if y_is_hex:
        y.mean(dim="vertex").plot(ax=ax.twinx(), label=y.long_name, c=palette[1])
    else:
        y.mean(dim=["lat", "lon"]).plot(ax=ax, label=y.long_name, c=palette[1])
    ax.legend()


def double_dataset(
    x: xr.DataArray,
    y: xr.DataArray,
    adj_method: ADJ_METHODS = "lagged_similarity",
    rr=0.05,
    saveto: str = None,
    svg=False,
):
    """Expects dataset to be already aligned and corrected"""

    x_is_hex = len(x.dims) == 2
    y_is_hex = len(x.dims) == 2

    if adj_method == "lagged_similarity":
        m = laged_pearson_similarity_matrix(x, y)
    elif adj_method == "mutual_information":
        m = mutual_information_matrix(x, y)
    else:
        raise ValueError(f"adj_method must be one of {ADJ_METHODS}")

    a = adjacency_matrix(m, rr)
    ll = link_lengths_like(a)
    avgll = average_link_length(a, ll)
    deg = degrees(a)
    dego = degrees(a, dim="vertex")
    fig = plt.figure(layout="constrained", figsize=(40, 20))
    gs = GridSpec(4, 4, figure=fig)

    ax1 = fig.add_subplot(gs[:2, :2], projection=ccrs.PlateCarree())
    plot_world_to_axis(deg, ax1, "plasma")

    ax2 = fig.add_subplot(gs[2:, 2:], projection=ccrs.PlateCarree())
    plot_world_to_axis(avgll, ax2, "cividis")

    ax3 = fig.add_subplot(gs[2:, :2], projection=ccrs.PlateCarree())
    plot_world_to_axis(dego, ax3, "viridis")

    ax4 = fig.add_subplot(gs[0, 2:])
    plot_meanovertime(x, y, ax4)

    ax5 = fig.add_subplot(gs[1, 2])
    plot_matrix_to_axis(a, ax5)

    ax6 = fig.add_subplot(gs[1, 3])
    m.where(m > max(0, m.quantile(0.01))).plot.hist(bins=100, ax=ax6)

    fig.suptitle(f"Full Network Analysis on {x.attrs['long_name']}")

    if saveto:
        if svg:
            fig.savefig(FIG_DIR / f"{saveto}.svg")
        fig.savefig(FIG_DIR / f"{saveto}.jpg")

    return fig
