from typing import Literal

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs

from chaotic_carbon_networks.matrix import (
    pearson_similarity_matrix,
    laged_pearson_similarity_matrix,
    mutual_information_matrix,
    adjacency_matrix,
    link_lengths_like,
    degrees,
    betweenness,
    average_link_length,
)
from chaotic_carbon_networks.viz import plot_matrix_to_axis, plot_world_to_axis
from chaotic_carbon_networks import ROOT


FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


ADJ_METHODS = Literal["similarity", "lagged_similarity", "mutual_information"]


def single_dataset(x: xr.DataArray, adj_method: ADJ_METHODS = "similarity", rr=0.05, saveto: str = None, svg=False):
    """Expects dataset to be already aligned and corrected"""

    is_hex = len(x.dims) == 2

    if adj_method == "similarity":
        m = pearson_similarity_matrix(x)
    elif adj_method == "lagged_similarity":
        m = laged_pearson_similarity_matrix(x)
    elif adj_method == "mutual_information":
        m = mutual_information_matrix(x, bins=32)
    else:
        raise ValueError(f"adj_method must be one of {ADJ_METHODS}")

    a = adjacency_matrix(m, rr)
    ll = link_lengths_like(a)
    avgll = average_link_length(a, ll)
    deg = degrees(a)
    bc = betweenness(a, k=100)
    fig = plt.figure(layout="constrained", figsize=(40, 20))
    gs = GridSpec(4, 4, figure=fig)

    ax1 = fig.add_subplot(gs[:2, :2], projection=ccrs.PlateCarree())
    plot_world_to_axis(deg, ax1, "plasma")

    ax2 = fig.add_subplot(gs[2:, 2:], projection=ccrs.PlateCarree())
    plot_world_to_axis(avgll, ax2, "cividis")

    ax3 = fig.add_subplot(gs[2:, :2], projection=ccrs.PlateCarree())
    plot_world_to_axis(bc, ax3, "viridis")

    ax4 = fig.add_subplot(gs[0, 2:])
    if is_hex:
        x.mean(dim=["vertex"], keep_attrs=True).plot(ax=ax4)
    else:
        x.mean(dim=["lat", "lon"], keep_attrs=True).plot(ax=ax4)

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
