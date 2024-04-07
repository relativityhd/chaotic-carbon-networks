import numpy as np
import xarray as xr
import networkx as nx
import pandas as pd
from typing import Literal

from chaotic_carbon_networks.hex import axis_is_hex

MDIMS = Literal["vertex", "vertex_other"]


def degrees(m: xr.DataArray, dim: MDIMS = "vertex_other", weighted=True):
    dimo = "vertex" if dim == "vertex_other" else "vertex_other"
    d = m.sum(dim=dim, keep_attrs=True)

    if not axis_is_hex(d, dimo):
        d = d.unstack(dimo)

        if weighted:
            lats = d.coords["lat"]
            weights = np.cos(lats * np.pi / 180) / np.cos(lats * np.pi / 180).sum()
            d = d * weights
    elif dim == "vertex":
        d = d.rename({"vertex_other": "vertex"})

    d.attrs = {
        "long_name": "Connectivity of Vertices",
        "units": "°",
        "valid_range": (0, np.inf),
        "actual_range": (d.min().item(), d.max().item()),
    }
    return d


def average_link_length(m: xr.DataArray, ll: xr.DataArray, dim: MDIMS = "vertex_other"):
    dimo = "vertex" if dim == "vertex_other" else "vertex_other"
    mll = m * ll
    avgll = mll.where(mll > 0).mean(dim=dim)
    avgll = avgll.fillna(0)

    if not axis_is_hex(avgll, dimo):
        avgll = avgll.unstack(dimo)
    elif dim == "vertex":
        avgll = avgll.rename({"vertex_other": "vertex"})

    avgll.attrs = {
        "long_name": "Average link length of Vertices",
        "units": "km",
        "valid_range": (0, np.inf),
        "actual_range": (avgll.min().item(), avgll.max().item()),
    }

    return avgll


def betweenness(m: xr.DataArray, k: int = 100):
    # TODO: Implement directed version
    assert m.shape[0] == m.shape[1], "Expect Graph to be non-directed."

    G = nx.from_numpy_array(m.values)
    bc = nx.betweenness_centrality(G, k=k)

    vertex_coords = m.coords["vertex"]
    if axis_is_hex(m, "vertex"):
        vc = vertex_coords.values.copy()
    else:
        vc = pd.MultiIndex.from_tuples(vertex_coords.values, names=("lat", "lon"))

    b = xr.DataArray(list(bc.values()), dims="vertex", coords={"vertex": vc})

    if axis_is_hex(m, "vertex"):
        b.coords["vertex"].attrs["hex_res"] = m.coords["vertex"].attrs["hex_res"]
    else:
        b = b.unstack("vertex")

    b.attrs = {
        "long_name": "Betweenness Centrality of Vertices",
        "units": "°",
        "valid_range": (0, 1),
        "actual_range": (b.min().item(), b.max().item()),
    }

    return b
