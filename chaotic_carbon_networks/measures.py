import numpy as np
import xarray as xr


def calc_degrees(adj_mtx: xr.DataArray, weighted=True):
    """Calculate degrees of vertices in adjacency matrix"""
    degrees = adj_mtx.sum(dim="vertex_other", keep_attrs=True).unstack("vertex")

    if weighted:
        lats = degrees.coords["lat"]
        weights = np.cos(lats * np.pi / 180) / np.cos(lats * np.pi / 180).sum()
        degrees = degrees * weights
        # degrees = degrees.where(degrees > 0, 0)
        degrees.attrs = {
            "long_name": "Area-weighted connectivity of vertices",
            "var_desc": "Area-weighted connectivity",
            "valid_range": (0, 1),
            "actual_range": (degrees.min().item(), degrees.max().item()),
        }
    else:
        degrees.attrs = {
            "long_name": "Connectivity of vertices",
            "var_desc": "Connectivity",
            "units": "°",
            "valid_range": (0, np.inf),
            "actual_range": (degrees.min().item(), degrees.max().item()),
        }

    return degrees


def calc_avg_link_length(ll_adj_mtx: xr.DataArray, weighted=False):
    """Calculate average link length of vertices in adjacency matrix"""

    # This is experimental. The idea behind it is the following: The average link length is calculated as the mean of all the link-lengths of a vertex. However, this does not take into account the fact that some vertices are more closer to each other than others. E.g. a cluster in one corner and a few points on the other side of the map. To encounter this, I try to add weights to the link-lengths. Each weight represents the pot. average distance of a vertex to all other vertices. This weight gets area-corrected. This is then multiplied with the link-lengths and the mean is calculated.
    if weighted:
        lats_i = ll_adj_mtx.coords["lat"] * np.pi / 180
        lons_i = ll_adj_mtx.coords["lon"] * np.pi / 180
        lats_j = ll_adj_mtx.coords["lat_other"] * np.pi / 180
        lons_j = ll_adj_mtx.coords["lon_other"] * np.pi / 180

        d_lat = lats_i - lats_j
        d_lon = lons_i - lons_j

        # Distance between two points on a sphere
        a = np.sin(d_lat / 2) ** 2 + np.cos(lats_i) * np.cos(lats_j) * np.sin(d_lon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        R = 6371.0088
        d = R * c

        weights = np.cos(lats_i) / np.cos(lats_i).sum()
        weights = weights * d
        # weights = weights.mean("vertex_other").unstack("vertex")
        weights /= weights.max()

        weighted_matrix = ll_adj_mtx.where(ll_adj_mtx > 0) * weights

        avg_link_length = weighted_matrix.mean(dim="vertex_other", keep_attrs=True).unstack("vertex")
        avg_link_length.attrs = {
            "long_name": "Corrected Average link length of vertices",
            "var_desc": "Corrected Average link length",
            "units": "km",
            "valid_range": (0, np.inf),
            "actual_range": (avg_link_length.min().item(), avg_link_length.max().item()),
        }
        return avg_link_length

    avg_link_length = ll_adj_mtx.where(ll_adj_mtx > 0).mean(dim="vertex_other", keep_attrs=True).unstack("vertex")
    avg_link_length.attrs = {
        "long_name": "Average link length of vertices",
        "var_desc": "Average link length",
        "units": "km",
        "valid_range": (0, np.inf),
        "actual_range": (avg_link_length.min().item(), avg_link_length.max().item()),
    }

    return avg_link_length
