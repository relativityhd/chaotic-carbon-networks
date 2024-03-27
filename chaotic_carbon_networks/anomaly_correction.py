import xarray as xr
from typing import Literal


def anomaly_correction_week(da: xr.DataArray):
    """Corrects the GRACE anomalies for the weekly cycle"""
    attrs = da.attrs.copy()
    da = da.groupby("time.week") - da.groupby("time.week").mean("time")
    da.attrs = attrs
    da.attrs["long_name"] += " Anomaly"
    return da


def anomaly_correction_month(da: xr.DataArray):
    """Corrects the GRACE anomalies for the seasonal cycle"""
    attrs = da.attrs.copy()
    da = da.groupby("time.month") - da.groupby("time.month").mean("time")
    da.attrs = attrs
    da.attrs["long_name"] += " Anomaly"
    return da
