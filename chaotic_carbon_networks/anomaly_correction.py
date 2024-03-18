import xarray as xr
from typing import Literal


def anomaly_correction_week(co2: xr.DataArray):
    """Corrects the GRACE anomalies for the weekly cycle"""
    co2 = co2.groupby("time.week") - co2.groupby("time.week").mean("time", keep_attrs=True)
    co2.attrs = {"units": "kgC", "long_name": "Carbon Dioxide Emissions Anomaly"}
    return co2


def anomaly_correction_month(co2: xr.DataArray):
    """Corrects the GRACE anomalies for the seasonal cycle"""
    co2 = co2.groupby("time.month") - co2.groupby("time.month").mean("time", keep_attrs=True)
    co2.attrs = {"units": "kgC", "long_name": "Carbon Dioxide Emissions Anomaly"}
    return co2


def anomaly_correction_spatial(co2: xr.DataArray):
    """Corrects the GRACE anomalies on location"""
    co2 = co2 - co2.mean("lat", keep_attrs=True).mean("lon", keep_attrs=True)
    co2.attrs = {"units": "kgC", "long_name": "Carbon Dioxide Emissions Anomaly"}
    return co2
