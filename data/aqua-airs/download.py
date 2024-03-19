"""
# How to Access GES DISC Data Using Python

<p></p>

<div style="background:#eeeeee; border:1px solid #cccccc;padding:5px 10px;">Please, be very judicious when working on long data time series residing on a remote data server.<br />
It is very likely that attempts to apply similar approaches on remote data, such as hourly data, for more than a year of data at a time, will result in a heavy load on the remote data server. This may lead to negative consequences, ranging from very slow performance that will be experienced by hundreds of other users, up to denial of service.</div>

### Overview

There are multiple ways to work with GES DISC data resources using Python. For example, the data can accessed using [techniques that rely on a native Python code](https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html).

Still, there are several third-party libraries that can further simplify the access. In the sections below, we describe four techniques that make use of Requests, Pydap, Xarray, and netCDF4-python libraries.

### Prerequisites

This notebook was written using Python 3.8, and requires these libraries and files:

- `netrc` file with valid Earthdata Login credentials
   - [How to Generate Earthdata Prerequisite Files](https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Generate%20Earthdata%20Prerequisite%20Files)
- [requests](https://docs.python-requests.org/en/latest/) (version 2.22.0 or later)
- [pydap](https://github.com/pydap/pydap) (we recommend using version 3.4.0 or later)
- [xarray](https://docs.xarray.dev/en/stable/)
- [netCDF4-python](https://github.com/Unidata/netcdf4-python) (we recommend using version 1.6.2)
"""

import functools
import pathlib
import shutil
import requests
from tqdm.auto import tqdm as atqdm
from multiprocessing import Pool
from tqdm import tqdm
from rich import print

RAW_DIR = pathlib.Path("./raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download(url):
    url = url.strip()
    filename = url.split("/")[-1].split("?")[0]
    print(filename.split(".")[2])
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))

    path = (RAW_DIR / filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with atqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path


if __name__ == "__main__":
    with open("subset_SNDRAQIL3CMCCP_2_20240319_201838_.txt") as f:
        urls = f.readlines()
        with Pool(12) as p:
            r = list(tqdm(p.imap(download, urls), total=len(urls)))
            print(r)
