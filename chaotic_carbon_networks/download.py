import functools
import shutil
import requests
from tqdm.auto import tqdm as atqdm
from multiprocessing import Pool
from tqdm import tqdm
from rich import print

from chaotic_carbon_networks import ROOT

DATA_DIR = ROOT / "data" / "aqua-airs"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download(url):
    """Downloads aqua-airs data from the given URL. You must set cookies etc. first to download the data: [How to Generate Earthdata Prerequisite Files](https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Generate%20Earthdata%20Prerequisite%20Files)

    Args:
        url (str): The url from the subset file

    Raises:
        RuntimeError: Non-200 Return Code

    Returns:
        Path: Resulting Path

    Usage:

    ```py
    from multiprocessing import Pool
    from tqdm import tqdm

    subset_file = "..." # DATA_DIR / "subset_SNDRAQIL3CMCCP_2_20240319_201838_.txt"
    with open(subset_file) as f:
        urls = f.readlines()
        with Pool(12) as p:
            r = list(tqdm(p.imap(download, urls), total=len(urls)))
            print(r)
    ```
    """

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
    with open(DATA_DIR / "subset_SNDRAQIL3CMCCP_2_20240319_201838_.txt") as f:
        urls = f.readlines()
        with Pool(12) as p:
            r = list(tqdm(p.imap(download, urls), total=len(urls)))
            print(r)
