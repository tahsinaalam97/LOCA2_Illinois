#!/usr/bin/env python3

#This file was created to download historical and projected LOCA2 daily variables (min-max temp, min-max relative humidity and precipitation
#Source: https://loca.ucsd.edu/loca-version-2-for-north-america-ca-jan-2023/


#Import necessary packages
import argparse
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import xarray as xr
import pandas as pd
import time
import sys

#File directories

BASE_OUT = Path("/data/keeling/a/tahsina2/b") #Main directory- update to your own
RAW_DIR = BASE_OUT / "raw_temp" #Where full sized files will be downloaded first
IL_DIR = BASE_OUT / "LOCA2_Illinois" #Where processed Illinois bound files will be saved
TASK_FILE = BASE_OUT / "loca2_all_tasks.csv" #No and details of files to be processed

#Check if paths exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
IL_DIR.mkdir(parents=True, exist_ok=True)

#State boundary (Illinois)
LAT_MIN, LAT_MAX = 36.8, 42.6
LON_MIN, LON_MAX = -91.7, -87.0

#Resolution (Native for LOCA2)
GRID = "0p0625deg"


#LOCA2 Directory
STANDARD_ROOT = "https://cirrus.ucsd.edu/~pierce/LOCA2/CONUS_regions_split"
HUMIDITY_ROOT = "https://cirrus.ucsd.edu/~pierce/LOCA2_humidity/f7db9baa82aaa3aa80015c4d8e93b8ea"

#Since we only want Illinois, I am downloading Central Region for faster processing

REGION = "cent" #other regions or full north america is also available

#Variables

STANDARD_VARS = ["tasmax", "tasmin", "pr"]  #only central region
HUMIDITY_VARS = ["hursmax", "hursmin"] #full north america available only

#Projection Scenarios
SCENARIOS = ["historical", "ssp245", "ssp370", "ssp585"]

#Create urls for the files you need
def get_links(url):
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        return [
            a.get("href").strip("/")
            for a in soup.find_all("a")
            if a.get("href")
            and not a.get("href").startswith("?")
            and a.get("href") != "../"
            and "Parent Directory" not in a.text
        ]
    except Exception:
        return []


def list_nc_files(url):
    links = get_links(url)
    files = []
    for href in links:
        if href.endswith(".nc") and "monthly" not in href:
            files.append(url.rstrip("/") + "/" + href)
    return sorted(set(files))

#Final list and details of files to be downloaded

def build_tasks():
    tasks = []

    print("Discovering standard LOCA2 models...")
    models = get_links(STANDARD_ROOT + "/")
    models = [m for m in models if m and not m.endswith(".txt")]

    for model in models:
        ens_url = f"{STANDARD_ROOT}/{model}/{REGION}/{GRID}/"
        ensembles = get_links(ens_url)

        for ensemble in ensembles:
            for scenario in SCENARIOS:
                for variable in STANDARD_VARS:
                    durl = f"{STANDARD_ROOT}/{model}/{REGION}/{GRID}/{ensemble}/{scenario}/{variable}/"
                    files = list_nc_files(durl)

                    for url in files:
                        tasks.append({
                            "source": "standard",
                            "model": model,
                            "ensemble": ensemble,
                            "scenario": scenario,
                            "variable": variable,
                            "url": url,
                            "filename": url.split("/")[-1],
                        })

    print("Discovering LOCA2 humidity models...")
    hmodels = get_links(HUMIDITY_ROOT + "/")
    hmodels = [m for m in hmodels if m and not m.endswith(".txt")]

    for model in hmodels:
        ens_url = f"{HUMIDITY_ROOT}/{model}/{GRID}/"
        ensembles = get_links(ens_url)

        for ensemble in ensembles:
            for scenario in SCENARIOS:
                for variable in HUMIDITY_VARS:
                    durl = f"{HUMIDITY_ROOT}/{model}/{GRID}/{ensemble}/{scenario}/{variable}/"
                    files = list_nc_files(durl)

                    for url in files:
                        tasks.append({
                            "source": "humidity",
                            "model": model,
                            "ensemble": ensemble,
                            "scenario": scenario,
                            "variable": variable,
                            "url": url,
                            "filename": url.split("/")[-1],
                        })

    df = pd.DataFrame(tasks)
    df = df.drop_duplicates()
    df.to_csv(TASK_FILE, index=False)

    print(f"Saved task file: {TASK_FILE}")
    print(f"Total tasks: {len(df)}")

#Download the raw files
def download_file(url, outpath):
    outpath = Path(outpath)

    if outpath.exists() and outpath.stat().st_size > 0:
        return outpath

    tmp = outpath.with_suffix(outpath.suffix + ".part")

    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    tmp.rename(outpath)
    return outpath

#Reduce the raw files to Illinois only
def crop_to_illinois(infile, outfile):
    infile = Path(infile)
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if outfile.exists() and outfile.stat().st_size > 0:
        return

    ds = xr.open_dataset(infile)

    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"

    if float(ds[lon_name].max()) > 180:
        lon_min = LON_MIN % 360
        lon_max = LON_MAX % 360
    else:
        lon_min = LON_MIN
        lon_max = LON_MAX

    lat_vals = ds[lat_name].values
    if lat_vals[0] < lat_vals[-1]:
        lat_slice = slice(LAT_MIN, LAT_MAX)
    else:
        lat_slice = slice(LAT_MAX, LAT_MIN)

    ds_il = ds.sel({
        lat_name: lat_slice,
        lon_name: slice(lon_min, lon_max),
    })

    if ds_il.sizes.get(lat_name, 0) == 0 or ds_il.sizes.get(lon_name, 0) == 0:
        ds.close()
        raise RuntimeError(f"Empty Illinois crop: {infile}")

    encoding = {
        v: {"zlib": True, "complevel": 4}
        for v in ds_il.data_vars
    }

    ds_il.to_netcdf(outfile, encoding=encoding)

    ds.close()
    ds_il.close()


def run_task(task_id):
    if not TASK_FILE.exists():
        raise FileNotFoundError(f"Task file does not exist: {TASK_FILE}")

    df = pd.read_csv(TASK_FILE)

    if task_id < 0 or task_id >= len(df):
        raise IndexError(f"task_id {task_id} outside range 0 to {len(df)-1}")

    row = df.iloc[task_id]

    raw_file = RAW_DIR / row["filename"]

    out_file = (
        IL_DIR
        / row["variable"]
        / row["scenario"]
        / row["model"]
        / row["ensemble"]
        / row["filename"].replace(".nc", ".Illinois.nc")
    )

    print("=" * 80)
    print(f"Task ID: {task_id}")
    print(f"Variable: {row['variable']}")
    print(f"Scenario: {row['scenario']}")
    print(f"Model: {row['model']}")
    print(f"Ensemble: {row['ensemble']}")
    print(f"URL: {row['url']}")
    print(f"Output: {out_file}")
    print("=" * 80)

    if out_file.exists() and out_file.stat().st_size > 0:
        print("Already completed.")
        return

    try:
        download_file(row["url"], raw_file)
        crop_to_illinois(raw_file, out_file)

    finally:
        if raw_file.exists():
            raw_file.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-tasks", action="store_true")
    parser.add_argument("--task-id", type=int, default=None)
    args = parser.parse_args()

    if args.build_tasks:
        build_tasks()
        return

    if args.task_id is None:
        raise ValueError("Use either --build-tasks or --task-id N")

    run_task(args.task_id)


if __name__ == "__main__":
    main()