#!/usr/bin/env python3
"""St-Pierre et al. (2003): daily county THI for Illinois.

Converted from the supplied St Pierre (2).ipynb.
Run with St_Pierre.slurm or python -u St_Pierre.py.
Keep loca2_common.py in the same directory.

Source paths, analysis periods and THI calculations follow the notebook.
RHmin > RHmax pairs are masked to NaN, as in the supplied notebook.
Existing NetCDF and optional CSV outputs are skipped (OVERWRITE=False).
Interactive plots and the one-off RH diagnostic are omitted from batch execution.
Dependencies: numpy, pandas, xarray, dask, netCDF4, cftime, geopandas,
regionmask and tqdm, installed in the pyWBM environment.
"""


from pathlib import Path
from contextlib import ExitStack
from datetime import datetime, timezone
import calendar
import json
import os
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import loca2_common as common

INPUT_ROOT = Path('/data/keeling/a/tahsina2/b/Climate_Impact/Data/LOCA2_Illinois')
OUTPUT_ROOT = INPUT_ROOT / 'St Pierre (2003)'
COUNTY_SHAPEFILE = INPUT_ROOT.parent / 'Illinois_County_Boundaries' / 'IL_BNDY_County_Py.shp'
SCENARIOS = ('ssp245', 'ssp370')
PERIODS = {
    'historical_1985_2014': (1985, 2014),
    'near_2015_2044': (2015, 2044),
    'mid_2035_2064': (2035, 2064),
    'late_2065_2094': (2065, 2094),
}
OVERWRITE = False
EXPORT_CSV = False  # Set True to export one compressed long table per run.
RUN_LIMIT = None    # Set to 1 for an initial trial, then None for all runs.


import argparse
import gzip
import dask
import geopandas as gpd

def load_counties():
    if not COUNTY_SHAPEFILE.is_file():
        raise FileNotFoundError(COUNTY_SHAPEFILE)
    counties = gpd.read_file(COUNTY_SHAPEFILE).to_crs('EPSG:4326').reset_index(drop=True)
    fips = counties['CO_FIPS'].astype(int).astype(str).str.zfill(3)
    counties['county_id'] = fips.where(fips.str.len() == 5, '17' + fips)
    counties['county_name'] = counties['COUNTY_NAM'].astype(str).str.strip()
    if len(counties) != 102 or not counties['county_id'].is_unique:
        raise ValueError('Expected 102 unique Illinois counties.')
    if not counties['county_id'].str.match(r'^17\d{3}$').all():
        raise ValueError('County identifiers must be five-digit Illinois FIPS codes.')
    return counties



def daily_files(folder):
    excluded = ('.monthly.', '.yearly.', '.annual.', '.seasonal.', '.us_county_avg.')
    return [p for p in sorted(folder.glob('*.nc'))
            if not any(tag in p.name.lower() for tag in excluded)]

def discover_temperature_runs():
    runs = set()
    for scenario in SCENARIOS:
        for variable in ('tasmin', 'tasmax'):
            root = INPUT_ROOT / variable / scenario
            if root.is_dir():
                for member_dir in root.glob('*/*'):
                    if member_dir.is_dir() and daily_files(member_dir):
                        runs.add((member_dir.parent.name, member_dir.name, scenario))
    return sorted(runs)


ALIASES = {'tasmin': ('tasmin',), 'tasmax': ('tasmax',),
           'hursmin': ('hursmin', 'rel_humid_min'),
           'hursmax': ('hursmax', 'rel_humid_max')}

def normalize_daily_time(data):
    if 'time' not in data.dims or data.sizes['time'] == 0:
        raise ValueError('Empty or missing daily time axis')
    if np.issubdtype(data.time.dtype, np.datetime64):
        dates = data.time.values.astype('datetime64[D]')
    else:
        dates = [t.replace(hour=0, minute=0, second=0, microsecond=0)
                 for t in data.time.values]
    data = data.assign_coords(time=dates).sortby('time')
    if not data.indexes['time'].is_unique:
        raise ValueError('Duplicate calendar dates in daily source; inspect overlapping files')
    return data

def convert_source(data, variable):
    original_units = str(data.attrs.get('units', ''))
    units = original_units.strip().lower().replace(' ', '').replace('_', '')
    attrs = dict(data.attrs)
    if variable in ('tasmin', 'tasmax'):
        if units in {'k', 'kelvin', 'degreek', 'degreesk'}:
            result = data - 273.15
            conversion = 'Kelvin to Celsius: subtract 273.15'
        elif units in {'c', 'degc', 'degreecelsius', 'degreescelsius', 'celsius', '°c'}:
            result = data
            conversion = 'Already Celsius'
        else:
            raise ValueError(f'{variable}: unsupported units {original_units!r}')
        output_units = 'degree_Celsius'
    else:
        if units in {'%', 'percent', 'percentage'}:
            result = data
            conversion = 'Already percent'
        elif units in {'1', 'fraction', '0-1'}:
            result = data * 100.0
            conversion = 'Fraction to percent: multiply by 100'
        else:
            raise ValueError(f'{variable}: unsupported RH units {original_units!r}')
        output_units = '%'
    result.attrs = attrs
    result.attrs.update(units=output_units, original_units=original_units, conversion=conversion)
    return result.where(np.isfinite(result))

def open_experiment(stack, variable, model, member, experiment, start, end,
                    manifest, missing):
    folder = INPUT_ROOT / variable / experiment / model / member
    paths = daily_files(folder)
    if not paths:
        if variable.startswith('hurs'):
            missing.append(f'{variable}/{experiment}: no daily files at {folder}')
            return None
        raise FileNotFoundError(f'Missing required temperature files: {folder}')
    pieces = []
    for path in paths:
        ds = stack.enter_context(xr.open_dataset(path, chunks={'time': 366}, decode_timedelta=False))
        name = next((n for n in ALIASES[variable] if n in ds.data_vars), None)
        if name is None:
            raise KeyError(f'{path.name}: expected {ALIASES[variable]}, found {list(ds.data_vars)}')
        data = ds[name]
        rename = {n: target for n, target in [('latitude', 'lat'), ('longitude', 'lon')]
                  if n in data.dims}
        data = data.rename(rename)
        if set(data.dims) != {'time', 'lat', 'lon'}:
            raise ValueError(f'{path.name}: expected time/lat/lon, found {data.dims}')
        data = data.sel(time=(data.time.dt.year >= start) & (data.time.dt.year <= end))
        if data.sizes['time'] == 0:
            continue
        data = convert_source(data, variable).rename(variable).sortby('lat').sortby('lon')
        pieces.append(data)
        manifest.append({'variable': variable, 'file': str(path),
                         'original_units': ds[name].attrs.get('units', ''),
                         'conversion': data.attrs['conversion']})
    if not pieces:
        if variable.startswith('hurs'):
            missing.append(f'{variable}/{experiment}: no records in {start}-{end}')
            return None
        raise ValueError(f'{variable}/{experiment}: no records in {start}-{end}')
    return normalize_daily_time(xr.concat(pieces, dim='time', join='exact',
                                         coords='minimal', compat='equals'))

def expected_days(year, calendar_name):
    if calendar_name == '360_day':
        return 360
    if calendar_name in ('noleap', '365_day'):
        return 365
    if calendar_name in ('all_leap', '366_day'):
        return 366
    if calendar_name == 'julian':
        return 366 if year % 4 == 0 else 365
    return 366 if calendar.isleap(year) else 365

def validate_temperature_time(data):
    cal = data.time.dt.calendar
    for year in range(1985, 2095):
        count = int((data.time.dt.year == year).sum().item())
        if count != expected_days(year, cal):
            raise ValueError(f'{data.name}: {year} has {count}/{expected_days(year, cal)} days')

def match_humidity(data, template, variable, missing):
    if data is None:
        return xr.full_like(template, np.nan, dtype='float64').rename(variable)
    if data.time.dt.calendar != template.time.dt.calendar:
        raise ValueError(f'{variable}: humidity and temperature calendars differ')
    for coord in ('lat', 'lon'):
        if not data[coord].equals(template[coord]):
            raise ValueError(f'{variable}: {coord} grid differs from temperature')
    missing_days = len(template.indexes['time'].difference(data.indexes['time']))
    if missing_days:
        missing.append(f'{variable}: {missing_days} dates absent in humidity segment')
    return data.reindex(time=template.time)

def calculate_thi(temperature_c, rh_percent):
    return ((1.8 * temperature_c + 32.0)
            - (0.55 - 0.0055 * rh_percent) * (1.8 * temperature_c - 26.0))

def create_result(model, member, scenario):
    manifest, missing, annual_parts = [], [], []
    with ExitStack() as stack:
        temperatures = {}
        for variable in ('tasmin', 'tasmax'):
            segments = [open_experiment(stack, variable, model, member, exp, lo, hi,
                                        manifest, missing)
                        for exp, lo, hi in [('historical', 1985, 2014), (scenario, 2015, 2094)]]
            temperatures[variable] = normalize_daily_time(xr.concat(
                segments, dim='time', join='exact', coords='minimal', compat='equals'))
            validate_temperature_time(temperatures[variable])
        tmin, tmax = xr.align(temperatures['tasmin'], temperatures['tasmax'], join='exact')
        humidity = {}
        for variable in ('hursmin', 'hursmax'):
            parts = []
            for experiment, lo, hi in [('historical', 1985, 2014), (scenario, 2015, 2094)]:
                template = tmin.sel(time=(tmin.time.dt.year >= lo) & (tmin.time.dt.year <= hi))
                source = open_experiment(stack, variable, model, member, experiment,
                                         lo, hi, manifest, missing)
                parts.append(match_humidity(source, template, variable, missing))
            humidity[variable] = xr.concat(parts, dim='time', join='exact')

        for year in range(1985, 2095):
            print(f'PROCESSING {model}/{member}/{scenario}: {year}', flush=True)
            yearly = xr.Dataset({'tasmin': tmin, 'tasmax': tmax, **humidity}).sel(
                time=tmin.time.dt.year == year).load()
            
            # Boundary checks for temperature and relative humidity
            for variable, lower, upper in [('tasmin', -100, 70), ('tasmax', -100, 70),
                                            ('hursmin', 0, 100), ('hursmax', 0, 100)]:
                invalid = (yearly[variable] < lower) | (yearly[variable] > upper)
                if bool(invalid.any().item()):
                    raise ValueError(f'{variable}: values outside {lower}..{upper} in {year}')
            
            if bool((yearly.tasmin > yearly.tasmax).any().item()):
                raise ValueError(f'Tmin exceeds Tmax in {year}')

            # --- SAFE HUMIDITY MASKING ---
            # Mask out RHmin > RHmax grid cells to NaN without altering coordinates/attrs
            rh_inversion = yearly.hursmin > yearly.hursmax
            if bool(rh_inversion.any().item()):
                inv_count = int(rh_inversion.sum().item())
                missing.append(
                    f'hursmin > hursmax inversion detected in {year} ({inv_count} points); masked to NaN.'
                )
                
                # Use xr.where with keep_attrs=True to prevent dropping metadata/coords
                yearly['hursmin'] = xr.where(~rh_inversion, yearly.hursmin, np.nan, keep_attrs=True)
                yearly['hursmax'] = xr.where(~rh_inversion, yearly.hursmax, np.nan, keep_attrs=True)

            grid = xr.Dataset({
                'THI_max': calculate_thi(yearly.tasmax, yearly.hursmin),
                'THI_min': calculate_thi(yearly.tasmin, yearly.hursmax),
            })
            
            for name in ('THI_max', 'THI_min'):
                grid[f'{name}_valid_area_fraction'] = np.isfinite(grid[name]).astype('float64')
                
            county = common.aggregate_to_illinois_counties(grid, counties=counties)
            annual_parts.append(county.transpose('time', 'county').load())

    result = xr.concat(annual_parts, dim='time', join='exact')
    for name, pairing in [('THI_max', 'Tmax with RHmin'), ('THI_min', 'Tmin with RHmax')]:
        result[name].attrs.update(units='1', display_units='THI points',
            long_name=f'Daily county-average {name}; {pairing}',
            formula='(1.8*T+32)-(0.55-0.0055*RH)*(1.8*T-26)',
            formula_inputs='T in degrees Celsius; RH in percent', pairing=pairing)
        result[f'{name}_valid_area_fraction'].attrs.update(
            units='1', long_name=f'Fraction of county grid-cell area with valid {name}',
            valid_min=0.0, valid_max=1.0)
            
    membership = np.array([
        ((result.time.dt.year >= lo) & (result.time.dt.year <= hi)).values
        for lo, hi in PERIODS.values()], dtype='int8')
    result['period_membership'] = xr.DataArray(membership, dims=('period', 'time'),
        coords={'period': list(PERIODS), 'time': result.time},
        attrs={'units': '1', 'long_name': '1 when date belongs to analysis period; 0 otherwise'})
    result['calendar_year'] = result.time.dt.year.astype('int16')
    result['calendar_year'].attrs['long_name'] = 'Source-calendar year'
    result.attrs.update(paper='St-Pierre, Cobanov & Schnitkey (2003)',
        model=model, member=member, scenario=scenario, temporal_resolution='daily',
        baseline_period='1985-2014', analysis_periods=json.dumps(PERIODS),
        source_calendar=tmin.time.dt.calendar,
        thi_method='Requested daily temperature/humidity-extreme pairing; not hourly THI extrema',
        missing_humidity=json.dumps(missing), source_manifest=json.dumps(manifest),
        missing_data_policy='Missing humidity and RHmin > RHmax pairs retained as NaN; available cells used for county means',
        temperature_conversion='Source Kelvin converted to Celsius before THI calculation',
        heat_load='Not calculated',
        created_utc=datetime.now(timezone.utc).isoformat())
        
    inversions = int((result.THI_max < result.THI_min).sum().item())
    result.attrs['county_day_thi_inversions'] = inversions
    if inversions:
        print(f'NOTE: {inversions} county-days have THI_max < THI_min; retained without swapping.')
    for message in missing:
        print('HUMIDITY NaN:', message)
    return result

def write_result(result, path):
    """Validate a temporary NetCDF, then publish without overwriting existing data."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp.nc', delete=False) as handle:
            temporary = Path(handle.name)
        encoding = {name: {'zlib': True, 'complevel': 4, 'shuffle': True}
                    for name in result.data_vars}
        result.to_netcdf(temporary, engine='netcdf4', encoding=encoding)
        with xr.open_dataset(temporary) as check:
            if check.sizes.get('county') != 102 or check.sizes['time'] != result.sizes['time']:
                raise ValueError('Written output has unexpected dimensions')
            for name in ('THI_max', 'THI_min'):
                xr.testing.assert_allclose(check[name], result[name])
        if OVERWRITE:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def export_csv():
    if True:
        for path in tqdm(sorted(OUTPUT_ROOT.glob('St_Pierre_2003__*.nc')), desc='Export CSV'):
            target = path.with_suffix('.csv.gz')
            if target.exists() and not OVERWRITE:
                print('SKIPPED:', target.name)
                continue
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.tmp', delete=False) as handle:
                    temporary = Path(handle.name)
                with xr.open_dataset(path) as ds, gzip.open(temporary, 'wt', newline='') as handle:
                    first = True
                    for period, (lo, hi) in PERIODS.items():
                        for year in range(lo, hi + 1):
                            names = ['THI_max', 'THI_min', 'THI_max_valid_area_fraction',
                                     'THI_min_valid_area_fraction']
                            block = ds[names].sel(time=ds.time.dt.year == year)
                            block = block.assign_coords(time=block.time.dt.strftime('%Y-%m-%d').values)
                            table = block.to_dataframe().reset_index().rename(columns={'time': 'date'})
                            for key in ('model', 'member', 'scenario'):
                                table[key] = ds.attrs[key]
                            table['run'] = f"{ds.attrs['model']}_{ds.attrs['member']}_{ds.attrs['scenario']}"
                            table['period'] = period
                            table['calendar'] = ds.attrs['source_calendar']
                            columns = ['run', 'model', 'member', 'scenario', 'period', 'calendar',
                                       'county_id', 'county_name', 'date'] + names
                            table[columns].to_csv(handle, index=False, header=first, na_rep='NaN')
                            first = False
                if OVERWRITE:
                    os.replace(temporary, target)
                else:
                    os.link(temporary, target)
                    temporary.unlink()
                print('WROTE:', target.name)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
    else:
        print('CSV export disabled. Set EXPORT_CSV = True to enable it.')


def main():
    global counties, RUN_LIMIT
    parser = argparse.ArgumentParser(description='Daily Illinois county THI from LOCA2.')
    parser.add_argument('--run-limit', type=int, default=RUN_LIMIT,
                        help='Process the first N discovered runs for a trial.')
    parser.add_argument('--export-csv', action='store_true', default=EXPORT_CSV,
                        help='Also export compressed long-format CSV tables.')
    args = parser.parse_args()
    if args.run_limit is not None and args.run_limit < 1:
        parser.error('--run-limit must be at least 1')
    RUN_LIMIT = args.run_limit
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = discover_temperature_runs()
    if not jobs:
        raise FileNotFoundError(f'No temperature runs under {INPUT_ROOT}')
    print('Input root:', INPUT_ROOT, flush=True)
    print('Output root:', OUTPUT_ROOT, flush=True)
    print('Overwrite:', OVERWRITE, flush=True)
    print('Temperature runs discovered:', len(jobs), flush=True)
    print(pd.DataFrame(jobs, columns=['model', 'member', 'scenario'])
          .groupby('scenario').size().rename('runs').to_string(), flush=True)
    counties = load_counties()
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    dask.config.set(scheduler='threads', num_workers=workers)
    completed = skipped = 0
    failures = []
    selected_jobs = jobs if RUN_LIMIT is None else jobs[:RUN_LIMIT]
    for model, member, scenario in tqdm(selected_jobs, desc='St Pierre daily THI', unit='run'):
        path = OUTPUT_ROOT / f'St_Pierre_2003__{model}_{member}_{scenario}.nc'
        if path.exists() and not OVERWRITE:
            skipped += 1
            print('SKIPPED:', path.name)
            continue
        result = None
        try:
            result = create_result(model, member, scenario)
            write_result(result, path)
            completed += 1
            print('WROTE:', path.name)
        except Exception as exc:
            failures.append({'model': model, 'member': member, 'scenario': scenario,
                             'error': f'{type(exc).__name__}: {exc}'})
            print('FAILED:', model, member, scenario, failures[-1]['error'])
        finally:
            if result is not None:
                result.close()
    print(f'Completed={completed}; skipped={skipped}; failed={len(failures)}')
    if failures:
        print(pd.DataFrame(failures).to_string(index=False))
    if args.export_csv:
        export_csv()
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
