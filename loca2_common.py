"""Shared LOCA2 Illinois loaders, indicators, county aggregation, and writer.
"""
from __future__ import annotations

import gc
import glob
import re
from pathlib import Path
import numpy as np
import xarray as xr
from tqdm.auto import tqdm


PERIODS = {'historical_1985_2014': (1985, 2014), 'near_2015_2044': (2015, 2044), 'mid_2035_2064': (2035, 2064), 'late_2065_2094': (2065, 2094)} #30 year periods for analysis
SCENARIOS = ('ssp245', 'ssp370') #Shared socioeconomic pathways. ssp585 is also available
PAPER_INPUT_ROOT = Path('/data/keeling/a/tahsina2/b/Climate_Impact/Data/LOCA2_Illinois') #Set accordingly
DATA_ROOT = Path('/data/keeling/a/tahsina2/b/Climate_Impact/Data/') #Set accordingly
PAPER_OUTPUT_ROOT = PAPER_INPUT_ROOT/'temp'
PAPER_SCENARIOS = SCENARIOS
SEASON_START = '03-01' 
SEASON_END = '08-31'


COUNTY_SHAPEFILE = DATA_ROOT / "Illinois_County_Boundaries" / "IL_BNDY_County_Py.shp"

def load_counties():
    """Load and validate the existing Illinois county shapefile."""
    import geopandas as gpd
    counties = gpd.read_file(COUNTY_SHAPEFILE).to_crs("EPSG:4326").reset_index(drop=True)
    if len(counties) != 102:
        raise ValueError(f"Expected 102 Illinois counties, found {len(counties)}.")
    if counties["COUNTY_NAM"].isna().any():
        raise ValueError("At least one county name is missing.")
    counties["county_id"] = "17" + counties["CO_FIPS"].astype(int).astype(str).str.zfill(3)
    counties["county_name"] = counties["COUNTY_NAM"].astype(str).str.strip()
    if not counties["county_id"].is_unique:
        raise ValueError("County FIPS identifiers are not unique.")
    return counties


def validate_month_day(value: str, argument: str) -> tuple[int, int]:
    """Parse and validate MM-DD without assuming a model calendar."""
    match = re.fullmatch('(\\d{2})-(\\d{2})', value)
    if not match:
        raise ValueError(f'{argument} must use MM-DD format; received {value!r}')
    month, day = map(int, match.groups())
    days_by_month = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    if month < 1 or month > 12 or day < 1 or (day > days_by_month[month - 1]):
        raise ValueError(f'Invalid {argument} date: {value!r}')
    return (month, day)


def nc_files(directory: Path) -> list[str]:
    return sorted(glob.glob(str(directory / '*.nc')))


def to_daily_precip_mm(pr: xr.DataArray) -> xr.DataArray:
    """Convert daily precipitation to millimetres per day."""
    units = str(pr.attrs.get('units', '')).strip().lower().replace(' ', '')
    flux_units = {'kgm-2s-1', 'kgm**-2s**-1', 'kg/m2/s', 'mm/s', 'mms-1'}
    daily_mm_units = {'mm', 'mm/day', 'mmd-1', 'mmday-1', 'kgm-2', 'kg/m2'}
    if units in flux_units:
        result = pr * 86400.0
    elif units in daily_mm_units:
        result = pr
    else:
        raise ValueError(f"Unsupported precipitation units: {pr.attrs.get('units')!r}. Expected a daily total in mm or a flux in kg m-2 s-1.")
    result = result.rename('daily_precipitation')
    result.attrs.update(units='mm day-1', long_name='daily precipitation')
    return result


def degree_days_above_threshold(tmin_c: xr.DataArray, tmax_c: xr.DataArray, threshold_c: float) -> xr.DataArray:
    """Daily degree days above a threshold using sinusoidal interpolation."""
    mean_temp = 0.5 * (tmin_c + tmax_c)
    temp_range = tmax_c - tmin_c
    safe_range = xr.where(temp_range > 0.0, temp_range, np.nan)
    cosine_argument = ((2.0 * threshold_c - tmax_c - tmin_c) / safe_range).clip(min=-1.0, max=1.0)
    t_bar = np.arccos(cosine_argument)
    partial_exposure = t_bar / np.pi * (mean_temp - threshold_c) + temp_range / (2.0 * np.pi) * np.sin(t_bar)
    degree_days = xr.where(threshold_c <= tmin_c, mean_temp - threshold_c, xr.where(threshold_c >= tmax_c, 0.0, partial_exposure))
    degree_days = xr.where(temp_range == 0.0, xr.where(mean_temp > threshold_c, mean_temp - threshold_c, 0.0), degree_days)
    return degree_days.clip(min=0.0)


def select_growing_season(daily_data: xr.DataArray, season_start: str, season_end: str) -> xr.DataArray:
    """Keep inclusive MM-DD dates, including seasons that cross New Year."""
    start_month, start_day = validate_month_day(season_start, 'season_start')
    end_month, end_day = validate_month_day(season_end, 'season_end')
    month_day = daily_data.time.dt.month * 100 + daily_data.time.dt.day
    start_key = start_month * 100 + start_day
    end_key = end_month * 100 + end_day
    if start_key <= end_key:
        mask = (month_day >= start_key) & (month_day <= end_key)
    else:
        mask = (month_day >= start_key) | (month_day <= end_key)
    return daily_data.sel(time=mask)


def safe_token(text: str) -> str:
    """Convert text into a filename-safe token."""
    return re.sub('[^A-Za-z0-9_.-]+', '_', text)


def open_time_series(files: list[str], variable: str) -> xr.DataArray:
    """
    Open and combine NetCDF files for one requested variable.

    Directory-to-internal-variable mappings:
        hursmin -> rel_humid_min
        hursmax -> rel_humid_max
    """
    if not files:
        raise FileNotFoundError(f'No NetCDF files found for {variable}')
    variable_aliases = {'hursmin': 'rel_humid_min', 'hursmax': 'rel_humid_max'}
    datasets = [xr.open_dataset(file, chunks={'time': 366}) for file in files]
    try:
        dataset = xr.combine_by_coords(datasets, combine_attrs='override', join='outer')
        available_variables = list(dataset.data_vars)
        if variable in dataset.data_vars:
            internal_variable = variable
        elif variable in variable_aliases and variable_aliases[variable] in dataset.data_vars:
            internal_variable = variable_aliases[variable]
        else:
            raise KeyError(f'Requested variable {variable!r} was not found. Expected internal name {variable_aliases.get(variable, variable)!r}. Available variables: {available_variables}. First file: {files[0]}')
        source = dataset[internal_variable]
        data = source.rename(variable).sortby('time')
        _, unique_index = np.unique(data.time.values, return_index=True)
        data = data.isel(time=np.sort(unique_index))
        data.attrs = source.attrs.copy()
        data.attrs['source_variable_name'] = internal_variable
        return data
    except Exception:
        for opened_dataset in datasets:
            opened_dataset.close()
        raise


def historical_dir(root: Path, variable: str, model: str, member: str) -> Path:
    """Locate the historical run corresponding to an SSP run."""
    exact = root / variable / 'historical' / model / member
    if nc_files(exact):
        return exact
    model_directory = root / variable / 'historical' / model
    candidates = sorted(model_directory.glob('*/')) if model_directory.is_dir() else []
    candidates = [path for path in candidates if nc_files(path)]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f'Could not uniquely match historical {variable} for {model}/{member}. Expected {exact} or exactly one historical member under {model_directory}.')


def combine_historical_future(root: Path, variable: str, model: str, member: str, scenario: str) -> xr.DataArray:
    """Combine 1985–2014 historical and 2015–2094 SSP data."""
    historical = open_time_series(nc_files(historical_dir(root, variable, model, member)), variable)
    future = open_time_series(nc_files(root / variable / scenario / model / member), variable)
    historical = historical.sel(time=(historical.time.dt.year >= 1985) & (historical.time.dt.year <= 2014))
    future = future.sel(time=(future.time.dt.year >= 2015) & (future.time.dt.year <= 2094))
    combined = xr.concat([historical, future], dim='time').sortby('time')
    _, unique_index = np.unique(combined.time.values, return_index=True)
    return combined.isel(time=np.sort(unique_index))


def to_celsius(data: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to degrees Celsius."""
    units = str(data.attrs.get('units', '')).strip().lower()
    kelvin_units = {'k', 'kelvin', 'degrees_k', 'degree_k'}
    celsius_units = {'c', 'degc', 'degree_celsius', 'degrees_celsius', '°c'}
    if units in kelvin_units:
        converted = data - 273.15
    elif units in celsius_units:
        converted = data
    else:
        raise ValueError(f"Unsupported or missing temperature units: {data.attrs.get('units')!r}")
    converted.attrs = data.attrs.copy()
    converted.attrs['units'] = 'degrees_Celsius'
    return converted


def to_fahrenheit(data_c):
    """Convert degrees Celsius to degrees Fahrenheit."""
    result = data_c * 9.0 / 5.0 + 32.0
    result.attrs = data_c.attrs.copy()
    result.attrs['units'] = 'degree_Fahrenheit'
    return result


def to_daily_precip_inches(data):
    """Convert LOCA2 daily precipitation to inches per day."""
    result = to_daily_precip_mm(data) / 25.4
    result = result.rename('daily_precipitation_inches')
    result.attrs.update(units='inch day-1', long_name='daily precipitation')
    return result


def to_daily_precip_cm(data):
    """Convert LOCA2 daily precipitation to centimeters per day."""
    result = to_daily_precip_mm(data) / 10.0
    result = result.rename('daily_precipitation_cm')
    result.attrs.update(units='cm day-1', long_name='daily precipitation')
    return result


def daily_mean_temperature(tmin, tmax):
    """Calculate daily mean temperature."""
    tmin, tmax = xr.align(tmin, tmax, join='exact')
    return 0.5 * (tmin + tmax)


def annual_sum(data):
    """Sum daily values within each calendar year."""
    return data.groupby('time.year').sum('time', skipna=False)


def annual_day_count(condition):
    """Count days satisfying a condition in every year."""
    return condition.astype('int16').groupby('time.year').sum('time', skipna=False).astype('int16')


def monthly_day_count(condition):
    """
    Count days satisfying a condition for every calendar month
    within every year.
    """
    monthly_parts = []
    for month in range(1, 13):
        selected = condition.where(condition.time.dt.month == month, drop=True)
        count = selected.astype('int16').groupby('time.year').sum('time', skipna=False).astype('int16').expand_dims(month=[month])
        monthly_parts.append(count)
    return xr.concat(monthly_parts, dim='month', join='exact').transpose('year', 'month', ...)


def monthly_sum(data):
    """Calculate monthly totals for every year."""
    monthly_parts = []
    for month in range(1, 13):
        selected = data.where(data.time.dt.month == month, drop=True)
        total = selected.groupby('time.year').sum('time', skipna=False).expand_dims(month=[month])
        monthly_parts.append(total)
    return xr.concat(monthly_parts, dim='month', join='exact').transpose('year', 'month', ...)


def annual_temperature_bin_counts(temperature, edges, labels):
    """Create annual day counts for nonoverlapping temperature bins."""
    output = {}
    for index, label in enumerate(labels):
        lower = edges[index]
        upper = edges[index + 1]
        if np.isneginf(lower):
            condition = temperature < upper
        elif np.isposinf(upper):
            condition = temperature >= lower
        else:
            condition = (temperature >= lower) & (temperature < upper)
        output[label] = annual_day_count(condition)
    return output


def monthly_temperature_bin_counts(temperature, edges, labels):
    """Create monthly day counts for nonoverlapping temperature bins."""
    output = {}
    for index, label in enumerate(labels):
        lower = edges[index]
        upper = edges[index + 1]
        if np.isneginf(lower):
            condition = temperature < upper
        elif np.isposinf(upper):
            condition = temperature >= lower
        else:
            condition = (temperature >= lower) & (temperature < upper)
        output[label] = monthly_day_count(condition)
    return output


def bounded_degree_days(tmin_c, tmax_c, lower_c, upper_c):
    """
    Sinusoidal degree days between lower and upper thresholds.
    """
    above_lower = degree_days_above_threshold(tmin_c, tmax_c, lower_c)
    above_upper = degree_days_above_threshold(tmin_c, tmax_c, upper_c)
    return (above_lower - above_upper).clip(min=0.0)


def pack_into_periods(dataset):
    """
    Convert an annual dataset from a calendar-year dimension to
    period and year_index dimensions used by the existing notebook.
    """
    if 'year' not in dataset.dims:
        raise ValueError('The paper dataset must contain a year dimension.')
    available_years = {int(year) for year in dataset.year.values}
    period_parts = []
    calendar_year_rows = []
    for label, (start_year, end_year) in PERIODS.items():
        expected_years = list(range(start_year, end_year + 1))
        missing_years = sorted(set(expected_years) - available_years)
        if missing_years:
            raise ValueError(f'{label} is missing years: {missing_years}')
        year_index = np.arange(1, len(expected_years) + 1)
        part = dataset.sel(year=expected_years).rename(year='year_index').assign_coords(year_index=year_index).expand_dims(period=[label])
        period_parts.append(part)
        calendar_year_rows.append(expected_years)
    result = xr.concat(period_parts, dim='period', join='exact', combine_attrs='override')
    result['calendar_year'] = xr.DataArray(np.asarray(calendar_year_rows, dtype='int16'), dims=('period', 'year_index'), coords={'period': list(PERIODS), 'year_index': np.arange(1, 31)}, attrs={'long_name': 'calendar year corresponding to each annual value', 'units': 'year'})
    return result


def discover_paper_runs(required_variables, scenarios=PAPER_SCENARIOS):
    """Find model/member/scenario runs with all required variables."""
    jobs = []
    for scenario in scenarios:
        reference_root = PAPER_INPUT_ROOT / required_variables[0] / scenario
        if not reference_root.is_dir():
            continue
        for model_dir in sorted((path for path in reference_root.iterdir() if path.is_dir())):
            for member_dir in sorted((path for path in model_dir.iterdir() if path.is_dir())):
                model = model_dir.name
                member = member_dir.name
                available = all((nc_files(PAPER_INPUT_ROOT / variable / scenario / model / member) for variable in required_variables))
                if available:
                    jobs.append((model, member, scenario))
    return jobs


def load_paper_inputs(model, member, scenario, required_variables):
    """Load the requested historical-plus-future LOCA2 variables."""
    loaded = {}
    for variable in required_variables:
        data = combine_historical_future(PAPER_INPUT_ROOT, variable, model, member, scenario)
        if variable in {'tasmin', 'tasmax'}:
            data = to_celsius(data)
        elif variable == 'pr':
            data = to_daily_precip_inches(data)
        loaded[variable] = data
    return loaded


def paper_directory_name(paper_name):
    """
    Keep the paper name readable while removing path separators.
    """
    return re.sub('[\\\\/]+', '_', paper_name).strip()


def run_paper(paper_name, required_variables, calculation_function, overwrite=False):
    """
    Write one NetCDF for every model/member/scenario.
    """
    output_directory = PAPER_OUTPUT_ROOT / paper_directory_name(paper_name)
    output_directory.mkdir(parents=True, exist_ok=True)
    jobs = discover_paper_runs(required_variables=required_variables)
    completed = 0
    skipped = 0
    failed = 0
    progress = tqdm(jobs, desc=paper_name, unit='run', dynamic_ncols=False)
    for model, member, scenario in progress:
        progress.set_postfix_str(f'{model} | {member} | {scenario}')
        output_file = output_directory / f'{safe_token(paper_name)}_{safe_token(model)}_{safe_token(member)}_{scenario}.nc'
        if output_file.exists() and (not overwrite):
            skipped += 1
            tqdm.write(f'SKIPPED existing: {output_file.name}')
            continue
        inputs = {}
        result = None
        try:
            inputs = load_paper_inputs(model=model, member=member, scenario=scenario, required_variables=required_variables)
            annual_dataset = calculation_function(inputs)
            result = pack_into_periods(annual_dataset)
            result.attrs.update(paper=paper_name, model=model, member=member, scenario=scenario, spatial_resolution='Illinois county', spatial_weighting='cosine-latitude grid-cell area correction; no LandScan, population, or crop weighting', periods='1985-2014, 2015-2044, 2035-2064, 2065-2094', file_structure='one historical-plus-future file per model, member, and scenario')
            encoding = {variable: {'zlib': True, 'complevel': 4, 'shuffle': True} for variable in result.data_vars}
            result.to_netcdf(output_file, encoding=encoding)
            completed += 1
            tqdm.write(f'WROTE: {output_file}')
        except Exception as error:
            failed += 1
            tqdm.write(f'FAILED: {model}/{member}/{scenario}: {type(error).__name__}: {error}')
        finally:
            if result is not None:
                result.close()
            for data in inputs.values():
                try:
                    data.close()
                except Exception:
                    pass
            del inputs
            result = None
            gc.collect()
    print(f'{paper_name}: completed={completed}, skipped={skipped}, failed={failed}')


def monthly_mean(data):
    """
    Calculate the mean for each calendar month and year.
    """
    monthly_parts = []
    for month in range(1, 13):
        selected = data.where(data.time.dt.month == month, drop=True)
        mean_value = selected.groupby('time.year').mean('time', skipna=False).expand_dims(month=[month])
        monthly_parts.append(mean_value)
    return xr.concat(monthly_parts, dim='month', join='exact').transpose('year', 'month', ...)


def aggregate_to_illinois_counties(dataset, counties=None):
    """
    Aggregate gridded LOCA2 indicators to Illinois counties.

    Weighting:
        - No LandScan or population weighting
        - No crop weighting
        - Cosine-latitude weighting is used to approximate the
          relative areas of regular latitude-longitude grid cells

    County assignment:
        A LOCA2 grid cell is assigned to the county containing its
        grid-cell center.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing lat and lon dimensions.

    counties : geopandas.GeoDataFrame
        Illinois county boundaries and county metadata.

    Returns
    -------
    xarray.Dataset
        County-level indicators with the lat and lon dimensions
        replaced by a county dimension.
    """
    import regionmask
    if counties is None:
        counties = load_counties()
    regions = regionmask.Regions(outlines=list(counties.geometry), numbers=list(range(len(counties))), names=counties['county_name'].tolist(), abbrevs=counties['county_id'].tolist(), name='Illinois counties')
    if not isinstance(dataset, xr.Dataset):
        raise TypeError('dataset must be an xarray.Dataset.')
    if not {'lat', 'lon'}.issubset(dataset.dims):
        raise ValueError(f'dataset must contain lat and lon dimensions. Available dimensions: {tuple(dataset.dims)}')
    if dataset['lat'].ndim != 1 or dataset['lon'].ndim != 1:
        raise ValueError('This function requires one-dimensional lat and lon coordinates.')
    dataset = dataset.sortby('lat').sortby('lon')
    county_mask = regions.mask_3D(dataset['lon'], dataset['lat'], drop=False, wrap_lon=180)
    county_mask = county_mask.rename(region='county').assign_coords(county=np.arange(len(counties), dtype=np.int32), county_id=('county', counties['county_id'].to_numpy()), county_name=('county', counties['county_name'].to_numpy()))
    latitude_weights = xr.DataArray(np.cos(np.deg2rad(dataset['lat'].values)), dims=('lat',), coords={'lat': dataset['lat']}, name='latitude_area_weight')
    spatial_weights = county_mask.astype('float64') * latitude_weights
    cells_per_county = county_mask.sum(dim=('lat', 'lon'))
    empty_county_indices = np.flatnonzero(cells_per_county.values == 0)
    if empty_county_indices.size:
        empty_county_names = counties.iloc[empty_county_indices]['county_name'].tolist()
        raise ValueError(f'No LOCA2 grid-cell centers were assigned to: {empty_county_names}')
    county_variables = {}
    for variable_name, data in dataset.data_vars.items():
        if {'lat', 'lon'}.issubset(data.dims):
            county_data = data.weighted(spatial_weights).mean(dim=('lat', 'lon'), skipna=True)
            preferred_order = [dimension for dimension in ('year', 'month', 'county') if dimension in county_data.dims]
            remaining_dimensions = [dimension for dimension in county_data.dims if dimension not in preferred_order]
            county_data = county_data.transpose(*preferred_order, *remaining_dimensions)
            county_data.attrs = data.attrs.copy()
            county_data.attrs.update(spatial_aggregation='county-level area-weighted mean', spatial_weighting='cosine of LOCA2 grid-cell-center latitude; no LandScan, population, or crop weighting', county_assignment='LOCA2 grid-cell center located within the county boundary')
            county_variables[variable_name] = county_data
        else:
            county_variables[variable_name] = data
    county_dataset = xr.Dataset(county_variables)
    county_dataset = county_dataset.assign_coords(county_id=('county', counties['county_id'].to_numpy()), county_name=('county', counties['county_name'].to_numpy()))
    county_dataset['county_id'].attrs.update(long_name='five-digit county FIPS code')
    county_dataset['county_name'].attrs.update(long_name='Illinois county name')
    county_dataset.attrs = dataset.attrs.copy()
    county_dataset.attrs.update(spatial_resolution='Illinois county', spatial_aggregation='area-weighted mean of LOCA2 grid cells assigned to each Illinois county', spatial_weighting='cosine-latitude grid-cell area correction; no LandScan, population, or crop weighting', county_assignment_method='grid-cell-center point-in-polygon', number_of_counties=len(counties))
    return county_dataset
