# LOCA2 Illinois Climate Indicators

Python scripts and Jupyter notebooks for downloading LOCA2 climate
projections and calculating weather indicators for climate-impact
applications in Illinois.

The workflow includes paper-specific indicators, aggregation to Illinois
counties, and a separate CONUS temperature workflow for the Wobus et al.
(2021) application.

These notebooks implement the configurations documented below. They
include project-specific adaptations and should not be interpreted as
exact reproductions of every original paper.

## Data source

LOCA2 statistically downscaled CMIP6 projections:

https://loca.ucsd.edu/loca-version-2-for-north-america-ca-jan-2023/

The Illinois indicator workflows use daily minimum temperature,
maximum temperature, and precipitation on the native LOCA2 grid,
approximately 1/16°.

The project uses historical simulations and future scenarios:

- SSP2-4.5 (`ssp245`)
- SSP3-7.0 (`ssp370`)

Model and ensemble-member availability varies by scenario. Each
model-member-scenario combination is processed separately.

## Repository components

- `download_loca2_il.py`: Illinois data download and processing.
- `download_loca2_il.slurm`: Slurm submission script for the downloader.
- `Processed File Exploration and Verification.ipynb`: inspection of
  downloaded and processed data.
- `loca2_common.py`: shared functions used by the paper notebooks.
- Paper-specific notebooks: indicator calculation and output generation.
- CONUS downloader and Wobus notebook: monthly temperature preparation
  and annual CONUS temperature calculation.

Keep `loca2_common.py` accessible to the notebooks, for example in the
same directory.

## Paper workflows

| Paper | Main indicators | Temporal aggregation |
|---|---|---|
| Deschênes & Greenstone (2011) | Daily mean temperature-bin counts; precipitation totals and categories | Annual |
| Burke & Emerick (2016) | Degree days using 29°C and alternative 28°C thresholds; precipitation terms using 42 cm and 50 cm thresholds | March–August, summarized annually |
| Park (2016) | Daily maximum temperature counts for 80–<90°F and ≥90°F | Annual |
| Behrer et al. (2021) | Daily maximum temperature counts for 80–<85°F, 85–<90°F, and ≥90°F | Annual |
| Carleton et al. (2022) | Annual sums of daily T, T², T³, T⁴, P, and P² | Annual |
| Hultgren et al. (2025) | GDD between 8°C and 31°C; KDD above 31°C; crop-phase precipitation and squared precipitation | March–August, summarized annually |
| Manderson & Considine (2025) | Daily mean temperature-bin counts and precipitation | Monthly |
| Wobus et al. (2021) application | Absolute annual CONUS-average temperature and baseline-relative warming | Annual |

Relative humidity is not a required component of the current workflow.
Some earlier Manderson outputs include a relative-humidity variable.

## Units and calculations

Input units must be checked before calculation.

- Temperature thresholds for degree days use degrees Celsius.
- Temperature-bin indicators and Carleton temperature powers use
  degrees Fahrenheit.
- Most precipitation indicators use inches.
- Burke & Emerick precipitation indicators use centimeters.
- Degree-day indicators have units of °C-days.
- Temperature-bin counts have units of days.

Daily mean temperature is estimated as:

    Tmean = (Tmin + Tmax) / 2

Degree days use a sinusoidal approximation between daily minimum
and maximum temperature. Exposure above a threshold integrates
temperature excess over the portion of the day above that threshold.

The agricultural notebooks use March 1 through August 31 as a
project-specific growing season. Crop precipitation phases are defined
explicitly in each notebook.

## County aggregation

The county workflows produce indicators for Illinois counties using
grid-area correction weights.

No population, LandScan, or crop-area weights are applied.

Cosine-latitude weighting accounts for the variation in grid-cell area
on a regular latitude–longitude grid. County values represent spatial
averages, not sums across grid cells.

Consequently, county-average temperature-bin day counts may be fractional.

The shared aggregation function defines county membership and
missing-value handling. Review that function when interpreting cells
near county boundaries or counties with missing data.

### Order of operations

Nonlinear transformations depend on when county aggregation occurs:

- Degree days and temperature-bin counts are calculated at grid cells
  before county averaging.
- Carleton daily powers are calculated before annual summation and
  county averaging.
- Hultgren precipitation-phase totals are averaged to counties before
  being squared.
- Burke & Emerick precipitation thresholds are applied to
  county-average growing-season precipitation.

For example, Carleton `sum(T²)` is the annual sum of squared daily
temperature, not the square of the annual temperature sum.

## Time periods and output structure

The Illinois workflow uses the following analysis windows:

| Period | Years |
|---|---|
| Historical | 1985–2014 |
| Near-term | 2015–2044 |
| Mid-century | 2035–2064 |
| Late-century | 2065–2094 |

The near-term and mid-century windows overlap intentionally.

Outputs are NetCDF files organized by paper, with separate files for
each processed model-member-scenario combination.

Depending on the processing stage, dimensions can include:

- `year`, `county`
- `period`, `year_index`, `county`
- `period`, `year_index`, `month`, `county`

Use `calendar_year`, when present, to map `year_index` to actual years.

Existing outputs are retained when `overwrite=False`.

## Wobus CONUS workflow

This workflow uses the contiguous United States rather than Illinois.

Monthly gridded minimum and maximum temperature are downloaded from
the LOCA2 North American archive. Source variables such as
`tasmin_tavg` and `tasmax_tavg` are standardized to `tasmin` and `tasmax`.

The downloader retains grid cells whose centers fall inside the
48 contiguous states and Washington, DC. Cells outside the mask are
set to missing. No regridding is performed.

Annual temperature is derived from monthly mean Tmin and Tmax using:

1. Mean temperature estimated as `(Tmin + Tmax) / 2`.
2. CONUS grid-area weighting.
3. Month-length weighting using the dataset calendar.

The requested annual series covers 1985–2094, with a 2001–2020
reference period for warming calculations. Historical and scenario
data are needed to construct that reference period.

This temperature estimate is derived from Tmin and Tmax; it is not
an independently downscaled `tas` variable.

## Setup and execution

The scripts were developed for the University of Illinois
Urbana-Champaign Keeling HPC system:

https://atmsillinois.github.io/

Before running:

1. Install the Python dependencies required by the selected workflow.
2. Configure input, output, and boundary-file paths.
3. Make the shared Python module accessible to each notebook.
4. Download the required climate data.
5. Run notebook cells in order, starting with imports and settings.
6. Inspect the output and failure logs.

Common dependencies include:

- NumPy
- pandas
- xarray
- Dask
- netCDF4
- cftime
- GeoPandas
- Shapely
- regionmask, where used by the aggregation implementation
- requests
- Beautiful Soup
- tqdm
- Matplotlib

Slurm scripts contain environment paths, resource requests, and log
locations specific to Keeling. Adapt these settings for another system.
Create log directories before submitting a job.

## Validation and interpretation

Before using outputs:

- Confirm input units and calendars.
- Check time coverage and missing dates.
- Confirm county identifiers and spatial coverage.
- Review failed or skipped model-member-scenario combinations.
- Verify the order of nonlinear transformations and aggregation.
- Record the processing configuration and code version.

An inner time alignment retains shared timestamps but can remove dates.
Annual sums over the remaining dates are not necessarily complete-year
totals. Missing-date checks are needed even when summation uses
`skipna=False`.

Some NetCDF readers interpret variables with `units="days"` as
timedeltas. If numeric day counts are intended, inspect the decoded
dtype and consider opening the file with `decode_timedelta=False`.

Storage requirements depend on the selected models, members, scenarios,
variables, temporal resolution, and retained intermediate files.

## References and attribution

Use the LOCA2 documentation for dataset references and version information:

https://loca.ucsd.edu/loca-version-2-for-north-america-ca-jan-2023/

For scientific applications, cite the original paper associated with
each indicator and document any changes to its season, thresholds,
units, weighting, or aggregation procedure.

Maintainer: Tahsina Alam
