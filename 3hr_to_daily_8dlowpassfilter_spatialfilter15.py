import xarray as xr
import numpy as np
import glob
import os
import re

from concurrent.futures import ProcessPoolExecutor


# ============================================================
# CONSTANTS
# ============================================================

a = 6.371e6
Omega = 7.292e-5
g = 9.80665
DEG2RAD = np.pi / 180.0


# ============================================================
# SETTINGS
# ============================================================

out_dir = os.getcwd()

files = sorted(
    glob.glob(
        "/pscratch/sd/x/xie7/ls4p/zppy/output/daily/*/*nc"
    )
)

NPROC = min(6, os.cpu_count())

FILES_PER_GROUP = 5

# Latitude decomposition
N_LAT_CHUNKS = 10
LAT_HALO = 3

# Lanczos low-pass
LONG_CUTOFF_DAYS = 16.0  #8.0
NWT = 61

# Zonal wavenumber filter
MAX_ZONAL_WAVENUMBER = 15


# ============================================================
# CHECK INPUT
# ============================================================

if len(files) == 0:
    raise RuntimeError("No input files found.")

print("Number of input files:", len(files))
print("NPROC:", NPROC)
print("FILES_PER_GROUP:", FILES_PER_GROUP)
print("N_LAT_CHUNKS:", N_LAT_CHUNKS)
print("LAT_HALO:", LAT_HALO)
print("Lanczos cutoff:", LONG_CUTOFF_DAYS, "days")
print("Lanczos NWT:", NWT)
print(
    "Maximum zonal wavenumber:",
    MAX_ZONAL_WAVENUMBER
)


# ============================================================
# LANCZOS WEIGHTS
# ============================================================

def lanczos_weights(
    cutoff_days,
    dt_days,
    nwt
):

    if nwt % 2 == 0:
        raise ValueError(
            "NWT must be odd."
        )

    m = (nwt - 1) // 2

    # Cutoff frequency [cycles/day]
    fc = 1.0 / cutoff_days

    # Normalized cutoff frequency
    fc_norm = fc * dt_days

    k = np.arange(
        -m,
        m + 1,
        dtype=float
    )

    h = np.zeros_like(k)

    # Ideal low-pass filter
    h[k == 0] = 2.0 * fc_norm

    nonzero = k != 0

    h[nonzero] = (
        np.sin(
            2.0
            * np.pi
            * fc_norm
            * k[nonzero]
        )
        /
        (
            np.pi
            * k[nonzero]
        )
    )

    # Lanczos sigma window
    sigma = np.ones_like(k)

    sigma[nonzero] = (
        np.sin(
            np.pi
            * k[nonzero]
            / (m + 1)
        )
        /
        (
            np.pi
            * k[nonzero]
            / (m + 1)
        )
    )

    weights = h * sigma

    # Normalize
    weights /= weights.sum()

    return weights


# ============================================================
# REFLECTIVE TEMPORAL FILTER
# ============================================================

def apply_filter(
    da,
    dim,
    weights
):

    n = len(weights)

    if n % 2 == 0:
        raise ValueError(
            "Number of weights must be odd."
        )

    m = (n - 1) // 2

    ndata = da.sizes[dim]

    if ndata <= m:
        raise ValueError(
            f"Not enough data points for "
            f"{n}-point filter."
        )

    # --------------------------------------------------------
    # Reflect left boundary
    # --------------------------------------------------------

    left = da.isel(
        {
            dim: slice(
                1,
                m + 1
            )
        }
    )

    left = left.isel(
        {
            dim: slice(
                None,
                None,
                -1
            )
        }
    )

    # --------------------------------------------------------
    # Reflect right boundary
    # --------------------------------------------------------

    right = da.isel(
        {
            dim: slice(
                -m - 1,
                -1
            )
        }
    )

    right = right.isel(
        {
            dim: slice(
                None,
                None,
                -1
            )
        }
    )

    # --------------------------------------------------------
    # Extended array
    # --------------------------------------------------------

    da_extended = xr.concat(
        [
            left,
            da,
            right
        ],
        dim=dim
    )

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    weights_da = xr.DataArray(
        weights,
        dims=["window"]
    )

    filtered_extended = (
        da_extended
        .rolling(
            {
                dim: n
            },
            center=True
        )
        .construct("window")
        .dot(weights_da)
    )

    # --------------------------------------------------------
    # Remove reflected portions
    # --------------------------------------------------------

    filtered = filtered_extended.isel(
        {
            dim: slice(
                m,
                m + ndata
            )
        }
    )

    # Restore original coordinates
    filtered = filtered.assign_coords(
        {
            dim: da[dim]
        }
    )

    return filtered


# ============================================================
# ZONAL WAVENUMBER FILTER
# ============================================================

def zonal_wavenumber_filter(
    da,
    max_wavenumber
):

    if "lon" not in da.dims:
        raise ValueError(
            "Input must contain lon dimension."
        )

    nlon = da.sizes["lon"]

    lon = da["lon"].values

    # Check uniform longitude spacing
    dlon = np.diff(lon)

    if not np.allclose(
        dlon,
        dlon[0],
        rtol=1e-5,
        atol=1e-8
    ):
        raise ValueError(
            "Longitude grid is not uniformly spaced."
        )

    # Longitude axis
    lon_axis = da.get_axis_num("lon")

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    data_fft = np.fft.fft(
        da.values,
        axis=lon_axis
    )

    # --------------------------------------------------------
    # Zonal wavenumber
    # --------------------------------------------------------

    wn = (
        np.fft.fftfreq(nlon)
        * nlon
    )

    keep = (
        np.abs(wn)
        <= max_wavenumber
    )

    # Reshape mask for broadcasting
    shape = [
        1
        for _ in range(
            data_fft.ndim
        )
    ]

    shape[lon_axis] = nlon

    keep_shape = np.reshape(
        keep,
        shape
    )

    # --------------------------------------------------------
    # Apply filter
    # --------------------------------------------------------

    data_fft *= keep_shape

    # --------------------------------------------------------
    # Inverse FFT
    # --------------------------------------------------------

    filtered_values = np.fft.ifft(
        data_fft,
        axis=lon_axis
    ).real

    filtered = xr.DataArray(
        filtered_values,
        coords=da.coords,
        dims=da.dims,
        attrs=da.attrs,
        name=da.name
    )

    return filtered


# ============================================================
# READ ONE FILE GROUP
# ============================================================

def read_file_group(
    file_group
):

    U_list = []
    V_list = []
    Z_list = []

    for f in file_group:

        print(
            "Reading:",
            os.path.basename(f)
        )

        ds = xr.open_dataset(
            f
        )

        # ----------------------------------------------------
        # Select 200 hPa
        # ----------------------------------------------------

        U = (
            ds["U"]
            .sel(
                lev=200,
                method="nearest"
            )
            .transpose(
                "time",
                "lat",
                "lon"
            )
            .load()
        )

        V = (
            ds["V"]
            .sel(
                lev=200,
                method="nearest"
            )
            .transpose(
                "time",
                "lat",
                "lon"
            )
            .load()
        )

        Z = (
            ds["Z3"]
            .sel(
                lev=200,
                method="nearest"
            )
            .transpose(
                "time",
                "lat",
                "lon"
            )
            .load()
        )

        U_list.append(U)
        V_list.append(V)
        Z_list.append(Z)

        ds.close()

    # --------------------------------------------------------
    # Concatenate files along time
    # --------------------------------------------------------

    U = xr.concat(
        U_list,
        dim="time"
    )

    V = xr.concat(
        V_list,
        dim="time"
    )

    Z = xr.concat(
        Z_list,
        dim="time"
    )

    # Explicitly guarantee dimension order
    U = U.transpose(
        "time",
        "lat",
        "lon"
    )

    V = V.transpose(
        "time",
        "lat",
        "lon"
    )

    Z = Z.transpose(
        "time",
        "lat",
        "lon"
    )

    return U, V, Z


# ============================================================
# DAILY MEAN
# ============================================================

def make_daily(
    da
):

    result = (
        da
        .resample(
            time="1D"
        )
        .mean(
            "time"
        )
    )

    # Explicit dimension order
    result = result.transpose(
        "time",
        "lat",
        "lon"
    )

    return result


# ============================================================
# PROCESS ONE LATITUDE CHUNK
# ============================================================

def process_lat_chunk(
    U,
    V,
    Z,
    U_bg,
    V_bg,
    Z_zm,
    lat_start,
    lat_end,
    chunk_id,
    group_id
):

    nlat = U.sizes["lat"]

    # --------------------------------------------------------
    # Add latitude halo
    # --------------------------------------------------------

    halo_start = max(
        0,
        lat_start - LAT_HALO
    )

    halo_end = min(
        nlat,
        lat_end + LAT_HALO
    )

    print(
        f"Group {group_id}: "
        f"latitude chunk {chunk_id}: "
        f"{lat_start}:{lat_end}, "
        f"halo {halo_start}:{halo_end}"
    )

    U_chunk = U.isel(
        lat=slice(
            halo_start,
            halo_end
        )
    )

    V_chunk = V.isel(
        lat=slice(
            halo_start,
            halo_end
        )
    )

    Z_chunk = Z.isel(
        lat=slice(
            halo_start,
            halo_end
        )
    )

    U_bg_chunk = U_bg.isel(
        lat=slice(
            halo_start,
            halo_end
        )
    )

    V_bg_chunk = V_bg.isel(
        lat=slice(
            halo_start,
            halo_end
        )
    )

    Z_zm_chunk = Z_zm.isel(
        lat=slice(
            halo_start,
            halo_end
        )
    )

    # --------------------------------------------------------
    # Latitude coordinates
    # --------------------------------------------------------

    lat_deg = (
        U_chunk["lat"].values
    )

    lat_rad = np.deg2rad(
        lat_deg
    )

    cos_lat = xr.DataArray(
        np.cos(lat_rad),
        coords={
            "lat": lat_deg
        },
        dims=["lat"]
    )

    # --------------------------------------------------------
    # Coriolis parameter
    # --------------------------------------------------------

    f = (
        2.0
        * Omega
        * np.sin(lat_rad)
    )

    f_da = xr.DataArray(
        f,
        coords={
            "lat": lat_deg
        },
        dims=["lat"]
    )

    # --------------------------------------------------------
    # Avoid singularity close to equator
    # --------------------------------------------------------

    f_da = xr.where(
        np.abs(f_da) < 1e-5,
        np.nan,
        f_da
    )

    # --------------------------------------------------------
    # Perturbation geopotential height
    # --------------------------------------------------------

    z_p = (
        Z_chunk
        - Z_zm_chunk
    )

    # --------------------------------------------------------
    # Geostrophic streamfunction
    # --------------------------------------------------------

    psi = (
        g
        * z_p
        / f_da
    )

    # --------------------------------------------------------
    # Derivatives
    # --------------------------------------------------------

    psi_lon = psi.differentiate(
        "lon"
    )

    psi_lat = psi.differentiate(
        "lat"
    )

    psi_x = (
        psi_lon
        /
        (
            DEG2RAD
            * a
            * cos_lat
        )
    )

    psi_y = (
        psi_lat
        /
        (
            DEG2RAD
            * a
        )
    )

    psi_xx = (
        psi_x.differentiate(
            "lon"
        )
        /
        (
            DEG2RAD
            * a
            * cos_lat
        )
    )

    psi_xy = (
        psi_x.differentiate(
            "lat"
        )
        /
        (
            DEG2RAD
            * a
        )
    )

    psi_yy = (
        psi_y.differentiate(
            "lat"
        )
        /
        (
            DEG2RAD
            * a
        )
    )

    # --------------------------------------------------------
    # TN wave-activity terms
    # --------------------------------------------------------

    term_xx = (
        psi_x**2
        -
        psi
        * psi_xx
    )

    term_xy = (
        psi_x
        * psi_y
        -
        psi
        * psi_xy
    )

    term_yy = (
        psi_y**2
        -
        psi
        * psi_yy
    )

    # --------------------------------------------------------
    # Background wind magnitude
    # --------------------------------------------------------

    U_mag = np.sqrt(
        U_bg_chunk**2
        +
        V_bg_chunk**2
    )

    # ========================================================
    # WEAK-WIND TREATMENT
    #
    # If |U| < 1 m/s, use 1 m/s.
    # ========================================================

    #U_mag = xr.where(
    #    U_mag < 1.0,
    #    1.0,
    #    U_mag
    #)

    # --------------------------------------------------------
    # WAF
    # --------------------------------------------------------

    WAF_x = (
        (
            U_bg_chunk
            * term_xx
            +
            V_bg_chunk
            * term_xy
        )
        /
        (
            2.0
            * U_mag
        )
    )

    WAF_y = (
        (
            U_bg_chunk
            * term_xy
            +
            V_bg_chunk
            * term_yy
        )
        /
        (
            2.0
            * U_mag
        )
    )

    # --------------------------------------------------------
    # Remove latitude halo
    # --------------------------------------------------------

    inner_start = (
        lat_start
        - halo_start
    )

    inner_end = (
        lat_end
        - halo_start
    )

    WAF_x = WAF_x.isel(
        lat=slice(
            inner_start,
            inner_end
        )
    )

    WAF_y = WAF_y.isel(
        lat=slice(
            inner_start,
            inner_end
        )
    )

    # --------------------------------------------------------
    # Explicit dimension order
    # --------------------------------------------------------

    WAF_x = WAF_x.transpose(
        "time",
        "lat",
        "lon"
    )

    WAF_y = WAF_y.transpose(
        "time",
        "lat",
        "lon"
    )

    return (
        WAF_x,
        WAF_y
    )


# ============================================================
# PROCESS ONE FILE GROUP
# ============================================================

def process_group(
    group_id,
    file_group
):

    print(
        "\n========================================"
    )

    print(
        f"PROCESSING GROUP {group_id}"
    )

    print(
        "========================================"
    )

    # ========================================================
    # OUTPUT NAME BASED ON FIRST FILE
    # ========================================================

    first_file = os.path.basename(
        file_group[0]
    )

    first_name = os.path.splitext(
        first_file
    )[0]

    filtered_file = os.path.join(
        out_dir,
        f"{first_name}_filtered_daily.nc"
    )

    output_file = os.path.join(
        out_dir,
        f"{first_name}_WAF.nc"
    )

    print(
        "First file of group:"
    )

    print(
        first_file
    )

    print(
        "Filtered output:"
    )

    print(
        filtered_file
    )

    print(
        "WAF output:"
    )

    print(
        output_file
    )

    # ========================================================
    # READ 3-HOURLY DATA
    # ========================================================

    U_3h, V_3h, Z_3h = (
        read_file_group(
            file_group
        )
    )

    print(
        "3-hourly dimensions:",
        U_3h.sizes
    )

    # ========================================================
    # 3-HOURLY -> DAILY
    # ========================================================

    print(
        "Converting 3-hourly to daily..."
    )

    U_daily = make_daily(
        U_3h
    )

    V_daily = make_daily(
        V_3h
    )

    Z_daily = make_daily(
        Z_3h
    )

    print(
        "Daily dimensions:",
        U_daily.sizes
    )

    # Free original 3-hourly arrays
    del U_3h
    del V_3h
    del Z_3h

    # ========================================================
    # LANCZOS WEIGHTS
    # ========================================================

    dt_days = 1.0

    weights = lanczos_weights(
        LONG_CUTOFF_DAYS,
        dt_days,
        NWT
    )

    print(
        "Lanczos weights sum:",
        weights.sum()
    )

    # ========================================================
    # 8-DAY LOW-PASS
    # ========================================================

    print(
        "Applying 8-day Lanczos low-pass..."
    )

    U_lp = apply_filter(
        U_daily,
        "time",
        weights
    )

    V_lp = apply_filter(
        V_daily,
        "time",
        weights
    )

    Z_lp = apply_filter(
        Z_daily,
        "time",
        weights
    )

    # Explicit dimension order
    U_lp = U_lp.transpose(
        "time",
        "lat",
        "lon"
    )

    V_lp = V_lp.transpose(
        "time",
        "lat",
        "lon"
    )

    Z_lp = Z_lp.transpose(
        "time",
        "lat",
        "lon"
    )

    # ========================================================
    # DIAGNOSTICS AFTER TEMPORAL FILTER
    # ========================================================

    print(
        "NaNs after Lanczos:"
    )

    print(
        "U:",
        int(
            np.isnan(
                U_lp.values
            ).sum()
        )
    )

    print(
        "V:",
        int(
            np.isnan(
                V_lp.values
            ).sum()
        )
    )

    print(
        "Z:",
        int(
            np.isnan(
                Z_lp.values
            ).sum()
        )
    )

    # ========================================================
    # GLOBAL ZONAL WAVENUMBER <= 15
    # ========================================================

    print(
        "Applying global zonal "
        "wavenumber <= 15 filter..."
    )

    U_filt = zonal_wavenumber_filter(
        U_lp,
        MAX_ZONAL_WAVENUMBER
    )

    V_filt = zonal_wavenumber_filter(
        V_lp,
        MAX_ZONAL_WAVENUMBER
    )

    Z_filt = zonal_wavenumber_filter(
        Z_lp,
        MAX_ZONAL_WAVENUMBER
    )

    # Explicit dimension order
    U_filt = U_filt.transpose(
        "time",
        "lat",
        "lon"
    )

    V_filt = V_filt.transpose(
        "time",
        "lat",
        "lon"
    )

    Z_filt = Z_filt.transpose(
        "time",
        "lat",
        "lon"
    )

    # ========================================================
    # DIAGNOSTICS AFTER SPATIAL FILTER
    # ========================================================

    print(
        "NaNs after zonal wavenumber filter:"
    )

    print(
        "U:",
        int(
            np.isnan(
                U_filt.values
            ).sum()
        )
    )

    print(
        "V:",
        int(
            np.isnan(
                V_filt.values
            ).sum()
        )
    )

    print(
        "Z:",
        int(
            np.isnan(
                Z_filt.values
            ).sum()
        )
    )

    # ========================================================
    # OUTPUT FILTERED DAILY U/V/Z
    # ========================================================

    print(
        "Writing filtered daily fields..."
    )

    ds_filtered = xr.Dataset(
        {
            "U200_filtered": U_filt,
            "V200_filtered": V_filt,
            "Z3_200_filtered": Z_filt
        }
    )

    # Explicitly guarantee ALL output variables
    # are (time, lat, lon)
    for var in ds_filtered.data_vars:
        ds_filtered[var] = (
            ds_filtered[var]
            .transpose(
                "time",
                "lat",
                "lon"
            )
        )

    ds_filtered[
        "U200_filtered"
    ].attrs[
        "description"
    ] = (
        "Daily 200-hPa U after "
        "8-day Lanczos low-pass and "
        "zonal wavenumber <= 15 filter"
    )

    ds_filtered[
        "V200_filtered"
    ].attrs[
        "description"
    ] = (
        "Daily 200-hPa V after "
        "8-day Lanczos low-pass and "
        "zonal wavenumber <= 15 filter"
    )

    ds_filtered[
        "Z3_200_filtered"
    ].attrs[
        "description"
    ] = (
        "Daily 200-hPa Z3 after "
        "8-day Lanczos low-pass and "
        "zonal wavenumber <= 15 filter"
    )

    ds_filtered.attrs[
        "temporal_filter"
    ] = (
        "8-day Lanczos low-pass"
    )

    ds_filtered.attrs[
        "temporal_boundary"
    ] = (
        "symmetric reflective"
    )

    ds_filtered.attrs[
        "spatial_filter"
    ] = (
        "Global zonal wavenumber <= 15"
    )

    ds_filtered.to_netcdf(
        filtered_file
    )

    ds_filtered.close()

    print(
        "Filtered daily fields written:"
    )

    print(
        filtered_file
    )

    # ========================================================
    # BACKGROUND ZONAL MEAN
    # ========================================================

    # Keep the existing structure:
    # background U/V are based on DAILY fields.
    #
    # Wave field is the filtered U/V/Z.
    # ========================================================

    U_bg = U_daily.mean(
        "lon"
    )

    V_bg = V_daily.mean(
        "lon"
    )

    Z_zm = Z_daily.mean(
        "lon"
    )

    # U_bg, V_bg, Z_zm are (time, lat)
    # and are intentionally kept this way because
    # they are background/zonal-mean fields.

    # ========================================================
    # LATITUDE CHUNKING
    # ========================================================

    nlat = U_filt.sizes[
        "lat"
    ]

    boundaries = np.linspace(
        0,
        nlat,
        N_LAT_CHUNKS + 1,
        dtype=int
    )

    futures = []

    results = []

    # ========================================================
    # PROCESS LATITUDE CHUNKS
    # ========================================================

    with ProcessPoolExecutor(
        max_workers=NPROC
    ) as executor:

        for chunk_id in range(
            N_LAT_CHUNKS
        ):

            lat_start = boundaries[
                chunk_id
            ]

            lat_end = boundaries[
                chunk_id + 1
            ]

            future = executor.submit(
                process_lat_chunk,
                U_filt,
                V_filt,
                Z_filt,
                U_bg,
                V_bg,
                Z_zm,
                lat_start,
                lat_end,
                chunk_id,
                group_id
            )

            futures.append(
                future
            )

        # ----------------------------------------------------
        # Collect results
        # ----------------------------------------------------

        for future in futures:

            WAF_x, WAF_y = (
                future.result()
            )

            results.append(
                (
                    WAF_x,
                    WAF_y
                )
            )

    # ========================================================
    # COMBINE LATITUDE CHUNKS
    # ========================================================

    WAF_x_list = [
        item[0]
        for item in results
    ]

    WAF_y_list = [
        item[1]
        for item in results
    ]

    WAF_x = xr.concat(
        WAF_x_list,
        dim="lat"
    )

    WAF_y = xr.concat(
        WAF_y_list,
        dim="lat"
    )

    # Sort latitude
    WAF_x = WAF_x.sortby(
        "lat"
    )

    WAF_y = WAF_y.sortby(
        "lat"
    )

    # ========================================================
    # EXPLICIT FINAL DIMENSION ORDER
    # ========================================================

    WAF_x = WAF_x.transpose(
        "time",
        "lat",
        "lon"
    )

    WAF_y = WAF_y.transpose(
        "time",
        "lat",
        "lon"
    )

    print(
        "Final WAF dimensions:"
    )

    print(
        "WAF_x:",
        WAF_x.dims
    )

    print(
        "WAF_y:",
        WAF_y.dims
    )

    # ========================================================
    # WAF DIAGNOSTICS
    # ========================================================

    print(
        "Final WAF NaNs:"
    )

    print(
        "WAF_x:",
        int(
            np.isnan(
                WAF_x.values
            ).sum()
        )
    )

    print(
        "WAF_y:",
        int(
            np.isnan(
                WAF_y.values
            ).sum()
        )
    )

    # ========================================================
    # OUTPUT WAF
    # ========================================================

    print(
        "Writing WAF..."
    )

    ds_out = xr.Dataset(
        {
            "WAF_x": WAF_x,
            "WAF_y": WAF_y
        }
    )

    # Explicitly guarantee ALL output variables
    # are (time, lat, lon)
    for var in ds_out.data_vars:
        ds_out[var] = (
            ds_out[var]
            .transpose(
                "time",
                "lat",
                "lon"
            )
        )

    ds_out[
        "WAF_x"
    ].attrs[
        "description"
    ] = (
        "200-hPa Takaya-Nakamura "
        "wave activity flux, zonal component"
    )

    ds_out[
        "WAF_y"
    ].attrs[
        "description"
    ] = (
        "200-hPa Takaya-Nakamura "
        "wave activity flux, meridional component"
    )

    ds_out.attrs[
        "temporal_filter"
    ] = (
        "8-day Lanczos low-pass"
    )

    ds_out.attrs[
        "temporal_boundary"
    ] = (
        "symmetric reflective"
    )

    ds_out.attrs[
        "spatial_filter"
    ] = (
        "Global zonal wavenumber <= 15"
    )

    ds_out.attrs[
        "weak_wind_treatment"
    ] = (
        "Background wind magnitude below "
        "1 m/s set to 1 m/s in WAF denominator"
    )

    ds_out.to_netcdf(
        output_file
    )

    ds_out.close()

    print(
        "WAF written:"
    )

    print(
        output_file
    )

    print(
        f"Finished group {group_id}"
    )

    return output_file


# ============================================================
# SPLIT FILES INTO GROUPS
# ============================================================

groups = [
    files[i:i + FILES_PER_GROUP]
    for i in range(
        0,
        len(files),
        FILES_PER_GROUP
    )
]


# ============================================================
# PROCESS GROUPS
# ============================================================

output_files = []

for group_id, file_group in enumerate(
    groups
):

    print(
        "\n"
        "################################################"
    )

    print(
        f"GROUP {group_id}"
    )

    print(
        "Files:"
    )

    for f in file_group:
        print(
            "  ",
            f
        )

    print(
        "################################################"
    )

    output_file = process_group(
        group_id,
        file_group
    )

    output_files.append(
        output_file
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n========================================"
)

print(
    "ALL GROUPS FINISHED"
)

print(
    "========================================"
)

for f in output_files:

    print(
        f
    )
