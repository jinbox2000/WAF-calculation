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


# ============================================================
# SETTINGS
# ============================================================

out_dir = os.getcwd()


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

files = sorted(
    glob.glob(
        "/pscratch/sd/x/xie7/ls4p/zppy/output/daily/*/*nc"
    )
)


# ------------------------------------------------------------
# Parallel workers
# ------------------------------------------------------------

NPROC = min(6, os.cpu_count())


# ------------------------------------------------------------
# Group size
# ------------------------------------------------------------

FILES_PER_GROUP = 5


# ------------------------------------------------------------
# Longitude decomposition
# ------------------------------------------------------------

N_LON_CHUNKS = 10


# ============================================================
# TEMPORAL SETTINGS
# ============================================================

INPUT_INTERVAL_HOURS = 3

DT_DAYS = INPUT_INTERVAL_HOURS / 24.0


# ============================================================
# BANDPASS FILTER
# ============================================================

SHORT_CUTOFF_DAYS = 2.0
LONG_CUTOFF_DAYS = 8.0

NWT = 61


# ============================================================
# LANCZOS LOW-PASS WEIGHTS
# ============================================================

def lanczos_lowpass_weights(
    nwt,
    cutoff_days,
    dt_days
):

    if nwt % 2 == 0:
        raise ValueError(
            "nwt must be odd"
        )

    m = (nwt - 1) // 2

    fc = 1.0 / cutoff_days

    fc_norm = fc * dt_days

    if fc_norm >= 0.5:
        raise ValueError(
            "Cutoff frequency is above Nyquist frequency"
        )

    k = np.arange(
        -m,
        m + 1,
        dtype=float
    )

    # --------------------------------------------------------
    # Ideal low-pass filter
    # --------------------------------------------------------

    h = np.zeros_like(k)

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

    # --------------------------------------------------------
    # Lanczos sigma factor
    # --------------------------------------------------------

    sigma = np.ones_like(k)

    sigma[nonzero] = (
        np.sin(
            np.pi
            * k[nonzero]
            /
            (m + 1)
        )
        /
        (
            np.pi
            * k[nonzero]
            /
            (m + 1)
        )
    )

    weights = h * sigma

    weights /= weights.sum()

    return weights


# ============================================================
# 2–8 DAY BANDPASS WEIGHTS
# ============================================================

def bandpass_weights(
    nwt,
    short_cutoff_days,
    long_cutoff_days,
    dt_days
):

    # LP 8 days
    w_long = lanczos_lowpass_weights(
        nwt,
        long_cutoff_days,
        dt_days
    )

    # LP 2 days
    w_short = lanczos_lowpass_weights(
        nwt,
        short_cutoff_days,
        dt_days
    )

    # --------------------------------------------------------
    # Bandpass:
    #
    # LP(8d) - LP(2d)
    # --------------------------------------------------------

    weights = (
        w_long
        -
        w_short
    )

    return weights


# ============================================================
# APPLY FILTER
# ============================================================

def apply_filter(
    da,
    dim,
    weights
):

    n = len(weights)

    if n % 2 == 0:
        raise ValueError(
            "Filter length must be odd"
        )

    weights_da = xr.DataArray(
        weights,
        dims=["window"]
    )

    filtered = (
        da
        .rolling(
            {
                dim: n
            },
            center=True,
            min_periods=n
        )
        .construct("window")
        .dot(weights_da)
    )

    return filtered


# ============================================================
# CREATE FIVE-FILE GROUPS
# ============================================================

def make_file_groups(
    file_list,
    group_size
):

    groups = []

    for i in range(
        0,
        len(file_list),
        group_size
    ):

        group = file_list[
            i:i + group_size
        ]

        if len(group) == 0:
            continue

        groups.append(
            group
        )

    return groups


# ============================================================
# GET LONGITUDE INFORMATION
# ============================================================

def get_lon_info(
    filename
):

    with xr.open_dataset(
        filename
    ) as ds:

        lon = ds["lon"].values

    nlon = len(lon)

    return lon, nlon


# ============================================================
# GET GLOBAL ZONAL MEAN
# ============================================================

def calculate_global_zonal_means(
    file_group
):

    U_parts = []
    V_parts = []
    Z_parts = []

    for fil in file_group:

        print(
            f"Reading global mean: {fil}"
        )

        ds = xr.open_dataset(
            fil
        )

        u = ds["U"].sel(
            lev=200,
            method="nearest"
        )

        v = ds["V"].sel(
            lev=200,
            method="nearest"
        )

        z = ds["Z3"].sel(
            lev=200,
            method="nearest"
        )

        u = u.transpose(
            "time",
            "lat",
            "lon"
        )

        v = v.transpose(
            "time",
            "lat",
            "lon"
        )

        z = z.transpose(
            "time",
            "lat",
            "lon"
        )

        # ----------------------------------------------------
        # Calculate zonal means
        # ----------------------------------------------------

        U_parts.append(
            u.mean(
                "lon"
            ).load()
        )

        V_parts.append(
            v.mean(
                "lon"
            ).load()
        )

        Z_parts.append(
            z.mean(
                "lon"
            ).load()
        )

        ds.close()

    # --------------------------------------------------------
    # Concatenate five files along time
    # --------------------------------------------------------

    U_bg = xr.concat(
        U_parts,
        dim="time"
    )

    V_bg = xr.concat(
        V_parts,
        dim="time"
    )

    Z_zm = xr.concat(
        Z_parts,
        dim="time"
    )

    return (
        U_bg,
        V_bg,
        Z_zm
    )


# ============================================================
# GET LONGITUDE CHUNKS
# ============================================================

def make_lon_chunks(
    nlon,
    nchunks
):

    edges = np.linspace(
        0,
        nlon,
        nchunks + 1,
        dtype=int
    )

    chunks = []

    for i in range(
        nchunks
    ):

        start = edges[i]
        end = edges[i + 1]

        if end <= start:
            continue

        chunks.append(
            (
                start,
                end
            )
        )

    return chunks


# ============================================================
# READ ONE LONGITUDE CHUNK FROM FIVE FILES
# ============================================================

def read_longitude_chunk(
    file_group,
    lon_start,
    lon_end,
    nlon
):

    # --------------------------------------------------------
    # Add longitude halo
    # --------------------------------------------------------

    LON_HALO = 2

    indices = np.arange(
        lon_start - LON_HALO,
        lon_end + LON_HALO
    )

    indices = (
        indices
        %
        nlon
    )

    datasets = []

    for fil in file_group:

        ds = xr.open_dataset(
            fil
        )

        u = ds["U"].sel(
            lev=200,
            method="nearest"
        )

        v = ds["V"].sel(
            lev=200,
            method="nearest"
        )

        z = ds["Z3"].sel(
            lev=200,
            method="nearest"
        )

        # ----------------------------------------------------
        # Read longitude chunk
        # ----------------------------------------------------

        u = u.isel(
            lon=indices
        )

        v = v.isel(
            lon=indices
        )

        z = z.isel(
            lon=indices
        )

        u = u.transpose(
            "time",
            "lat",
            "lon"
        )

        v = v.transpose(
            "time",
            "lat",
            "lon"
        )

        z = z.transpose(
            "time",
            "lat",
            "lon"
        )

        # ----------------------------------------------------
        # Load this chunk
        # ----------------------------------------------------

        u = u.load()
        v = v.load()
        z = z.load()

        datasets.append(
            (
                u,
                v,
                z
            )
        )

        ds.close()

    # --------------------------------------------------------
    # Concatenate five files along time
    # --------------------------------------------------------

    u = xr.concat(
        [x[0] for x in datasets],
        dim="time"
    )

    v = xr.concat(
        [x[1] for x in datasets],
        dim="time"
    )

    z = xr.concat(
        [x[2] for x in datasets],
        dim="time"
    )

    return (
        u,
        v,
        z
    )


# ============================================================
# CALCULATE WAF FOR ONE LONGITUDE CHUNK
# ============================================================

def calculate_waf_chunk(
    u,
    v,
    z,
    U_bg,
    V_bg,
    Z_zm,
    lat_deg,
    lon_start,
    lon_end
):

    LON_HALO = 2

    # ========================================================
    # COORDINATES
    # ========================================================

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

    # ========================================================
    # CORIOLIS
    # ========================================================

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

    f_da = xr.where(
        np.abs(f_da) < 1.0e-5,
        np.nan,
        f_da
    )

    # ========================================================
    # WAVE GEOPOTENTIAL HEIGHT
    # ========================================================

    z_p = (
        z
        -
        Z_zm
    )

    # ========================================================
    # STREAMFUNCTION
    # ========================================================

    psi = (
        g
        * z_p
        /
        f_da
    )

    # ========================================================
    # DERIVATIVES
    # ========================================================

    DEG2RAD = np.pi / 180.0

    psi_lon = (
        psi
        .differentiate(
            "lon"
        )
    )

    psi_lat = (
        psi
        .differentiate(
            "lat"
        )
    )

    psi_x = (
        psi_lon
        /
        DEG2RAD
        /
        (
            a
            * cos_lat
        )
    )

    psi_y = (
        psi_lat
        /
        DEG2RAD
        /
        a
    )

    # --------------------------------------------------------
    # Second derivatives
    # --------------------------------------------------------

    psi_xx = (
        psi_x
        .differentiate(
            "lon"
        )
        /
        DEG2RAD
        /
        (
            a
            * cos_lat
        )
    )

    psi_xy = (
        psi_x
        .differentiate(
            "lat"
        )
        /
        DEG2RAD
        /
        a
    )

    psi_yy = (
        psi_y
        .differentiate(
            "lat"
        )
        /
        DEG2RAD
        /
        a
    )

    # ========================================================
    # TN1997 TERMS
    # ========================================================

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

    # ========================================================
    # BACKGROUND FLOW
    # ========================================================

    U_mag = np.sqrt(
        U_bg**2
        +
        V_bg**2
    )

    U_mag = xr.where(
        U_mag < 1.0,
        np.nan,
        U_mag
    )

    # ========================================================
    # TN1997 WAF
    # ========================================================

    WAF_x = (
        (
            U_bg
            * term_xx
            +
            V_bg
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
            U_bg
            * term_xy
            +
            V_bg
            * term_yy
        )
        /
        (
            2.0
            * U_mag
        )
    )

    # ========================================================
    # TRIM LONGITUDE HALO
    # ========================================================

    WAF_x = WAF_x.isel(
        lon=slice(
            LON_HALO,
            LON_HALO
            +
            (
                lon_end
                -
                lon_start
            )
        )
    )

    WAF_y = WAF_y.isel(
        lon=slice(
            LON_HALO,
            LON_HALO
            +
            (
                lon_end
                -
                lon_start
            )
        )
    )

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
# PROCESS ONE FIVE-FILE + LONGITUDE CHUNK
# ============================================================

def process_chunk(
    args
):

    (
        group_id,
        file_group,
        lon_start,
        lon_end,
        nlon
    ) = args

    try:

        print(
            "\n=========================================="
        )

        print(
            f"GROUP {group_id}: "
            f"lon {lon_start}:{lon_end}"
        )

        print(
            "=========================================="
        )

        # ====================================================
        # GLOBAL ZONAL MEAN
        # ====================================================

        U_bg, V_bg, Z_zm = (
            calculate_global_zonal_means(
                file_group
            )
        )

        # ====================================================
        # READ FIVE FILES FOR THIS LONGITUDE CHUNK
        # ====================================================

        u, v, z = read_longitude_chunk(
            file_group,
            lon_start,
            lon_end,
            nlon
        )

        print(
            f"Concatenated time length: "
            f"{u.sizes['time']}"
        )

        print(
            f"Chunk shape: "
            f"{u.shape}"
        )

        # ====================================================
        # BANDPASS FILTER
        # ====================================================

        weights = bandpass_weights(
            NWT,
            SHORT_CUTOFF_DAYS,
            LONG_CUTOFF_DAYS,
            DT_DAYS
        )

        print(
            "Applying 2–8 day bandpass to U..."
        )

        u_filt = apply_filter(
            u,
            "time",
            weights
        )

        print(
            "Applying 2–8 day bandpass to V..."
        )

        v_filt = apply_filter(
            v,
            "time",
            weights
        )

        print(
            "Applying 2–8 day bandpass to Z..."
        )

        z_filt = apply_filter(
            z,
            "time",
            weights
        )

        # ====================================================
        # WAF
        # ====================================================

        lat_deg = u["lat"]

        WAF_x, WAF_y = (
            calculate_waf_chunk(
                u_filt,
                v_filt,
                z_filt,
                U_bg,
                V_bg,
                Z_zm,
                lat_deg,
                lon_start,
                lon_end
            )
        )

        # ====================================================
        # OUTPUT CHUNK
        # ====================================================

        chunk_name = (
            f"group{group_id:04d}"
            f"_lon{lon_start:04d}-{lon_end-1:04d}"
        )

        out_path = os.path.join(
            out_dir,
            f"{chunk_name}_WAF_TN1997_2-8d.nc"
        )

        out = xr.Dataset({

            "WAF_x_200hpa":
                WAF_x,

            "WAF_y_200hpa":
                WAF_y

        })

        out.attrs = {

            "description":
                "200-hPa Takaya-Nakamura TN1997 WAF",

            "processing":
                "Five-file concatenation -> 2-8 day Lanczos bandpass of U/V/Z -> TN1997 WAF",

            "waf_method":
                "Takaya-Nakamura (1997)",

            "pressure_level_hPa":
                200,

            "input_sampling_hours":
                INPUT_INTERVAL_HOURS,

            "bandpass":
                "2-8 days",

            "filter":
                "Lanczos",

            "filter_length":
                NWT,

            "longitude_chunk":
                f"{lon_start}:{lon_end}",

            "files_per_group":
                len(file_group)
        }

        out[
            "WAF_x_200hpa"
        ].attrs = {

            "description":
                "2-8 day bandpass-filtered 200-hPa TN1997 WAF zonal component",

            "units":
                "m2 s-2"
        }

        out[
            "WAF_y_200hpa"
        ].attrs = {

            "description":
                "2-8 day bandpass-filtered 200-hPa TN1997 WAF meridional component",

            "units":
                "m2 s-2"
        }

        out.to_netcdf(
            out_path
        )

        out.close()

        print(
            f"SAVED: {out_path}"
        )

        return out_path

    except Exception as e:

        return (
            f"FAILED group={group_id}, "
            f"lon={lon_start}:{lon_end}: "
            f"{repr(e)}"
        )


# ============================================================
# MERGE TEN LONGITUDE CHUNKS
# ============================================================

def merge_group(
    group_id,
    chunk_files,
    file_group
):

    valid_files = [
        f
        for f in chunk_files
        if os.path.exists(f)
    ]

    if len(valid_files) == 0:

        return (
            f"NO FILES TO MERGE: group {group_id}"
        )

    print(
        "\n=========================================="
    )

    print(
        f"MERGING GROUP {group_id}"
    )

    print(
        "=========================================="
    )

    datasets = []

    for f in valid_files:

        print(
            f"Opening: {f}"
        )

        datasets.append(
            xr.open_dataset(f)
        )

    merged = xr.concat(
        datasets,
        dim="lon"
    )

    merged = merged.sortby(
        "lon"
    )

    # ========================================================
    # OUTPUT NAME
    #
    # Keep the naming based on the FIRST INPUT FILE.
    #
    # Example:
    #
    # input:
    #   000000.nc
    #
    # output:
    #   000000_WAF_TN1997_2-8d.nc
    #
    # This prevents the original input from being overwritten.
    # ========================================================

    first_file = file_group[0]

    first_basename = os.path.basename(
        first_file
    )

    first_stem, first_ext = os.path.splitext(
        first_basename
    )

    out_path = os.path.join(
        out_dir,
        f"{first_stem}_WAF_TN1997_2-8d{first_ext}"
    )

    merged.to_netcdf(
        out_path
    )

    merged.close()

    for ds in datasets:
        ds.close()

    print(
        f"MERGED: {out_path}"
    )

    # ========================================================
    # NO FILE DELETION
    # ========================================================
    #
    # DO NOT DELETE:
    #
    #   - original five input files
    #   - longitude chunk files
    #   - merged output
    #
    # There is intentionally no os.remove() call.
    # ========================================================

    print(
        "No files deleted."
    )

    return out_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=========================================="
    )

    print(
        "200-hPa TN1997 WAF"
    )

    print(
        "Five-file groups + longitude decomposition"
    )

    print(
        "=========================================="
    )

    print(
        f"Input files: {len(files)}"
    )

    print(
        f"Files/group: {FILES_PER_GROUP}"
    )

    print(
        f"Longitude chunks: {N_LON_CHUNKS}"
    )

    print(
        f"Workers: {NPROC}"
    )

    print(
        "Input: 3-hourly"
    )

    print(
        "Filter: 2–8 day Lanczos bandpass"
    )

    print(
        f"Filter length: {NWT}"
    )

    print(
        "WAF: TN1997"
    )

    print(
        "Level: 200 hPa"
    )

    print(
        "IMPORTANT: NO FILES WILL BE DELETED"
    )

    print(
        "=========================================="
    )

    # ========================================================
    # MAKE FIVE-FILE GROUPS
    # ========================================================

    file_groups = make_file_groups(
        files,
        FILES_PER_GROUP
    )

    print(
        f"Number of five-file groups: "
        f"{len(file_groups)}"
    )

    # ========================================================
    # GET GLOBAL GRID
    # ========================================================

    if len(files) == 0:

        raise RuntimeError(
            "No input files found."
        )

    lon, nlon = get_lon_info(
        files[0]
    )

    lon_chunks = make_lon_chunks(
        nlon,
        N_LON_CHUNKS
    )

    print(
        f"Total longitude points: {nlon}"
    )

    print(
        f"Longitude chunks: {lon_chunks}"
    )

    # ========================================================
    # PROCESS EACH FIVE-FILE GROUP
    # ========================================================

    for group_id, file_group in enumerate(
        file_groups
    ):

        print(
            "\n\n##########################################"
        )

        print(
            f"START GROUP {group_id}"
        )

        print(
            "Files:"
        )

        for f in file_group:

            print(
                f"  {f}"
            )

        print(
            "##########################################"
        )

        # ----------------------------------------------------
        # Build tasks
        # ----------------------------------------------------

        tasks = []

        for (
            lon_start,
            lon_end
        ) in lon_chunks:

            tasks.append(
                (
                    group_id,
                    file_group,
                    lon_start,
                    lon_end,
                    nlon
                )
            )

        # ----------------------------------------------------
        # Parallel longitude processing
        # ----------------------------------------------------

        with ProcessPoolExecutor(
            max_workers=NPROC
        ) as executor:

            results = list(
                executor.map(
                    process_chunk,
                    tasks,
                    chunksize=1
                )
            )

        # ----------------------------------------------------
        # Find successful chunk files
        # ----------------------------------------------------

        chunk_files = []

        for result in results:

            print(
                result
            )

            if (
                isinstance(
                    result,
                    str
                )
                and
                result.endswith(
                    ".nc"
                )
                and
                os.path.exists(
                    result
                )
            ):

                chunk_files.append(
                    result
                )

        # ----------------------------------------------------
        # Merge longitude chunks
        # ----------------------------------------------------

        merged_file = merge_group(
            group_id,
            chunk_files,
            file_group
        )

        print(
            "\n=========================================="
        )

        print(
            f"GROUP {group_id} COMPLETE"
        )

        print(
            f"Output: {merged_file}"
        )

        print(
            "=========================================="
        )

    print(
        "\n=========================================="
    )

    print(
        "ALL GROUPS COMPLETED"
    )

    print(
        "NO FILES WERE DELETED"
    )

    print(
        "=========================================="
    )
