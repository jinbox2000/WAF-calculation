import xarray as xr
import numpy as np
import glob
import os

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

# Input files
files = sorted(
    glob.glob(
        "/pscratch/sd/x/xie7/ls4p/zppy/output/daily/*/*nc"
    )
)

# Number of parallel workers
NPROC = min(6, os.cpu_count())

# ------------------------------------------------------------
# Low-pass filter settings
# ------------------------------------------------------------

CUTOFF_DAYS = 8.0
DT_DAYS = 0.125

# 61-point filter
NWT = 61


# ============================================================
# LANCZOS FILTER WEIGHTS
# ============================================================

def lanczos_weights(
    nwt,
    cutoff_days,
    dt_days
):

    if nwt % 2 == 0:
        raise ValueError(
            "nwt must be odd"
        )

    m = (nwt - 1) // 2

    # Cutoff frequency [cycles/day]
    fc = 1.0 / cutoff_days

    # Cutoff frequency [cycles/sample]
    fc_norm = fc * dt_days

    k = np.arange(
        -m,
        m + 1,
        dtype=float
    )

    # ========================================================
    # Ideal low-pass filter
    # ========================================================

    h = np.zeros_like(k)

    h[k == 0] = (
        2.0
        * fc_norm
    )

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

    # ========================================================
    # Lanczos sigma factor
    # ========================================================

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

    # Normalize
    weights /= weights.sum()

    return weights


# ============================================================
# APPLY LANCZOS FILTER
# ============================================================

def lanczos_filter_1d(
    da,
    dim,
    weights
):

    n = len(weights)

    if n % 2 == 0:
        raise ValueError(
            "Number of weights must be odd"
        )

    weights_da = xr.DataArray(
        weights,
        dims=["window"]
    )

    filtered = (
        da
        .rolling(
            {dim: n},
            center=True,
            min_periods=n
        )
        .construct("window")
        .dot(weights_da)
    )

    return filtered


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(fil):

    try:

        print(
            f"Processing: {fil}"
        )

        # ====================================================
        # OPEN DATA
        # ====================================================

        ds = xr.open_dataset(
            fil
        )

        # ====================================================
        # CHECK VARIABLES
        # ====================================================

        if "lev" not in ds:
            ds.close()
            return (
                f"SKIP (no lev): {fil}"
            )

        if "U" not in ds:
            ds.close()
            return (
                f"SKIP (no U): {fil}"
            )

        if "V" not in ds:
            ds.close()
            return (
                f"SKIP (no V): {fil}"
            )

        if "Z3" not in ds:
            ds.close()
            return (
                f"SKIP (no Z3): {fil}"
            )

        if "time" not in ds.coords:
            ds.close()
            return (
                f"SKIP (no time): {fil}"
            )

        if "lat" not in ds.coords:
            ds.close()
            return (
                f"SKIP (no lat): {fil}"
            )

        if "lon" not in ds.coords:
            ds.close()
            return (
                f"SKIP (no lon): {fil}"
            )

        # ====================================================
        # 200 hPa
        # ====================================================

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

        # ====================================================
        # FORCE INPUT VARIABLES TO:
        #
        #     (time, lat, lon)
        #
        # ====================================================

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

        # ====================================================
        # COORDINATES
        # ====================================================

        lat_deg = ds["lat"]

        lon_deg = ds["lon"]

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

        # ====================================================
        # CORIOLIS PARAMETER
        # ====================================================

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

        # Avoid equatorial singularity
        f_da = xr.where(
            np.abs(f_da) < 1.0e-5,
            np.nan,
            f_da
        )

        # ====================================================
        # ZONAL-MEAN BACKGROUND FLOW
        #
        # Result:
        #     (time, lat)
        # ====================================================

        U_bg = u.mean(
            "lon"
        )

        V_bg = v.mean(
            "lon"
        )

        # ====================================================
        # GEOPOTENTIAL HEIGHT PERTURBATION
        #
        # Result:
        #     (time, lat, lon)
        # ====================================================

        z_zm = z.mean(
            "lon"
        )

        z_p = (
            z
            - z_zm
        )

        # ====================================================
        # PERTURBATION STREAMFUNCTION
        #
        # psi' = g Z' / f
        #
        # Result:
        #     (time, lat, lon)
        # ====================================================

        psi = (
            g
            * z_p
            / f_da
        )

        # ====================================================
        # FIRST DERIVATIVES
        # ====================================================

        dpsi_dlon = (
            psi
            .differentiate("lon")
            * 180.0
            / np.pi
        )

        dpsi_dlat = (
            psi
            .differentiate("lat")
            * 180.0
            / np.pi
        )

        # ====================================================
        # SECOND DERIVATIVES
        # ====================================================

        d2psi_dlon2 = (
            dpsi_dlon
            .differentiate("lon")
            * 180.0
            / np.pi
        )

        d2psi_dlat2 = (
            dpsi_dlat
            .differentiate("lat")
            * 180.0
            / np.pi
        )

        d2psi_dlondlat = (
            dpsi_dlat
            .differentiate("lon")
            * 180.0
            / np.pi
        )

        # ====================================================
        # TN01 TERMS
        # ====================================================

        term_xx = (
            dpsi_dlon**2
            -
            psi
            * d2psi_dlon2
        )

        term_xy = (
            dpsi_dlon
            * dpsi_dlat
            -
            psi
            * d2psi_dlondlat
        )

        term_yy = (
            dpsi_dlat**2
            -
            psi
            * d2psi_dlat2
        )

        # ====================================================
        # BACKGROUND WIND MAGNITUDE
        #
        # Result:
        #     (time, lat)
        # ====================================================

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

        # ====================================================
        # PRESSURE FACTOR
        # ====================================================

        p_lev = 200.0

        p = (
            p_lev
            /
            1000.0
        )

        # ====================================================
        # TN01 COEFFICIENT
        #
        # Result:
        #     (time, lat)
        # ====================================================

        coeff = (
            p
            * cos_lat
            /
            (
                2.0
                * U_mag
            )
        )

        # ====================================================
        # TN01 WAF
        #
        # Result:
        #     (time, lat, lon)
        # ====================================================

        WAF_x = (
            coeff
            /
            (
                a**2
                * cos_lat
            )
            *
            (
                (
                    U_bg
                    /
                    cos_lat
                )
                * term_xx
                +
                V_bg
                * term_xy
            )
        )

        WAF_y = (
            coeff
            /
            a**2
            *
            (
                (
                    U_bg
                    /
                    cos_lat
                )
                * term_xy
                +
                V_bg
                * term_yy
            )
        )

        # ====================================================
        # EXPLICITLY FORCE WAF DIMENSIONS
        #
        #     (time, lat, lon)
        # ====================================================

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

        # ====================================================
        # LANCZOS FILTER
        # ====================================================

        weights = lanczos_weights(
            NWT,
            CUTOFF_DAYS,
            DT_DAYS
        )

        WAF_x_LP = (
            lanczos_filter_1d(
                WAF_x,
                "time",
                weights
            )
        )

        WAF_y_LP = (
            lanczos_filter_1d(
                WAF_y,
                "time",
                weights
            )
        )

        # ====================================================
        # EXPLICITLY FORCE LOW-PASS WAF DIMENSIONS
        #
        #     (time, lat, lon)
        # ====================================================

        WAF_x_LP = WAF_x_LP.transpose(
            "time",
            "lat",
            "lon"
        )

        WAF_y_LP = WAF_y_LP.transpose(
            "time",
            "lat",
            "lon"
        )

        # ====================================================
        # TIME MEANS
        #
        # Result:
        #     (lat, lon)
        # ====================================================

        WAF_x_mean = (
            WAF_x.mean(
                "time",
                skipna=True
            )
            .transpose(
                "lat",
                "lon"
            )
        )

        WAF_y_mean = (
            WAF_y.mean(
                "time",
                skipna=True
            )
            .transpose(
                "lat",
                "lon"
            )
        )

        WAF_x_LP_mean = (
            WAF_x_LP.mean(
                "time",
                skipna=True
            )
            .transpose(
                "lat",
                "lon"
            )
        )

        WAF_y_LP_mean = (
            WAF_y_LP.mean(
                "time",
                skipna=True
            )
            .transpose(
                "lat",
                "lon"
            )
        )

        # ====================================================
        # OUTPUT FILE
        # ====================================================

        base = (
            os.path.basename(fil)
            .replace(
                ".nc",
                ""
            )
        )

        out_path = os.path.join(
            out_dir,
            f"{base}_200hpa_WAF_TN01_3h_LP8d.nc"
        )

        # ====================================================
        # OUTPUT DATASET
        # ====================================================

        out = xr.Dataset({

            # ------------------------------------------------
            # 3-hourly WAF
            #
            # (time, lat, lon)
            # ------------------------------------------------

            "WAF_x_200hpa":
                WAF_x,

            "WAF_y_200hpa":
                WAF_y,

            # ------------------------------------------------
            # 8-day low-pass WAF
            #
            # (time, lat, lon)
            # ------------------------------------------------

            "WAF_x_200hpa_LP8d":
                WAF_x_LP,

            "WAF_y_200hpa_LP8d":
                WAF_y_LP,

            # ------------------------------------------------
            # Time-mean low-pass WAF
            #
            # (lat, lon)
            # ------------------------------------------------

            "WAF_x_200hpa_LP8d_mean":
                WAF_x_LP_mean,

            "WAF_y_200hpa_LP8d_mean":
                WAF_y_LP_mean,

            # ------------------------------------------------
            # Time-mean raw WAF
            #
            # (lat, lon)
            # ------------------------------------------------

            "WAF_x_200hpa_mean":
                WAF_x_mean,

            "WAF_y_200hpa_mean":
                WAF_y_mean

        })

        # ====================================================
        # VERIFY OUTPUT DIMENSIONS
        # ====================================================

        expected_time_dim = (
            "time",
            "lat",
            "lon"
        )

        expected_mean_dim = (
            "lat",
            "lon"
        )

        assert (
            out[
                "WAF_x_200hpa"
            ].dims
            ==
            expected_time_dim
        )

        assert (
            out[
                "WAF_y_200hpa"
            ].dims
            ==
            expected_time_dim
        )

        assert (
            out[
                "WAF_x_200hpa_LP8d"
            ].dims
            ==
            expected_time_dim
        )

        assert (
            out[
                "WAF_y_200hpa_LP8d"
            ].dims
            ==
            expected_time_dim
        )

        assert (
            out[
                "WAF_x_200hpa_mean"
            ].dims
            ==
            expected_mean_dim
        )

        assert (
            out[
                "WAF_y_200hpa_mean"
            ].dims
            ==
            expected_mean_dim
        )

        # ====================================================
        # ATTRIBUTES
        # ====================================================

        out.attrs = {

            "description":
                "200-hPa Takaya-Nakamura wave-activity flux from 3-hourly E3SM data",

            "processing":
                "3-hourly -> TN01 WAF -> 8-day Lanczos low-pass",

            "waf_method":
                "Takaya-Nakamura (2001) horizontal WAF",

            "streamfunction":
                "psi_prime = g * Z_prime / f",

            "background_flow":
                "instantaneous zonal-mean 200-hPa U and V",

            "geopotential_height":
                "Z3 assumed to be geopotential height in meters",

            "low_pass_filter":
                "Lanczos",

            "cutoff_period_days":
                CUTOFF_DAYS,

            "sampling_interval_hours":
                3,

            "filter_length":
                NWT,

            "pressure_level_hPa":
                200
        }

        # ====================================================
        # VARIABLE ATTRIBUTES
        # ====================================================

        out[
            "WAF_x_200hpa"
        ].attrs = {

            "description":
                "Instantaneous 200-hPa TN01 WAF zonal component",

            "units":
                "m2 s-2",

            "time_resolution":
                "3-hourly"
        }

        out[
            "WAF_y_200hpa"
        ].attrs = {

            "description":
                "Instantaneous 200-hPa TN01 WAF meridional component",

            "units":
                "m2 s-2",

            "time_resolution":
                "3-hourly"
        }

        out[
            "WAF_x_200hpa_LP8d"
        ].attrs = {

            "description":
                "8-day low-pass filtered 200-hPa TN01 WAF zonal component",

            "units":
                "m2 s-2",

            "filter":
                "Lanczos",

            "cutoff_days":
                CUTOFF_DAYS,

            "time_resolution":
                "3-hourly"
        }

        out[
            "WAF_y_200hpa_LP8d"
        ].attrs = {

            "description":
                "8-day low-pass filtered 200-hPa TN01 WAF meridional component",

            "units":
                "m2 s-2",

            "filter":
                "Lanczos",

            "cutoff_days":
                CUTOFF_DAYS,

            "time_resolution":
                "3-hourly"
        }

        out[
            "WAF_x_200hpa_mean"
        ].attrs = {

            "description":
                "Time-mean 200-hPa TN01 WAF zonal component",

            "units":
                "m2 s-2"
        }

        out[
            "WAF_y_200hpa_mean"
        ].attrs = {

            "description":
                "Time-mean 200-hPa TN01 WAF meridional component",

            "units":
                "m2 s-2"
        }

        out[
            "WAF_x_200hpa_LP8d_mean"
        ].attrs = {

            "description":
                "Time-mean 8-day low-pass 200-hPa TN01 WAF zonal component",

            "units":
                "m2 s-2",

            "filter":
                "Lanczos",

            "cutoff_days":
                CUTOFF_DAYS
        }

        out[
            "WAF_y_200hpa_LP8d_mean"
        ].attrs = {

            "description":
                "Time-mean 8-day low-pass 200-hPa TN01 WAF meridional component",

            "units":
                "m2 s-2",

            "filter":
                "Lanczos",

            "cutoff_days":
                CUTOFF_DAYS
        }

        # ====================================================
        # SAVE
        # ====================================================

        out.to_netcdf(
            out_path,
            encoding={
                var: {
                    "_FillValue": 1.0e36
                }
                for var in out.data_vars
            }
            | {
                "lat": {
                    "_FillValue": 1.0e36
                },
                "lon": {
                    "_FillValue": 1.0e36
                },
                "time": {
                    "dtype": "float64"
                }
            }
        )

        ds.close()

        return (
            f"SAVED: {out_path}"
        )

    except Exception as e:

        return (
            f"FAILED: {fil} | {repr(e)}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=========================================="
    )

    print(
        "200-hPa Takaya-Nakamura WAF calculation"
    )

    print(
        "=========================================="
    )

    print(
        f"Input files: {len(files)}"
    )

    print(
        f"Workers: {NPROC}"
    )

    print(
        "Input:       3-hourly"
    )

    print(
        "WAF:         TN01"
    )

    print(
        "Level:       200 hPa"
    )

    print(
        "Low-pass:    8 days"
    )

    print(
        f"Lanczos NWT: {NWT}"
    )

    print(
        "Time-dependent output: (time, lat, lon)"
    )

    print(
        "Mean output:           (lat, lon)"
    )

    print(
        "=========================================="
    )

    with ProcessPoolExecutor(
        max_workers=NPROC
    ) as executor:

        results = list(
            executor.map(
                process_file,
                files,
                chunksize=1
            )
        )

    print(
        "=========================================="
    )

    for r in results:
        print(r)

    print(
        "=========================================="
    )

    print(
        "ALL FILES COMPLETED"
    )

    print(
        "=========================================="
    )
