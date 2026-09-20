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
#
# Input after resampling:
#     12-hourly = 0.5 day
#
# Cutoff:
#     8 days
#
# NWT:
#     number of Lanczos weights
# ------------------------------------------------------------

CUTOFF_DAYS = 8.0
DT_DAYS = 0.5

# 61-point filter = 30.5-day total window
NWT = 61


# ============================================================
# LANCZOS FILTER WEIGHTS
# ============================================================

def lanczos_weights(nwt, cutoff_days, dt_days):

    """
    Generate Lanczos low-pass filter weights.

    nwt:
        Odd number of weights.

    cutoff_days:
        Cutoff period in days.

    dt_days:
        Sampling interval in days.
    """

    if nwt % 2 == 0:
        raise ValueError(
            "nwt must be odd"
        )

    m = (nwt - 1) // 2

    # Cutoff frequency [cycles/day]
    fc = 1.0 / cutoff_days

    # Cutoff frequency in cycles/sample
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

    """
    Apply Lanczos filter along one dimension.

    IMPORTANT:
    Do not perform arithmetic on the time coordinate.
    This works with cftime.DatetimeNoLeap.
    """

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

        # ====================================================
        # 3-HOURLY -> 12-HOURLY
        #
        # Average four 3-hourly values.
        # ====================================================

        u12 = (
            u
            .resample(
                time="12h"
            )
            .mean()
        )

        v12 = (
            v
            .resample(
                time="12h"
            )
            .mean()
        )

        # ====================================================
        # COORDINATES
        # ====================================================

        lat_deg = ds["lat"]

        lon_deg = ds["lon"]

        lat_rad = np.deg2rad(
            lat_deg
        )

        lat_da = xr.DataArray(
            lat_rad,
            coords={
                "lat": lat_deg
            },
            dims=["lat"]
        )

        # ====================================================
        # ZONAL-MEAN PERTURBATIONS
        #
        # u' = u - zonal mean(u)
        # v' = v - zonal mean(v)
        # ====================================================

        u_zm = u12.mean(
            "lon"
        )

        v_zm = v12.mean(
            "lon"
        )

        u_p = (
            u12
            - u_zm
        )

        v_p = (
            v12
            - v_zm
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
        # STREAMFUNCTION APPROXIMATION
        #
        # psi' ≈ u'/f
        # ====================================================

        psi = (
            u_p
            / f_da
        )

        # ====================================================
        # SPATIAL DERIVATIVES
        #
        # x = a cos(phi) lambda
        # y = a phi
        #
        # Coordinates are degrees, so convert to radians.
        # ====================================================

        dpsi_dlon = (
            psi
            .differentiate("lon")
        )

        dpsi_dlat = (
            psi
            .differentiate("lat")
        )

        # meters per degree longitude
        dx_dlon = (
            a
            * np.cos(lat_da)
            * np.pi
            / 180.0
        )

        # meters per degree latitude
        dy_dlat = (
            a
            * np.pi
            / 180.0
        )

        dpsi_dx = (
            dpsi_dlon
            /
            dx_dlon
        )

        dpsi_dy = (
            dpsi_dlat
            /
            dy_dlat
        )

        # ====================================================
        # INSTANTANEOUS WAF
        #
        # Calculated at every 12-hourly timestep.
        # ====================================================

        WAF_x = (
            -dpsi_dx
            * dpsi_dy
        )

        WAF_y = (
            0.5
            *
            (
                dpsi_dx**2
                -
                dpsi_dy**2
            )
        )

        # ====================================================
        # LANCZOS FILTER
        #
        # Filter WAF itself.
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
        # MEAN WAF
        #
        # Mean of the low-pass filtered WAF.
        # ====================================================

        WAF_x_LP_mean = (
            WAF_x_LP.mean(
                "time",
                skipna=True
            )
        )

        WAF_y_LP_mean = (
            WAF_y_LP.mean(
                "time",
                skipna=True
            )
        )

        # ====================================================
        # MEAN RAW WAF
        # ====================================================

        WAF_x_mean = (
            WAF_x.mean(
                "time"
            )
        )

        WAF_y_mean = (
            WAF_y.mean(
                "time"
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
            f"{base}_200hpa_WAF_12h_LP8d.nc"
        )

        # ====================================================
        # OUTPUT DATASET
        # ====================================================

        out = xr.Dataset({

            # ------------------------------------------------
            # 12-hourly instantaneous WAF
            # ------------------------------------------------

            "WAF_x_200hpa":

                WAF_x,

            "WAF_y_200hpa":

                WAF_y,

            # ------------------------------------------------
            # 8-day low-pass WAF
            # ------------------------------------------------

            "WAF_x_200hpa_LP8d":

                WAF_x_LP,

            "WAF_y_200hpa_LP8d":

                WAF_y_LP,

            # ------------------------------------------------
            # Mean low-pass WAF
            # ------------------------------------------------

            "WAF_x_200hpa_LP8d_mean":

                WAF_x_LP_mean,

            "WAF_y_200hpa_LP8d_mean":

                WAF_y_LP_mean,

            # ------------------------------------------------
            # Mean raw WAF
            # ------------------------------------------------

            "WAF_x_200hpa_mean":

                WAF_x_mean,

            "WAF_y_200hpa_mean":

                WAF_y_mean

        })

        # ====================================================
        # ATTRIBUTES
        # ====================================================

        out.attrs = {

            "description":
                "200-hPa WAF from 3-hourly input",

            "processing":
                "3-hourly -> 12-hourly -> WAF -> 8-day Lanczos low-pass",

            "waf_method":
                "psi_prime = u_prime / f",

            "low_pass_filter":
                "Lanczos",

            "cutoff_period_days":
                CUTOFF_DAYS,

            "sampling_interval_hours":
                12,

            "filter_length":
                NWT
        }

        out[
            "WAF_x_200hpa"
        ].attrs = {

            "description":
                "Instantaneous 200-hPa WAF x-component",

            "time_resolution":
                "12-hourly"
        }

        out[
            "WAF_y_200hpa"
        ].attrs = {

            "description":
                "Instantaneous 200-hPa WAF y-component",

            "time_resolution":
                "12-hourly"
        }

        out[
            "WAF_x_200hpa_LP8d"
        ].attrs = {

            "description":
                "8-day low-pass filtered WAF x-component",

            "filter":
                "Lanczos",

            "cutoff_days":
                CUTOFF_DAYS
        }

        out[
            "WAF_y_200hpa_LP8d"
        ].attrs = {

            "description":
                "8-day low-pass filtered WAF y-component",

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
                "lev": {
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
        "200-hPa WAF calculation"
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
        "WAF:         12-hourly"
    )

    print(
        "Low-pass:    8 days"
    )

    print(
        f"Lanczos NWT: {NWT}"
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
