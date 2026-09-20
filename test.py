import xarray as xr
import numpy as np
import glob
import os
from concurrent.futures import ProcessPoolExecutor


# ============================================================
# Constants
# ============================================================

a = 6.371e6
Omega = 7.292e-5
g = 9.80665

out_dir = os.getcwd()

files = sorted(
    glob.glob(
        "/pscratch/sd/x/xie7/ls4p/zppy/output/daily/*/*nc"
    )
)

NPROC = min(6, os.cpu_count())


# ============================================================
# Temporal settings
# ============================================================

INPUT_INTERVAL_HOURS = 3
WAF_INTERVAL_HOURS = 12

N_AVG = WAF_INTERVAL_HOURS // INPUT_INTERVAL_HOURS

DT_DAYS = 0.5


# ============================================================
# Lanczos low-pass filter settings
# ============================================================

CUTOFF_DAYS = 8.0
NWT = 61


# ============================================================
# Lanczos weights
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

    fc = 1.0 / cutoff_days
    fc_norm = fc * dt_days

    k = np.arange(
        -m,
        m + 1,
        dtype=float
    )

    h = np.zeros_like(k)

    h[k == 0] = (
        2.0 * fc_norm
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

    weights /= weights.sum()

    return weights


# ============================================================
# Lanczos filter
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
# Print diagnostic range
# ============================================================

def print_range(
    name,
    da
):

    try:

        vmin = float(
            da.min(skipna=True)
        )

        vmax = float(
            da.max(skipna=True)
        )

        vmean = float(
            da.mean(skipna=True)
        )

        print(
            f"{name:15s}: "
            f"min={vmin:.6e}, "
            f"max={vmax:.6e}, "
            f"mean={vmean:.6e}",
            flush=True
        )

    except Exception as e:

        print(
            f"{name:15s}: "
            f"unable to calculate range: {e}",
            flush=True
        )


# ============================================================
# Process one file
# ============================================================

def process_file(fil):

    try:

        print(
            "",
            flush=True
        )

        print(
            "==================================================",
            flush=True
        )

        print(
            f"Processing: {fil}",
            flush=True
        )

        print(
            "==================================================",
            flush=True
        )


        # ====================================================
        # Open dataset
        # ====================================================

        ds = xr.open_dataset(
            fil
        )


        # ====================================================
        # Check variables
        # ====================================================

        required_vars = [
            "U",
            "V",
            "Z3",
            "time",
            "lat",
            "lon"
        ]

        for var in required_vars:

            if var not in ds:

                raise ValueError(
                    f"Missing variable/dimension: {var}"
                )


        # ====================================================
        # Print Z3 metadata
        # ====================================================

        print(
            "",
            flush=True
        )

        print(
            "Z3 metadata:",
            flush=True
        )

        print(
            ds["Z3"].attrs,
            flush=True
        )


        # ====================================================
        # Select 200 hPa
        # ====================================================

        u = (
            ds["U"]
            .sel(
                lev=200,
                method="nearest"
            )
        )

        v = (
            ds["V"]
            .sel(
                lev=200,
                method="nearest"
            )
        )

        z = (
            ds["Z3"]
            .sel(
                lev=200,
                method="nearest"
            )
        )


        # ====================================================
        # Dimension order
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
        # Original 3-hourly diagnostics
        # ====================================================

        print(
            "",
            flush=True
        )

        print(
            "Original 3-hourly fields:",
            flush=True
        )

        print_range(
            "U",
            u
        )

        print_range(
            "V",
            v
        )

        print_range(
            "Z3",
            z
        )


        # ====================================================
        # 3-hourly -> 12-hourly averaging
        # ====================================================

        ntime = u.sizes["time"]

        n_complete = (
            ntime // N_AVG
        ) * N_AVG


        if n_complete < N_AVG:

            raise ValueError(
                "Not enough time steps "
                "for 12-hour averaging"
            )


        u = u.isel(
            time=slice(
                0,
                n_complete
            )
        )

        v = v.isel(
            time=slice(
                0,
                n_complete
            )
        )

        z = z.isel(
            time=slice(
                0,
                n_complete
            )
        )


        u = (
            u
            .coarsen(
                time=N_AVG,
                boundary="trim"
            )
            .mean()
        )

        v = (
            v
            .coarsen(
                time=N_AVG,
                boundary="trim"
            )
            .mean()
        )

        z = (
            z
            .coarsen(
                time=N_AVG,
                boundary="trim"
            )
            .mean()
        )


        # ====================================================
        # Ensure dimension order
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
        # Coordinates
        # ====================================================

        lat_deg = ds["lat"]
        lon_deg = ds["lon"]


        lat_rad = xr.DataArray(
            np.deg2rad(
                lat_deg.values
            ),
            coords={
                "lat": lat_deg
            },
            dims=["lat"]
        )


        # ====================================================
        # cos(latitude)
        # ====================================================

        cos_lat = xr.DataArray(
            np.cos(
                lat_rad.values
            ),
            coords={
                "lat": lat_deg
            },
            dims=["lat"]
        )


        cos_lat = xr.where(
            np.abs(cos_lat) < 1.0e-6,
            np.nan,
            cos_lat
        )


        # ====================================================
        # Coriolis parameter
        # ====================================================

        f = (
            2.0
            * Omega
            * np.sin(
                lat_rad
            )
        )


        f_da = xr.DataArray(
            f.values,
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
        # Zonal-mean background flow
        # ====================================================

        U_bg = (
            u
            .mean(
                "lon"
            )
        )

        V_bg = (
            v
            .mean(
                "lon"
            )
        )


        # ====================================================
        # Zonal-mean geopotential height
        # ====================================================

        z_zm = (
            z
            .mean(
                "lon"
            )
        )


        # ====================================================
        # Perturbation geopotential height
        # ====================================================

        z_p = (
            z
            -
            z_zm
        )


        # ====================================================
        # Diagnostics: perturbation height
        # ====================================================

        print(
            "",
            flush=True
        )

        print(
            "After 12-hour averaging:",
            flush=True
        )

        print_range(
            "Z'",
            z_p
        )

        print_range(
            "U_bg",
            U_bg
        )

        print_range(
            "V_bg",
            V_bg
        )


        # ====================================================
        # Streamfunction
        #
        # Assumption:
        #
        # Z3 is geopotential height in meters.
        #
        # psi' = g Z' / f
        # ====================================================

        psi = (
            g
            * z_p
            / f_da
        )


        # ====================================================
        # Diagnostic: streamfunction
        # ====================================================

        print(
            "",
            flush=True
        )

        print(
            "Streamfunction diagnostics:",
            flush=True
        )

        print_range(
            "psi",
            psi
        )


        # ====================================================
        # Physical derivatives
        #
        # x = a cos(phi) lambda
        # y = a phi
        #
        # Coordinates are degrees.
        # ====================================================

        degree_to_radian = (
            np.pi / 180.0
        )


        # ====================================================
        # dpsi/dx
        # ====================================================

        psi_lon = (
            psi
            .differentiate(
                "lon"
            )
        )


        psi_x = (
            psi_lon
            * degree_to_radian
            /
            (
                a
                * cos_lat
            )
        )


        # ====================================================
        # dpsi/dy
        # ====================================================

        psi_lat = (
            psi
            .differentiate(
                "lat"
            )
        )


        psi_y = (
            psi_lat
            * degree_to_radian
            /
            a
        )


        # ====================================================
        # Geostrophic perturbation winds
        #
        # u' = -dpsi/dy
        # v' =  dpsi/dx
        # ====================================================

        u_p = -psi_y

        v_p = psi_x


        # ====================================================
        # Diagnostics: perturbation winds
        # ====================================================

        print(
            "",
            flush=True
        )

        print(
            "Perturbation wind diagnostics:",
            flush=True
        )

        print_range(
            "psi_x",
            psi_x
        )

        print_range(
            "psi_y",
            psi_y
        )

        print_range(
            "u'",
            u_p
        )

        print_range(
            "v'",
            v_p
        )


        # ====================================================
        # d2psi/dx2
        # ====================================================

        psi_xx = (
            psi_x
            .differentiate(
                "lon"
            )
            * degree_to_radian
            /
            (
                a
                * cos_lat
            )
        )


        # ====================================================
        # d2psi/dxdy
        # ====================================================

        psi_xy = (
            psi_x
            .differentiate(
                "lat"
            )
            * degree_to_radian
            /
            a
        )


        # ====================================================
        # d2psi/dy2
        # ====================================================

        psi_yy = (
            psi_y
            .differentiate(
                "lat"
            )
            * degree_to_radian
            /
            a
        )


        # ====================================================
        # TN1997 quadratic terms
        #
        # A = psi_x^2 - psi psi_xx
        #
        # B = psi_x psi_y - psi psi_xy
        #
        # C = psi_y^2 - psi psi_yy
        # ====================================================

        term_xx = (
            psi_x**2
            -
            psi * psi_xx
        )


        term_xy = (
            psi_x * psi_y
            -
            psi * psi_xy
        )


        term_yy = (
            psi_y**2
            -
            psi * psi_yy
        )


        # ====================================================
        # Diagnostics: quadratic terms
        # ====================================================

        print(
            "",
            flush=True
        )

        print(
            "TN1997 quadratic-term diagnostics:",
            flush=True
        )

        print_range(
            "term_xx",
            term_xx
        )

        print_range(
            "term_xy",
            term_xy
        )

        print_range(
            "term_yy",
            term_yy
        )


        # ====================================================
        # Background wind magnitude
        # ====================================================

        U_mag = np.sqrt(
            U_bg**2
            +
            V_bg**2
        )


        # Avoid division by very weak flow

        U_mag = xr.where(
            U_mag < 1.0,
            np.nan,
            U_mag
        )


        # ====================================================
        # Diagnostic: background-flow magnitude
        # ====================================================

        print(
            "",
            flush=True
        )

        print(
            "Background flow:",
            flush=True
        )

        print_range(
            "|U_bg|",
            U_mag
        )


        # ====================================================
        # TN1997 WAF zonal component
        #
        # W_x =
        #
        # 1/(2|U|)
        #
        # [ U A + V B ]
        # ====================================================

        WAF_x = (
            1.0
            /
            (
                2.0
                * U_mag
            )
        ) * (
            U_bg * term_xx
            +
            V_bg * term_xy
        )


        # ====================================================
        # TN1997 WAF meridional component
        #
        # W_y =
        #
        # 1/(2|U|)
        #
        # [ U B + V C ]
        # ====================================================

        WAF_y = (
            1.0
            /
            (
                2.0
                * U_mag
            )
        ) * (
            U_bg * term_xy
            +
            V_bg * term_yy
        )


        # ====================================================
        # WAF diagnostics
        # ====================================================

        print(
            "",
            flush=True
        )

        print(
            "TN1997 WAF diagnostics:",
            flush=True
        )

        print_range(
            "WAF_x",
            WAF_x
        )

        print_range(
            "WAF_y",
            WAF_y
        )


        # ====================================================
        # Dimension order
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
        # Lanczos low-pass filter
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
        # Time means
        # ====================================================

        WAF_x_LP_mean = (
            WAF_x_LP
            .mean(
                "time",
                skipna=True
            )
        )

        WAF_y_LP_mean = (
            WAF_y_LP
            .mean(
                "time",
                skipna=True
            )
        )

        WAF_x_mean = (
            WAF_x
            .mean(
                "time",
                skipna=True
            )
        )

        WAF_y_mean = (
            WAF_y
            .mean(
                "time",
                skipna=True
            )
        )


        # ====================================================
        # Output filename
        # ====================================================

        base = os.path.basename(
            fil
        )

        if base.endswith(
            ".nc"
        ):

            base = base[:-3]


        outfile = os.path.join(
            out_dir,
            f"{base}_200hpa_WAF_TN97_12h_LP8d.nc"
        )


        # ====================================================
        # Output dataset
        # ====================================================

        out = xr.Dataset(

            {

                "WAF_x_200hpa":
                    WAF_x,

                "WAF_y_200hpa":
                    WAF_y,

                "WAF_x_200hpa_LP8d":
                    WAF_x_LP,

                "WAF_y_200hpa_LP8d":
                    WAF_y_LP,

                "WAF_x_200hpa_LP8d_mean":
                    WAF_x_LP_mean,

                "WAF_y_200hpa_LP8d_mean":
                    WAF_y_LP_mean,

                "WAF_x_200hpa_mean":
                    WAF_x_mean,

                "WAF_y_200hpa_mean":
                    WAF_y_mean

            },

            coords={

                "time":
                    WAF_x.time,

                "lat":
                    lat_deg,

                "lon":
                    lon_deg

            }

        )


        # ====================================================
        # Global attributes
        # ====================================================

        out.attrs.update({

            "description":
                "200-hPa Takaya-Nakamura (1997) "
                "stationary Rossby-wave activity flux "
                "from 12-hourly averaged E3SM data",

            "processing":
                "3-hourly -> 12-hourly averaging "
                "-> TN1997 WAF "
                "-> 8-day Lanczos low-pass",

            "waf_method":
                "Takaya-Nakamura (1997) stationary "
                "Rossby-wave WAF",

            "streamfunction":
                "psi_prime = g * Z_prime / f",

            "background_flow":
                "instantaneous zonal-mean 200-hPa U and V",

            "geopotential_height":
                "Z3 assumed to be geopotential height in meters",

            "coordinate_derivatives":
                "physical derivatives using "
                "x=a*cos(latitude)*lambda and y=a*latitude",

            "low_pass_filter":
                "Lanczos",

            "cutoff_period_days":
                CUTOFF_DAYS,

            "input_sampling_hours":
                INPUT_INTERVAL_HOURS,

            "WAF_sampling_hours":
                WAF_INTERVAL_HOURS,

            "filter_length":
                NWT,

            "pressure_level_hPa":
                200

        })


        # ====================================================
        # Variable attributes
        # ====================================================

        out[
            "WAF_x_200hpa"
        ].attrs.update({

            "long_name":
                "200-hPa TN1997 wave-activity flux "
                "zonal component",

            "units":
                "m2 s-2"

        })


        out[
            "WAF_y_200hpa"
        ].attrs.update({

            "long_name":
                "200-hPa TN1997 wave-activity flux "
                "meridional component",

            "units":
                "m2 s-2"

        })


        out[
            "WAF_x_200hpa_LP8d"
        ].attrs.update({

            "long_name":
                "8-day low-pass filtered "
                "TN1997 WAF zonal component",

            "units":
                "m2 s-2"

        })


        out[
            "WAF_y_200hpa_LP8d"
        ].attrs.update({

            "long_name":
                "8-day low-pass filtered "
                "TN1997 WAF meridional component",

            "units":
                "m2 s-2"

        })


        out[
            "WAF_x_200hpa_LP8d_mean"
        ].attrs.update({

            "long_name":
                "time-mean 8-day low-pass "
                "TN1997 WAF zonal component",

            "units":
                "m2 s-2"

        })


        out[
            "WAF_y_200hpa_LP8d_mean"
        ].attrs.update({

            "long_name":
                "time-mean 8-day low-pass "
                "TN1997 WAF meridional component",

            "units":
                "m2 s-2"

        })


        out[
            "WAF_x_200hpa_mean"
        ].attrs.update({

            "long_name":
                "time-mean TN1997 WAF zonal component",

            "units":
                "m2 s-2"

        })


        out[
            "WAF_y_200hpa_mean"
        ].attrs.update({

            "long_name":
                "time-mean TN1997 WAF meridional component",

            "units":
                "m2 s-2"

        })


        # ====================================================
        # Fill values
        # ====================================================

        fill_value = 1.0e36


        encoding = {

            "WAF_x_200hpa": {
                "_FillValue": fill_value
            },

            "WAF_y_200hpa": {
                "_FillValue": fill_value
            },

            "WAF_x_200hpa_LP8d": {
                "_FillValue": fill_value
            },

            "WAF_y_200hpa_LP8d": {
                "_FillValue": fill_value
            },

            "WAF_x_200hpa_LP8d_mean": {
                "_FillValue": fill_value
            },

            "WAF_y_200hpa_LP8d_mean": {
                "_FillValue": fill_value
            },

            "WAF_x_200hpa_mean": {
                "_FillValue": fill_value
            },

            "WAF_y_200hpa_mean": {
                "_FillValue": fill_value
            },

            "lat": {
                "_FillValue": fill_value
            },

            "lon": {
                "_FillValue": fill_value
            }

        }


        # ====================================================
        # Write output
        # ====================================================

        out.to_netcdf(
            outfile,
            encoding=encoding
        )


        # ====================================================
        # Close
        # ====================================================

        ds.close()

        out.close()


        print(
            "",
            flush=True
        )

        print(
            f"SAVED: {outfile}",
            flush=True
        )

        print(
            "==================================================",
            flush=True
        )


        return (
            f"SAVED: {outfile}"
        )


    except Exception as e:

        print(
            "",
            flush=True
        )

        print(
            f"FAILED: {fil}",
            flush=True
        )

        print(
            repr(e),
            flush=True
        )

        print(
            "==================================================",
            flush=True
        )

        return (
            f"FAILED: {fil} -> {repr(e)}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "=================================================="
    )

    print(
        "200-hPa Takaya-Nakamura WAF calculation"
    )

    print(
        "=================================================="
    )

    print(
        f"Number of files: {len(files)}"
    )

    print(
        f"Input sampling: "
        f"{INPUT_INTERVAL_HOURS}-hourly"
    )

    print(
        f"Averaging: "
        f"{INPUT_INTERVAL_HOURS}-hourly -> "
        f"{WAF_INTERVAL_HOURS}-hourly"
    )

    print(
        "WAF formulation: "
        "Takaya-Nakamura 1997 (TN97)"
    )

    print(
        "Background flow: "
        "instantaneous zonal mean"
    )

    print(
        f"Low-pass cutoff: "
        f"{CUTOFF_DAYS} days"
    )

    print(
        f"Lanczos window: "
        f"{NWT}"
    )

    print(
        f"Number of processes: "
        f"{NPROC}"
    )

    print(
        "Diagnostics: ENABLED"
    )

    print(
        "=================================================="
    )


    if len(files) == 0:

        print(
            "No input files found."
        )

    else:

        with ProcessPoolExecutor(
            max_workers=NPROC
        ) as executor:

            results = executor.map(
                process_file,
                files
            )

            for result in results:

                print(
                    result,
                    flush=True
                )


    print(
        "=================================================="
    )

    print(
        "Finished."
    )

    print(
        "=================================================="
    )
