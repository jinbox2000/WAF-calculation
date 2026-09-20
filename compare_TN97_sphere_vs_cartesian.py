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

# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

files = sorted(
    glob.glob(
        #"/pscratch/sd/x/xie7/ls4p/zppy/output/daily/*/*nc"
        "/pscratch/sd/x/xie7/ls4p/zppy/output/backup_test/*nc"
    )
)
#


# ------------------------------------------------------------
# Parallel workers
# ------------------------------------------------------------

NPROC = min(6, os.cpu_count())


# ============================================================
# TEMPORAL SETTINGS
# ============================================================

# Original input:
#     3-hourly
#
# Average:
#     4 consecutive 3-hourly values
#
# Result:
#     12-hourly

INPUT_INTERVAL_HOURS = 3
WAF_INTERVAL_HOURS = 12

N_AVG = WAF_INTERVAL_HOURS // INPUT_INTERVAL_HOURS

DT_DAYS = 0.5


# ============================================================
# LOW-PASS FILTER SETTINGS
# ============================================================

CUTOFF_DAYS = 8.0

# 61-point Lanczos filter
NWT = 61


# ============================================================
# LANCZOS FILTER WEIGHTS
# ============================================================

def lanczos_weights(nwt, cutoff_days, dt_days):

    if nwt % 2 == 0:
        raise ValueError("nwt must be odd")

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

        ds = xr.open_dataset(fil)


        # ====================================================
        # CHECK VARIABLES
        # ====================================================

        required = [
            "lev",
            "U",
            "V",
            "Z3"
        ]

        for var in required:

            if var not in ds:

                ds.close()

                return (
                    f"SKIP (no {var}): {fil}"
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
        # SELECT 200 hPa
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
        # FORCE:
        #
        #     (time, lat, lon)
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
        # 3-HOURLY -> 12-HOURLY
        #
        # Every four consecutive samples:
        #
        # 00, 03, 06, 09
        #
        # 12, 15, 18, 21
        # ====================================================

        ntime = u.sizes["time"]

        n_complete = (
            ntime // N_AVG
        ) * N_AVG


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
        # FORCE:
        #
        #     (time, lat, lon)
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

        tan_lat = xr.DataArray(
            np.tan(lat_rad),
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
        # ZONAL-MEAN BASIC FLOW
        #
        # Absolute input U,V
        #
        # Background:
        #
        #     Ubar = zonal mean U
        #     Vbar = zonal mean V
        # ====================================================

        U_bg = u.mean(
            "lon"
        )

        V_bg = v.mean(
            "lon"
        )


        # ====================================================
        # WAVE / EDDY GEOPOTENTIAL HEIGHT
        #
        #     Z' = Z - ZonalMean(Z)
        # ====================================================

        z_zm = z.mean(
            "lon"
        )

        z_p = (
            z
            - z_zm
        )


        # ====================================================
        # STREAMFUNCTION
        #
        #     psi' = g Z' / f
        # ====================================================

        psi = (
            g
            * z_p
            / f_da
        )


        # ====================================================
        # DEGREE -> RADIAN
        # ====================================================

        DEG2RAD = np.pi / 180.0


        # ====================================================
        #
        # EXISTING CARTESIAN PHYSICAL DERIVATIVES
        #
        # ====================================================

        psi_lon_deg = (
            psi
            .differentiate("lon")
        )

        psi_lat_deg = (
            psi
            .differentiate("lat")
        )


        # ----------------------------------------------------
        # First derivatives
        # ----------------------------------------------------

        psi_x = (
            psi_lon_deg
            / DEG2RAD
            /
            (
                a
                * cos_lat
            )
        )

        psi_y = (
            psi_lat_deg
            / DEG2RAD
            /
            a
        )


        # ----------------------------------------------------
        # Existing second derivatives
        #
        # KEEPING YOUR ORIGINAL CODE
        # ----------------------------------------------------

        psi_xx = (
            psi_x
            .differentiate("lon")
            / DEG2RAD
            /
            (
                a
                * cos_lat
            )
        )

        psi_xy = (
            psi_x
            .differentiate("lat")
            / DEG2RAD
            /
            a
        )

        psi_yy = (
            psi_y
            .differentiate("lat")
            / DEG2RAD
            /
            a
        )


        # ====================================================
        # EXISTING TN1997 QUADRATIC TERMS
        # ====================================================

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


        # ====================================================
        # BACKGROUND WIND MAGNITUDE
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
        # EXISTING CARTESIAN TN1997 WAF
        # ====================================================

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


        # ====================================================
        #
        # NEW SPHERICAL-COORDINATE DIAGNOSTIC
        #
        # Derivatives are calculated directly in lambda/phi.
        #
        # ====================================================

        # ----------------------------------------------------
        # First derivatives with respect to radians
        # ----------------------------------------------------

        psi_lambda = (
            psi_lon_deg
            / DEG2RAD
        )

        psi_phi = (
            psi_lat_deg
            / DEG2RAD
        )


        # ----------------------------------------------------
        # Second derivatives with respect to radians
        # ----------------------------------------------------

        psi_lambdalambda_deg = (
            psi_lon_deg
            .differentiate("lon")
        )

        psi_lambdaphi_deg = (
            psi_lon_deg
            .differentiate("lat")
        )

        psi_phiphi_deg = (
            psi_lat_deg
            .differentiate("lat")
        )


        psi_lambdalambda = (
            psi_lambdalambda_deg
            /
            DEG2RAD**2
        )

        psi_lambdaphi = (
            psi_lambdaphi_deg
            /
            DEG2RAD**2
        )

        psi_phiphi = (
            psi_phiphi_deg
            /
            DEG2RAD**2
        )


        # ====================================================
        # SPHERICAL PHYSICAL SECOND DERIVATIVES
        #
        # These are the physical derivatives associated with
        # longitude-latitude coordinates on a sphere.
        #
        # psi_xx =
        #
        #     psi_ll
        #     /
        #     (a^2 cos^2 phi)
        #
        #
        # psi_xy =
        #
        #     [psi_lphi + tan(phi) psi_l]
        #     /
        #     (a^2 cos(phi))
        #
        #
        # psi_yy =
        #
        #     psi_phiphi / a^2
        #
        # ====================================================

        psi_xx_sph = (
            psi_lambdalambda
            /
            (
                a**2
                * cos_lat**2
            )
        )


        psi_xy_sph = (
            (
                psi_lambdaphi
                +
                tan_lat
                * psi_lambda
            )
            /
            (
                a**2
                * cos_lat
            )
        )


        psi_yy_sph = (
            psi_phiphi
            /
            a**2
        )


        # ====================================================
        # SPHERICAL FIRST DERIVATIVES
        #
        # Same physical first derivatives:
        #
        #     psi_x = psi_lambda/(a cos phi)
        #     psi_y = psi_phi/a
        # ====================================================

        psi_x_sph = (
            psi_lambda
            /
            (
                a
                * cos_lat
            )
        )


        psi_y_sph = (
            psi_phi
            /
            a
        )


        # ====================================================
        # SPHERICAL TN1997 QUADRATIC TERMS
        # ====================================================

        term_xx_sph = (
            psi_x_sph**2
            -
            psi
            * psi_xx_sph
        )


        term_xy_sph = (
            psi_x_sph
            * psi_y_sph
            -
            psi
            * psi_xy_sph
        )


        term_yy_sph = (
            psi_y_sph**2
            -
            psi
            * psi_yy_sph
        )


        # ====================================================
        # SPHERICAL TN1997 WAF
        # ====================================================

        WAF_x_sph = (
            (
                U_bg
                * term_xx_sph
                +
                V_bg
                * term_xy_sph
            )
            /
            (
                2.0
                * U_mag
            )
        )


        WAF_y_sph = (
            (
                U_bg
                * term_xy_sph
                +
                V_bg
                * term_yy_sph
            )
            /
            (
                2.0
                * U_mag
            )
        )


        # ====================================================
        # DIFFERENCES
        #
        # Spherical - original Cartesian
        # ====================================================

        WAF_x_difference = (
            WAF_x_sph
            -
            WAF_x
        )

        WAF_y_difference = (
            WAF_y_sph
            -
            WAF_y
        )


        # ====================================================
        # DIFFERENCE IN MIXED SECOND DERIVATIVE
        #
        # This is particularly important.
        # ====================================================

        psi_xy_difference = (
            psi_xy_sph
            -
            psi_xy
        )


        # ====================================================
        # FORCE DIMENSIONS
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

        WAF_x_sph = WAF_x_sph.transpose(
            "time",
            "lat",
            "lon"
        )

        WAF_y_sph = WAF_y_sph.transpose(
            "time",
            "lat",
            "lon"
        )

        WAF_x_difference = (
            WAF_x_difference
            .transpose(
                "time",
                "lat",
                "lon"
            )
        )

        WAF_y_difference = (
            WAF_y_difference
            .transpose(
                "time",
                "lat",
                "lon"
            )
        )


        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        print(
            "\n=========================================="
        )

        print(
            "STREAMFUNCTION DIAGNOSTICS"
        )

        print(
            f"psi             : "
            f"min={float(psi.min()):.6e}, "
            f"max={float(psi.max()):.6e}, "
            f"mean={float(psi.mean()):.6e}"
        )


        # ----------------------------------------------------
        # Physical derivatives
        # ----------------------------------------------------

        print(
            "\nPHYSICAL DERIVATIVE DIAGNOSTICS"
        )

        print(
            f"psi_x           : "
            f"min={float(psi_x.min()):.6e}, "
            f"max={float(psi_x.max()):.6e}, "
            f"mean={float(psi_x.mean()):.6e}"
        )

        print(
            f"psi_y           : "
            f"min={float(psi_y.min()):.6e}, "
            f"max={float(psi_y.max()):.6e}, "
            f"mean={float(psi_y.mean()):.6e}"
        )

        print(
            f"psi_xx          : "
            f"min={float(psi_xx.min()):.6e}, "
            f"max={float(psi_xx.max()):.6e}, "
            f"mean={float(psi_xx.mean()):.6e}"
        )

        print(
            f"psi_xy          : "
            f"min={float(psi_xy.min()):.6e}, "
            f"max={float(psi_xy.max()):.6e}, "
            f"mean={float(psi_xy.mean()):.6e}"
        )

        print(
            f"psi_yy          : "
            f"min={float(psi_yy.min()):.6e}, "
            f"max={float(psi_yy.max()):.6e}, "
            f"mean={float(psi_yy.mean()):.6e}"
        )


        # ----------------------------------------------------
        # Spherical derivatives
        # ----------------------------------------------------

        print(
            "\nSPHERICAL DERIVATIVE DIAGNOSTICS"
        )

        print(
            f"psi_x_sph       : "
            f"min={float(psi_x_sph.min()):.6e}, "
            f"max={float(psi_x_sph.max()):.6e}, "
            f"mean={float(psi_x_sph.mean()):.6e}"
        )

        print(
            f"psi_y_sph       : "
            f"min={float(psi_y_sph.min()):.6e}, "
            f"max={float(psi_y_sph.max()):.6e}, "
            f"mean={float(psi_y_sph.mean()):.6e}"
        )

        print(
            f"psi_xx_sph      : "
            f"min={float(psi_xx_sph.min()):.6e}, "
            f"max={float(psi_xx_sph.max()):.6e}, "
            f"mean={float(psi_xx_sph.mean()):.6e}"
        )

        print(
            f"psi_xy_sph      : "
            f"min={float(psi_xy_sph.min()):.6e}, "
            f"max={float(psi_xy_sph.max()):.6e}, "
            f"mean={float(psi_xy_sph.mean()):.6e}"
        )

        print(
            f"psi_yy_sph      : "
            f"min={float(psi_yy_sph.min()):.6e}, "
            f"max={float(psi_yy_sph.max()):.6e}, "
            f"mean={float(psi_yy_sph.mean()):.6e}"
        )


        # ----------------------------------------------------
        # Mixed derivative difference
        # ----------------------------------------------------

        print(
            "\nMIXED DERIVATIVE CHECK"
        )

        print(
            f"psi_xy_sph - psi_xy:"
        )

        print(
            f"min={float(psi_xy_difference.min()):.6e}, "
            f"max={float(psi_xy_difference.max()):.6e}, "
            f"mean={float(psi_xy_difference.mean()):.6e}"
        )


        # ----------------------------------------------------
        # TN1997 quadratic terms
        # ----------------------------------------------------

        print(
            "\nTN1997 QUADRATIC TERMS"
        )

        for name, da in [
            ("term_xx", term_xx),
            ("term_xy", term_xy),
            ("term_yy", term_yy),
        ]:

            print(
                f"{name:16s}: "
                f"min={float(da.min()):.6e}, "
                f"max={float(da.max()):.6e}, "
                f"mean={float(da.mean()):.6e}"
            )


        print(
            "\nTN1997 SPHERICAL QUADRATIC TERMS"
        )

        for name, da in [
            ("term_xx_sph", term_xx_sph),
            ("term_xy_sph", term_xy_sph),
            ("term_yy_sph", term_yy_sph),
        ]:

            print(
                f"{name:16s}: "
                f"min={float(da.min()):.6e}, "
                f"max={float(da.max()):.6e}, "
                f"mean={float(da.mean()):.6e}"
            )


        # ----------------------------------------------------
        # Background flow
        # ----------------------------------------------------

        print(
            "\nBACKGROUND FLOW"
        )

        print(
            f"|U_bg|          : "
            f"min={float(U_mag.min()):.6e}, "
            f"max={float(U_mag.max()):.6e}, "
            f"mean={float(U_mag.mean()):.6e}"
        )


        # ----------------------------------------------------
        # Original WAF
        # ----------------------------------------------------

        print(
            "\nTN1997 WAF — ORIGINAL CARTESIAN"
        )

        print(
            f"WAF_x           : "
            f"min={float(WAF_x.min()):.6e}, "
            f"max={float(WAF_x.max()):.6e}, "
            f"mean={float(WAF_x.mean()):.6e}"
        )

        print(
            f"WAF_y           : "
            f"min={float(WAF_y.min()):.6e}, "
            f"max={float(WAF_y.max()):.6e}, "
            f"mean={float(WAF_y.mean()):.6e}"
        )


        # ----------------------------------------------------
        # New spherical WAF
        # ----------------------------------------------------

        print(
            "\nTN1997 WAF — SPHERICAL"
        )

        print(
            f"WAF_x_sph       : "
            f"min={float(WAF_x_sph.min()):.6e}, "
            f"max={float(WAF_x_sph.max()):.6e}, "
            f"mean={float(WAF_x_sph.mean()):.6e}"
        )

        print(
            f"WAF_y_sph       : "
            f"min={float(WAF_y_sph.min()):.6e}, "
            f"max={float(WAF_y_sph.max()):.6e}, "
            f"mean={float(WAF_y_sph.mean()):.6e}"
        )


        # ----------------------------------------------------
        # Difference
        # ----------------------------------------------------

        print(
            "\nSPHERICAL - ORIGINAL DIFFERENCE"
        )

        print(
            f"dWAF_x          : "
            f"min={float(WAF_x_difference.min()):.6e}, "
            f"max={float(WAF_x_difference.max()):.6e}, "
            f"mean={float(WAF_x_difference.mean()):.6e}"
        )

        print(
            f"dWAF_y          : "
            f"min={float(WAF_y_difference.min()):.6e}, "
            f"max={float(WAF_y_difference.max()):.6e}, "
            f"mean={float(WAF_y_difference.mean()):.6e}"
        )


        print(
            "==========================================\n"
        )


        # ====================================================
        # LANCZOS FILTER
        #
        # Keep filtering the ORIGINAL WAF.
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
        # FORCE LOW-PASS DIMENSIONS
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
        # ====================================================

        WAF_x_mean = (
            WAF_x
            .mean(
                "time",
                skipna=True
            )
            .transpose(
                "lat",
                "lon"
            )
        )

        WAF_y_mean = (
            WAF_y
            .mean(
                "time",
                skipna=True
            )
            .transpose(
                "lat",
                "lon"
            )
        )


        WAF_x_LP_mean = (
            WAF_x_LP
            .mean(
                "time",
                skipna=True
            )
            .transpose(
                "lat",
                "lon"
            )
        )

        WAF_y_LP_mean = (
            WAF_y_LP
            .mean(
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
            f"{base}_200hpa_WAF_TN1997_12h_LP8d.nc"
        )


        # ====================================================
        # OUTPUT DATASET
        # ====================================================

        out = xr.Dataset({

            # ------------------------------------------------
            # Original Cartesian WAF
            # ------------------------------------------------

            "WAF_x_200hpa":
                WAF_x,

            "WAF_y_200hpa":
                WAF_y,


            # ------------------------------------------------
            # New spherical WAF
            # ------------------------------------------------

            "WAF_x_200hpa_spherical":
                WAF_x_sph,

            "WAF_y_200hpa_spherical":
                WAF_y_sph,


            # ------------------------------------------------
            # Difference
            # ------------------------------------------------

            "WAF_x_200hpa_spherical_minus_cartesian":
                WAF_x_difference,

            "WAF_y_200hpa_spherical_minus_cartesian":
                WAF_y_difference,


            # ------------------------------------------------
            # Original low-pass WAF
            # ------------------------------------------------

            "WAF_x_200hpa_LP8d":
                WAF_x_LP,

            "WAF_y_200hpa_LP8d":
                WAF_y_LP,


            # ------------------------------------------------
            # Time-mean low-pass WAF
            # ------------------------------------------------

            "WAF_x_200hpa_LP8d_mean":
                WAF_x_LP_mean,

            "WAF_y_200hpa_LP8d_mean":
                WAF_y_LP_mean,


            # ------------------------------------------------
            # Time-mean raw WAF
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


        for var in [
            "WAF_x_200hpa",
            "WAF_y_200hpa",
            "WAF_x_200hpa_spherical",
            "WAF_y_200hpa_spherical",
            "WAF_x_200hpa_spherical_minus_cartesian",
            "WAF_y_200hpa_spherical_minus_cartesian",
            "WAF_x_200hpa_LP8d",
            "WAF_y_200hpa_LP8d"
        ]:

            assert (
                out[var].dims
                ==
                expected_time_dim
            )


        for var in [
            "WAF_x_200hpa_mean",
            "WAF_y_200hpa_mean",
            "WAF_x_200hpa_LP8d_mean",
            "WAF_y_200hpa_LP8d_mean"
        ]:

            assert (
                out[var].dims
                ==
                expected_mean_dim
            )


        # ====================================================
        # GLOBAL ATTRIBUTES
        # ====================================================

        out.attrs = {

            "description":
                "200-hPa Takaya-Nakamura 1997 "
                "stationary Rossby-wave activity flux",

            "processing":
                "3-hourly -> 12-hourly averaging "
                "-> TN1997 WAF -> 8-day Lanczos low-pass",

            "waf_method":
                "Takaya-Nakamura (1997) stationary "
                "wave-activity flux",

            "streamfunction":
                "psi_prime = g * Z_prime / f",

            "perturbation_height":
                "Z_prime = Z - zonal_mean(Z)",

            "background_flow":
                "zonal-mean absolute 200-hPa U and V",

            "geopotential_height":
                "Z3 assumed to be geopotential height in meters",

            "coordinate_units":
                "latitude and longitude in degrees",

            "spherical_diagnostic":
                "Direct spherical-coordinate metric treatment "
                "including the mixed-derivative metric term",

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
        }


        # ====================================================
        # VARIABLE ATTRIBUTES
        # ====================================================

        out[
            "WAF_x_200hpa"
        ].attrs = {

            "description":
                "12-hourly 200-hPa TN1997 WAF "
                "zonal component, original Cartesian "
                "physical-derivative implementation",

            "units":
                "m2 s-2",

            "time_resolution":
                "12-hourly"
        }


        out[
            "WAF_y_200hpa"
        ].attrs = {

            "description":
                "12-hourly 200-hPa TN1997 WAF "
                "meridional component, original Cartesian "
                "physical-derivative implementation",

            "units":
                "m2 s-2",

            "time_resolution":
                "12-hourly"
        }


        out[
            "WAF_x_200hpa_spherical"
        ].attrs = {

            "description":
                "12-hourly 200-hPa TN1997 WAF "
                "zonal component using direct spherical "
                "longitude-latitude derivatives",

            "units":
                "m2 s-2",

            "time_resolution":
                "12-hourly"
        }


        out[
            "WAF_y_200hpa_spherical"
        ].attrs = {

            "description":
                "12-hourly 200-hPa TN1997 WAF "
                "meridional component using direct spherical "
                "longitude-latitude derivatives",

            "units":
                "m2 s-2",

            "time_resolution":
                "12-hourly"
        }


        out[
            "WAF_x_200hpa_spherical_minus_cartesian"
        ].attrs = {

            "description":
                "Spherical-coordinate TN1997 WAF minus "
                "original Cartesian TN1997 WAF",

            "units":
                "m2 s-2"
        }


        out[
            "WAF_y_200hpa_spherical_minus_cartesian"
        ].attrs = {

            "description":
                "Spherical-coordinate TN1997 WAF minus "
                "original Cartesian TN1997 WAF",

            "units":
                "m2 s-2"
        }


        out[
            "WAF_x_200hpa_LP8d"
        ].attrs = {

            "description":
                "8-day low-pass filtered "
                "200-hPa TN1997 WAF zonal component",

            "units":
                "m2 s-2",

            "filter":
                "Lanczos",

            "cutoff_days":
                CUTOFF_DAYS,

            "time_resolution":
                "12-hourly"
        }


        out[
            "WAF_y_200hpa_LP8d"
        ].attrs = {

            "description":
                "8-day low-pass filtered "
                "200-hPa TN1997 WAF meridional component",

            "units":
                "m2 s-2",

            "filter":
                "Lanczos",

            "cutoff_days":
                CUTOFF_DAYS,

            "time_resolution":
                "12-hourly"
        }


        out[
            "WAF_x_200hpa_mean"
        ].attrs = {

            "description":
                "Time-mean 200-hPa TN1997 "
                "WAF zonal component",

            "units":
                "m2 s-2"
        }


        out[
            "WAF_y_200hpa_mean"
        ].attrs = {

            "description":
                "Time-mean 200-hPa TN1997 "
                "WAF meridional component",

            "units":
                "m2 s-2"
        }


        out[
            "WAF_x_200hpa_LP8d_mean"
        ].attrs = {

            "description":
                "Time-mean 8-day low-pass "
                "200-hPa TN1997 WAF zonal component",

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
                "Time-mean 8-day low-pass "
                "200-hPa TN1997 WAF meridional component",

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
            |
            {
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
        "200-hPa Takaya-Nakamura 1997 WAF"
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
        "Averaging:   3-hourly -> 12-hourly"
    )

    print(
        "WAF:         TN1997"
    )

    print(
        "Level:       200 hPa"
    )

    print(
        "Perturbation: Z - zonal mean(Z)"
    )

    print(
        "Background:  zonal-mean U,V"
    )

    print(
        "Low-pass:    8 days"
    )

    print(
        f"Lanczos NWT: {NWT}"
    )

    print(
        "WAF output:  (time, lat, lon)"
    )

    print(
        "Mean output: (lat, lon)"
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
