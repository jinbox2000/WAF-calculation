import xarray as xr
import numpy as np
import glob
import os
from concurrent.futures import ProcessPoolExecutor

# =========================
# CONSTANTS
# =========================
a = 6.371e6
Omega = 7.292e-5

# =========================
# OUTPUT
# =========================
out_dir = os.getcwd()

# =========================
# FILE LIST
# =========================
files = sorted(glob.glob(
    "/pscratch/sd/x/xie7/ls4p/zppy/output/daily/*/*nc"
))

# =========================
# WORKER
# =========================
def process_file(fil):
    try:
        print(f"Processing: {fil}")

        ds = xr.open_dataset(fil)

        if 'lev' not in ds:
            return f"SKIP (no lev): {fil}"

        u = ds['U']
        v = ds['V']

        # =========================
        # LEVEL SELECTION
        # =========================
        try:
            u_jet = u.sel(lev=200, method='nearest')
            v_jet = v.sel(lev=200, method='nearest')
        except:
            return f"SKIP (no lev=200): {fil}"

        lat = np.deg2rad(ds['lat'])
        lat_da = xr.DataArray(lat, coords={'lat': ds['lat']}, dims=['lat'])

        lon_name = 'lon' if 'lon' in ds.coords else None
        if lon_name is None:
            return f"SKIP (no lon): {fil}"

        # ======================================================
        # EDDY DECOMPOSITION (EMFC)
        # ======================================================
        u_zm = u_jet.mean(lon_name)
        v_zm = v_jet.mean(lon_name)

        u_p = u_jet - u_zm
        v_p = v_jet - v_zm

        uv = u_p * v_p

        EMFC_local = -(uv * np.cos(lat_da)).differentiate('lat') / (a * np.cos(lat_da))
        EMFC_clim = EMFC_local.mean('time')

        # ======================================================
        # WAF (wave activity flux)
        # ======================================================
        f = 2 * Omega * np.sin(lat)
        f_da = xr.DataArray(f, coords={'lat': ds['lat']}, dims=['lat'])
        f_da = xr.where(np.abs(f_da) < 1e-5, np.nan, f_da)

        psi = u_p / f_da

        dpsi_dx = psi.differentiate(lon_name)
        dpsi_dy = psi.differentiate('lat')

        WAF_x = -dpsi_dx * dpsi_dy
        WAF_y = 0.5 * (dpsi_dx**2 - dpsi_dy**2)

        WAF_x_clim = WAF_x.mean('time')
        WAF_y_clim = WAF_y.mean('time')

        # ======================================================
        # ================= 3D REFRACTIVE INDEX n² =============
        # ======================================================

        # KEEP FULL STRUCTURE: (time, lat, lon)
        U = u_jet

        beta = 2 * Omega * np.cos(lat_da) / a

        # meridional curvature
        dU_dy = U.differentiate('lat')
        d2U_dy2 = dU_dy.differentiate('lat')

        # raw 3D refractive index
        n2 = (beta - d2U_dy2) / (U + 1e-6)

        # mask weak wind regions (avoid blow-up)
        n2 = xr.where(np.abs(U) < 1.0, np.nan, n2)

        # OPTIONAL: light temporal smoothing (recommended for interpretation)
        n2_smooth = n2.rolling(time=3, center=True).mean()

        # ======================================================
        # OUTPUT
        # ======================================================
        base = os.path.basename(fil).replace(".nc", "")
        out_path = os.path.join(out_dir, f"{base}_200hpa_EMFC_WAF_n2_3D.nc")

        out = xr.Dataset({
            # EMFC
            "EMFC_local_200hpa": EMFC_local,
            "EMFC_clim_200hpa": EMFC_clim,

            # WAF
            "WAF_x_200hpa": WAF_x,
            "WAF_y_200hpa": WAF_y,
            "WAF_x_clim_200hpa": WAF_x_clim,
            "WAF_y_clim_200hpa": WAF_y_clim,

            # 3D refractive index
            "n2_3D_200hpa": n2,
            "n2_3D_smooth_200hpa": n2_smooth
        })

        out.to_netcdf(out_path)

        return f"SAVED: {out_path}"

    except Exception as e:
        return f"FAILED: {fil} | {e}"


# =========================
# PARALLEL EXECUTION
# =========================
if __name__ == "__main__":

    nproc = min(6, os.cpu_count())
    print(f"Using {nproc} workers")

    with ProcessPoolExecutor(max_workers=nproc) as executor:
        results = list(executor.map(process_file, files, chunksize=1))

    for r in results:
        print(r)

    print("ALL FILES COMPLETED")
