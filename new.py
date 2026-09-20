import xarray as xr
import numpy as np
import glob
import os
from concurrent.futures import ProcessPoolExecutor

# =========================
# Constants
# =========================
a = 6.371e6  # Earth radius (m)
Omega = 7.292e-5

# =========================
# Output directory
# =========================
out_dir = os.getcwd()

# =========================
# File list
# =========================
files = sorted(glob.glob(
    "/pscratch/sd/x/xie7/ls4p/zppy/output/daily/*/*nc"
))

# =========================
# Worker function
# =========================
def process_file(fil):
    try:
        print(f"Processing: {fil}")

        ds = xr.open_dataset(fil)

        # -------------------------
        # Check variables
        # -------------------------
        if 'lev' not in ds:
            return f"SKIP (no lev): {fil}"

        u = ds['U']
        v = ds['V']

        # -------------------------
        # Select 200 hPa
        # -------------------------
        try:
            u_jet = u.sel(lev=200, method='nearest')
            v_jet = v.sel(lev=200, method='nearest')
        except:
            return f"SKIP (no lev=200): {fil}"

        lat = np.deg2rad(ds['lat'])
        lon = ds['lon']

        lat_da = xr.DataArray(lat, coords={'lat': ds['lat']}, dims=['lat'])

        # =========================================================
        # Eddy decomposition
        # =========================================================
        u_zm = u_jet.mean('lon')
        v_zm = v_jet.mean('lon')

        u_prime = u_jet - u_zm
        v_prime = v_jet - v_zm

        uv = u_prime * v_prime

        # =========================================================
        # EMFC (your original method)
        # =========================================================
        EMFC_local = -(uv * np.cos(lat_da)).differentiate('lat') / (a * np.cos(lat_da))

        uv_clim = uv.mean('time')
        EMFC_clim = -(uv_clim * np.cos(lat_da)).differentiate('lat') / (a * np.cos(lat_da))

        # =========================================================
        # ================= WAVE ACTIVITY FLUX ====================
        # =========================================================

        # --- Coriolis parameter
        f = 2 * Omega * np.sin(lat)
        f_da = xr.DataArray(f, coords={'lat': ds['lat']}, dims=['lat'])

        # --- streamfunction proxy (QG assumption)
        psi = u_prime / f_da
        psi_zm = u_zm / f_da

        # -------------------------
        # gradients
        # -------------------------
        dpsi_dlon = psi.differentiate('lon')
        dpsi_dlat = psi.differentiate('lat')

        dpsi_clim_dlon = psi_zm.differentiate('lon')
        dpsi_clim_dlat = psi_zm.differentiate('lat')

        # -------------------------
        # WAF components (Plumb-style proxy)
        # -------------------------
        WAF_x = -dpsi_dlon * dpsi_dlat
        WAF_y = 0.5 * (dpsi_dlon**2 - dpsi_dlat**2)

        WAF_x_clim = -dpsi_clim_dlon * dpsi_clim_dlat
        WAF_y_clim = 0.5 * (dpsi_clim_dlon**2 - dpsi_clim_dlat**2)

        # =========================================================
        # OUTPUT
        # =========================================================
        base = os.path.basename(fil).replace(".nc", "")

        out_path = os.path.join(out_dir, f"{base}_200hpa_EMFC_WAF.nc")

        out = xr.Dataset({
            # EMFC
            "EMFC_local_200hpa": EMFC_local,
            "EMFC_climatology_200hpa": EMFC_clim,

            # WAF
            "WAF_x_200hpa": WAF_x,
            "WAF_y_200hpa": WAF_y,
            "WAF_x_climatology_200hpa": WAF_x_clim,
            "WAF_y_climatology_200hpa": WAF_y_clim
        })

        out.to_netcdf(out_path)

        return f"SAVED: {out_path}"

    except Exception as e:
        return f"FAILED: {fil} | {e}"


# =========================
# Parallel execution
# =========================
if __name__ == "__main__":

    nproc = min(6, os.cpu_count())
    print(f"Using {nproc} workers")

    with ProcessPoolExecutor(max_workers=nproc) as executor:
        results = list(executor.map(process_file, files, chunksize=1))

    for r in results:
        print(r)

    print("ALL FILES COMPLETED")
