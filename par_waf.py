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

        # =========================
        # EDDY DECOMPOSITION
        # =========================
        u_zm = u_jet.mean(lon_name)
        v_zm = v_jet.mean(lon_name)

        u_p = u_jet - u_zm
        v_p = v_jet - v_zm

        uv = u_p * v_p

        # =========================
        # EMFC
        # =========================
        EMFC_local = -(uv * np.cos(lat_da)).differentiate('lat') / (a * np.cos(lat_da))
        EMFC_clim = EMFC_local.mean('time')

        # =========================
        # ================= WAF =================
        # =========================

        f = 2 * Omega * np.sin(lat)
        f_da = xr.DataArray(f, coords={'lat': ds['lat']}, dims=['lat'])

        # avoid singularity
        f_da = xr.where(np.abs(f_da) < 1e-5, np.nan, f_da)

        # -------------------------
        # Streamfunction proxy
        # ψ' ≈ u'/f
        # -------------------------
        psi = u_p / f_da

        # -------------------------
        # gradients (must keep lon)
        # -------------------------
        dpsi_dx = psi.differentiate(lon_name)
        dpsi_dy = psi.differentiate('lat')

        # =========================
        # WAF (instantaneous)
        # =========================
        WAF_x = -dpsi_dx * dpsi_dy
        WAF_y = 0.5 * (dpsi_dx**2 - dpsi_dy**2)

        # =========================
        # WAF climatology (CORRECT)
        # time mean of WAF, NOT zonal mean ψ
        # =========================
        WAF_x_clim = WAF_x.mean('time')
        WAF_y_clim = WAF_y.mean('time')

        # =========================
        # OUTPUT
        # =========================
        base = os.path.basename(fil).replace(".nc", "")

        out_path = os.path.join(out_dir, f"{base}_200hpa_EMFC_WAF.nc")

        out = xr.Dataset({
            # EMFC
            "EMFC_local_200hpa": EMFC_local,
            "EMFC_clim_200hpa": EMFC_clim,

            # WAF instantaneous
            "WAF_x_200hpa": WAF_x,
            "WAF_y_200hpa": WAF_y,

            # WAF climatology (time mean)
            "WAF_x_clim_200hpa": WAF_x_clim,
            "WAF_y_clim_200hpa": WAF_y_clim
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
