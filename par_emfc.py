import xarray as xr
import numpy as np
import glob
import os
from concurrent.futures import ProcessPoolExecutor

# =========================
# Constants
# =========================
a = 6.371e6  # Earth radius (m)

# =========================
# Output directory (current folder)
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
        # Check vertical coordinate
        # -------------------------
        if 'lev' not in ds:
            return f"SKIP (no lev): {fil}"

        u = ds['U']
        v = ds['V']

        # -------------------------
        # 200 hPa selection (via lev)
        # -------------------------
        try:
            u_jet = u.sel(lev=200, method='nearest')
            v_jet = v.sel(lev=200, method='nearest')
        except Exception as e:
            return f"SKIP (no lev=200): {fil}"

        lat = np.deg2rad(ds['lat'])

        # -------------------------
        # Eddy decomposition
        # -------------------------
        u_zm = u_jet.mean('lon')
        v_zm = v_jet.mean('lon')

        u_prime = u_jet - u_zm
        v_prime = v_jet - v_zm

        uv = u_prime * v_prime

        # -------------------------
        # EMFC at 200 hPa
        # -------------------------
        EMFC_local = -(uv * np.cos(lat)).differentiate('lat') / (a * np.cos(lat))

        uv_clim = uv.mean('time')
        EMFC_clim = -(uv_clim * np.cos(lat)).differentiate('lat') / (a * np.cos(lat))

        # -------------------------
        # Output name
        # -------------------------
        base = os.path.basename(fil)

        if ".nc" in base:
            base = base.split(".nc")[0]

        out_name = f"{base}_200hpa_EMFC.nc"
        out_path = os.path.join(out_dir, out_name)

        # -------------------------
        # Save output
        # -------------------------
        out = xr.Dataset({
            "EMFC_local_200hpa": EMFC_local,
            "EMFC_climatology_200hpa": EMFC_clim
        })

        out.to_netcdf(out_path)

        return f"SAVED: {out_path}"

    except Exception as e:
        return f"FAILED: {fil} | {e}"


# =========================
# Parallel execution
# =========================
if __name__ == "__main__":

    # safe for HPC filesystem (avoid overloading I/O)
    nproc = min(6, os.cpu_count())

    print(f"Using {nproc} parallel workers")

    with ProcessPoolExecutor(max_workers=nproc) as executor:
        results = list(executor.map(process_file, files, chunksize=1))

    # =========================
    # Summary
    # =========================
    for r in results:
        print(r)

    print("ALL FILES COMPLETED")
