import xarray as xr
import numpy as np
import glob
import os

# =========================
# Constants
# =========================
a = 6.371e6  # Earth radius (m)

# =========================
# Output directory (CURRENT FOLDER)
# =========================
out_dir = os.getcwd()

# =========================
# File list
# =========================
files = sorted(glob.glob(
    "/pscratch/sd/x/xie7/ls4p/zppy/output/daily/*/*nc"
))

# =========================
# Loop over files
# =========================
for fil in files:
    print(f"Processing: {fil}")

    ds = xr.open_dataset(fil)

    # -------------------------
    # Read variables
    # -------------------------
    u = ds['U']
    v = ds['V']
    lat = np.deg2rad(ds['lat'])

    # -------------------------
    # Handle pressure levels
    # -------------------------
    if 'plev' in ds:
        p = ds['plev']

        # hPa → Pa if needed
        if p.max() < 2000:
            p = p * 100.0

        ds = ds.assign_coords(plev=p)

        # standard pressure levels
        target_plevs = np.array([1000, 850, 700, 500, 300, 200, 100]) * 100
        ds = ds.interp(plev=target_plevs)

        u = ds['U']
        v = ds['V']

    # -------------------------
    # Eddy decomposition (zonal mean at each time)
    # -------------------------
    u_zm = u.mean('lon')
    v_zm = v.mean('lon')

    u_prime = u - u_zm
    v_prime = v - v_zm

    uv = u_prime * v_prime

    # -------------------------
    # EMFC (local, longitude-resolved)
    # -------------------------
    EMFC_local = -(uv * np.cos(lat)).differentiate('lat') / (a * np.cos(lat))

    # -------------------------
    # EMFC climatology (within file)
    # -------------------------
    uv_clim = uv.mean('time')

    EMFC_clim = -(uv_clim * np.cos(lat)).differentiate('lat') / (a * np.cos(lat))

    # -------------------------
    # Output filename (CURRENT DIRECTORY)
    # -------------------------
    base = os.path.basename(fil)

    # safe rename: strip everything after first ".nc"
    if ".nc" in base:
        base = base.split(".nc")[0] + ".nc"

    out_name = base.replace(".nc", "_processed.nc")
    out_name = os.path.join(out_dir, out_name)

    # -------------------------
    # Save output
    # -------------------------
    out = xr.Dataset({
        "EMFC_local": EMFC_local,
        "EMFC_climatology": EMFC_clim
    })

    out.to_netcdf(out_name)

    print(f"Saved: {out_name}")

print("All files processed.")
