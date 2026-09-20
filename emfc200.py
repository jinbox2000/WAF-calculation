import xarray as xr
import numpy as np
import glob
import os

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
# Loop over files
# =========================
for fil in files:
    print(f"Processing: {fil}")

    ds = xr.open_dataset(fil)

    # -------------------------
    # Check required variables
    # -------------------------
    if 'lev' not in ds:
        print("Skipping (no lev):", fil)
        continue

    u = ds['U']
    v = ds['V']

    # -------------------------
    # 200 hPa selection (LEV must be pressure-based!)
    # -------------------------
    try:
        u_jet = u.sel(lev=200, method='nearest')
        v_jet = v.sel(lev=200, method='nearest')
    except Exception as e:
        print("Skipping (cannot select lev=200):", fil)
        continue

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
    # EMFC (200 hPa)
    # -------------------------
    EMFC_local = -(uv * np.cos(lat)).differentiate('lat') / (a * np.cos(lat))

    uv_clim = uv.mean('time')
    EMFC_clim = -(uv_clim * np.cos(lat)).differentiate('lat') / (a * np.cos(lat))

    # -------------------------
    # Output filename (200 hPa tagged)
    # -------------------------
    base = os.path.basename(fil)

    if ".nc" in base:
        base = base.split(".nc")[0]

    out_name = f"{base}_200hpa_EMFC.nc"
    out_name = os.path.join(out_dir, out_name)

    # -------------------------
    # Save output
    # -------------------------
    out = xr.Dataset({
        "EMFC_local_200hpa": EMFC_local,
        "EMFC_climatology_200hpa": EMFC_clim
    })

    out.to_netcdf(out_name)

    print(f"Saved: {out_name}")

print("All files processed.")
