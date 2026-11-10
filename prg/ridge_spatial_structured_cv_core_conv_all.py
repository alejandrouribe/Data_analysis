from unittest import result
from matplotlib.pylab import i0
import xarray as xr
import numpy as np
import os
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import sys
# ============================================================
# Prepare spatial predictor windows
# ============================================================
def prepare_rolling(ds, predictors, lat_win, lon_win, local_vars=None):
    rolled = []
    #
    # Apply periodic padding along longitude
    pad = lon_win // 2
    lon = ds.lon
    ds_left  = ds.isel(lon=slice(-pad, None))
    ds_right = ds.isel(lon=slice(0, pad))
    ds_left  = ds_left.assign_coords(lon=lon[-pad:] - 360)
    ds_right = ds_right.assign_coords(lon=lon[:pad] + 360)
    ds_ext = xr.concat([ds_left, ds, ds_right], dim="lon")
    # 
    for var in predictors:
        # Keep local predictors as time series without spatial windows
        if var in local_vars:
            da = ds[var]
            da = da.chunk({"time": -1})
            rolled.append(da)
            continue
        #
        # Construct a centered spatial window around each grid point
        lat_offsets = np.arange(-(lat_win // 2), lat_win // 2 + 1)
        lon_offsets = np.arange(-(lon_win // 2), lon_win // 2 + 1)
        da = ds_ext[var].rolling(
            lat=lat_win, lon=lon_win, center=True
        ).construct(lat="lat_win", lon="lon_win")
        #
        # Restore the original longitude domain
        da = da.sel(lon=lon)
        #
        # Assign relative window coordinates
        lat_window = ds.lat.values[:, None] + lat_offsets[None, :]
        lon_window = ds.lon.values[:, None] + lon_offsets[None, :]
        da = da.assign_coords(
            lat_win=(("lat", "lat_win"), lat_window),
            lon_win=(("lon", "lon_win"), lon_window),
        )
        #
        # Keep the full time series in each Dask chunk
        da = da.chunk({"time": -1})
        rolled.append(da)
    return rolled

# ============================================================
# Local Ridge regression with temporal cross-validation
# ============================================================
def ridge_local_structured_cv(y, *X_vars, alphas):
    # Build the design matrix from spatial-window and local predictors
    X_list = []
    for X in X_vars:
        if X.ndim == 3:
            # Flatten spatial windows while preserving the time dimension
            t = X.shape[0]
            X_flat = X.reshape(t, -1)
            X_list.append(X_flat)
        elif X.ndim == 1:
            # Add a feature dimension to local time series
            X_list.append(X[:, None])
        else:         
            raise ValueError(f"Unexpected shape: {X.shape}")
    # Combine all predictors into a single design matrix
    X = np.concatenate(X_list, axis=1)
    t = X.shape[0]
    # Identify time steps with complete predictor and target data
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    coef_size = X.shape[1]
    N_OUT = coef_size + 4 + 2 * t
    coef_flat = np.full(coef_size, np.nan)
    stats = np.full(4, np.nan)
    y_obs_full = np.full(t, np.nan)
    y_pred_full = np.full(t, np.nan)
    #
    if valid.sum() < 20:
        return np.concatenate([coef_flat, stats, y_obs_full, y_pred_full])
    #
    y_clean = y[valid]
    X_clean = X[valid]
    # Standardize predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    # Select the Ridge regularization parameter using temporal CV
    tscv = TimeSeriesSplit(n_splits=5)
    model = RidgeCV(alphas=alphas, cv=tscv)
    model.fit(X_scaled, y_clean)
    #
    coef_flat = model.coef_
    y_pred = model.predict(X_scaled)
    # Calculate model diagnostics
    stats = np.array([
        model.alpha_,
        r2_score(y_clean, y_pred),
        mean_squared_error(y_clean, y_pred),
        np.corrcoef(y_clean, y_pred)[0, 1],
    ])
    # Restore observations and predictions to the original time axis
    y_obs_full[valid] = y_clean
    y_pred_full[valid] = y_pred
    return np.concatenate([
        coef_flat,
        stats,
        y_obs_full,
        y_pred_full
    ])

# ============================================================
# Apply spatially structured Ridge regression across the grid
# ============================================================
def ridge_spatial_structured_cv(ds, y, predictors, lat_win, lon_win, alphas, local_vars):
    if local_vars is None:
        local_vars = []
    rolled = prepare_rolling(ds, predictors, lat_win, lon_win, local_vars=local_vars)
    spatial_vars = [v for v in predictors if v not in local_vars]
    n_spatial = len(spatial_vars)
    n_local = len(local_vars)
    coef_size = n_spatial * lat_win * lon_win + n_local
    total_size = coef_size + 4 + 2 * y.sizes["time"]
    #
    result = xr.apply_ufunc(
        ridge_local_structured_cv,
        y,
        *rolled,
        input_core_dims=[
            ["time"],  # y
        ] + [
            ["time", "lat_win", "lon_win"] if p not in local_vars
            else ["time"]
            for p in predictors
        ],
        output_core_dims=[["coef"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        kwargs={"alphas": alphas},
        dask_gufunc_kwargs={
            "output_sizes": {"coef": total_size}
        },
    )
    i0 = n_spatial * lat_win * lon_win
    i1 = i0 + n_local
    i2 = i1 + 4
    i3 = i2 + y.sizes["time"]

    coef_spatial_flat = result.isel(coef=slice(0, i0))
    if n_local > 0:
        coef_local = result.isel(coef=slice(i0, i1)).isel(coef=0)
    else:
        coef_local = None
        
    alpha   = result.isel(coef=i1)
    r2      = result.isel(coef=i1 + 1)
    mse     = result.isel(coef=i1 + 2)
    pearson = result.isel(coef=i1 + 3)
    #
    time_coord = y.coords["time"]
    y_obs = (
        result.isel(coef=slice(i2, i3))
        .rename(coef="time")
        .assign_coords(time=time_coord)
    )

    y_pred = (
        result.isel(coef=slice(i3, i3 + y.sizes["time"]))
        .rename(coef="time")
        .assign_coords(time=time_coord)
    )    
    #  --- Reshape coeficientes ---
    coef_spatial = coef_spatial_flat.data.reshape(
        coef_spatial_flat.shape[:-1] + (n_spatial, lat_win, lon_win)
    )

    coef_spatial = xr.DataArray(
        coef_spatial,
        dims=coef_spatial_flat.dims[:-1] + ("predictor", "lat_win", "lon_win"),
        coords={
            **{k: coef_spatial_flat.coords[k] for k in coef_spatial_flat.dims if k != "coef"},
            "predictor": spatial_vars,
            "lat_win": rolled[0].coords["lat_win"],
            "lon_win": rolled[0].coords["lon_win"],
        },
    )
    return coef_spatial, coef_local, alpha, r2, mse, pearson, y_obs, y_pred
# ------------------------------------------------- 
#  Run spatially structured ridge regression with cross-validation
# -------------------------------------------------
def main():
    # ------------------------------------------------- 
    #  Load datasets and prepare masks
    # -------------------------------------------------
    res="5deg"
    base_path = "/nobackup/proj/disk/bolinc/personal/alur/AC3S/data"
    file_end_name = f"_monthly_anom_deseasonal_detrended_{res}.nc"
    paths = {
        "CRE": f"{base_path}/CERES_FluxByCldTyp/monthly/processed/CERES_FBCT_DEEP_CRE{file_end_name}",   
        "DU300": f"{base_path}/ERA5/monthly/processed/DU300{file_end_name}",
        "SUT": f"{base_path}/ERA5/monthly/processed/SUT{file_end_name}",
        "UTRH": f"{base_path}/ERA5/monthly/processed/UTRH{file_end_name}",
        "T2m": f"{base_path}/ERA5/monthly/processed/t2m{file_end_name}",
        "W500": f"{base_path}/ERA5/monthly/processed/W500{file_end_name}",
        "RH700": f"{base_path}/ERA5/monthly/processed/RH_700{file_end_name}",    
        "suaod550": f"{base_path}/CAMS/monthly/processed/suaod550{file_end_name}",
        "SO4CMASS": f"{base_path}/MERRA/monthly/processed/SO4CMASS{file_end_name}",
        "SO4_MASS_CON": f"{base_path}/MERRA/monthly/processed/SO4_MASS_CON{file_end_name}"
    }
    datasets = []
    for path in paths.values():
        ds = xr.open_dataset(path)#.sel(lat=slice(-35, 35),lon=slice(270,290))
        ds = ds.chunk({"time": -1, "lat": 4, "lon": 4})  
        # Sort by time
        ds = ds.sortby("time")
            # --- Fix pressure level properly ---
        if "pressure_level" in ds.sizes:
            if "W500" in path:
                ds = ds.sel(pressure_level=500, method="nearest")
            elif "RH_700" in path:
                ds = ds.sel(pressure_level=700, method="nearest")
            ds = ds.drop_vars("pressure_level")   
        # Remove duplicates if they exist
        _, index = np.unique(ds["time"], return_index=True)
        ds = ds.isel(time=index)
        ds["time"] = ds.indexes["time"].to_period("M").to_timestamp()   
        datasets.append(ds)
    ds_all = xr.merge(datasets)
    # ---- Masks ----
    # 1. Tropics
    tropics_mask = np.abs(ds_all.lat) <= 25
    tropics_2d = tropics_mask.broadcast_like(ds_all["cre_net_deep"].isel(time=0))
    # 2. Valid data fraction
    nt = ds_all.sizes["time"]
    valid_fraction = ds_all["cre_net_deep"].notnull().sum("time") / nt
    data_mask = valid_fraction >= 0.1
    # 3. Ocean mask
    lsm = xr.open_dataset("/nobackup/proj/disk/bolinc/personal/alur/AC3S/data/ERA5/monthly/ERA5_land_sea_mask_5deg.nc").isel(time=0)["lsm"]
    ocean_mask = xr.where(lsm > 0, 0, 1)
    # 4. Convective mask
    data_w =xr.open_dataset("/nobackup/proj/disk/bolinc/personal/alur/AC3S/data/ERA5/monthly/W_monthly_ERA5_5deg_corect_dim.nc")
    conv_mask = data_w.W500 < 0
    # 5. Combine masks
    final_mask = tropics_2d & data_mask & ocean_mask & conv_mask
    # Apply mask
    #ds_masked = ds_all.where(final_mask)
    # ---- Prepare predictors ----
    predictors_suaod550 = [v for v in ds_all.data_vars if v not in ["cre_net_deep", "cre_lw_deep", "cre_sw_deep", "cf_deep", "SO4_MASS_CON", "SO4CMASS"]]
    predictors_SO4_MASS_CON = [v for v in ds_all.data_vars if v not in ["cre_net_deep", "cre_lw_deep", "cre_sw_deep", "cf_deep", "suaod550", "SO4CMASS"]]
    predictors_SO4CMASS = [v for v in ds_all.data_vars if v not in ["cre_net_deep", "cre_lw_deep", "cre_sw_deep", "cf_deep", "suaod550", "SO4_MASS_CON"]]
    #
    #
    ds_all = ds_all.chunk({"time": -1, "lat": 4, "lon": 4})
    #
    targets = {
        "net": ds_all["cre_net_deep"],
        "sw": ds_all["cre_sw_deep"],
        "lw": ds_all["cre_lw_deep"],
    }
    #
    lat_win = 27#int(sys.argv[1])
    lon_win = 27#int(sys.argv[2])
    #alphas = [0.1, 1]
    alphas = np.logspace(-4, 4, 20)
    #------------------------------------------------
    # Run ridge regression with SO4_MASS_CON as predictor
    #------------------------------------------------
    results = []

    for name, y_var in targets.items():
 
        print(f"Running ridge for {name}")

        y = y_var.where(final_mask).chunk({"time": -1})

        beta = ridge_spatial_structured_cv(
            ds=ds_all,
            y=y,
            predictors=predictors_SO4_MASS_CON,
            lat_win=lat_win,
            lon_win=lon_win,
            alphas=alphas,
            local_vars=None
        )
        coef_spatial, coef_local, alpha, r2, mse, pearson, y_obs, y_pred = [
            x.compute() if x is not None else None for x in beta
        ]
        ds_out = xr.Dataset({
            "coef_spatial": coef_spatial,
            "coef_local": coef_local,
            "alpha": alpha,
            "r2": r2,
            "mse": mse,
            "pearson": pearson,
            "y_obs": y_obs,
            "y_pred": y_pred
        })
        # add dimension identifying component
        ds_out = ds_out.expand_dims({"component": [name]})
        results.append(ds_out)
    ds_all_out = xr.concat(results, dim="component")
    ds_all_out.to_netcdf(f"/nobackup/proj/disk/bolinc/personal/alur/AC3S/prod_data/Ridge_{lat_win}x{lon_win}_results_{res}_SO4_MASS_CON_cores_ALL_components_conv.nc")
if __name__ == "__main__":
    main()
