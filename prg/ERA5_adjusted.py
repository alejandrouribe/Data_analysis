import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pmdarima.arima import auto_arima

# ---------------------------------------------------------------------
# 1. Data preparation: global-mean TOA radiative fluxes
# ---------------------------------------------------------------------
input_dir = "/home/alejandro/emergent_constrains/data/"
output_dir = "/home/alejandro/emergent_constrains/data_prod/"

# Load CERES-ERBE observations and ERA5 reanalysis.
ceres = xr.open_dataset(
    input_dir + 'CERES_ERBE/DEEP-C_TOA_v5_1985-2020.nc'
).sel(time=slice("1985-01", "2019-12"))

era5 = xr.open_dataset(
    input_dir + "ERA5/ERA5_1959_2021.nc"
).sel(time=slice("1971-01", "2019-12"))

# Calculate latitude-weighted global-mean radiative fluxes.
weights_ceres = np.cos(np.deg2rad(ceres.lat))
weights_era5 = np.cos(np.deg2rad(era5.latitude))

lw_ceres = ceres.OLR.weighted(weights_ceres).mean(dim=("lat", "lon"))
sw_ceres = ceres.ASR.weighted(weights_ceres).mean(dim=("lat", "lon"))
net_ceres = lw_ceres + sw_ceres

lw_era5 = (
    era5.ttr.weighted(weights_era5).mean(dim=("latitude", "longitude"))
    / (24 * 60 * 60)
)
sw_era5 = (
    era5.tsr.weighted(weights_era5).mean(dim=("latitude", "longitude"))
    / (24 * 60 * 60)
)
net_era5 = lw_era5 + sw_era5

observations = (lw_ceres, sw_ceres, net_ceres)
reanalysis = (lw_era5, sw_era5, net_era5)
variable_names = ("LW", "SW", "NET")

# Create a monthly timeline for the full reconstruction period.
full_time = pd.date_range(
    "1971-01-01", "2019-12-01", freq="MS"
) + pd.Timedelta(days=14)

# Initialise output series with missing values.
extended_series = [
    xr.DataArray(
        np.full(len(full_time), np.nan),
        coords={"time": full_time},
        dims=("time",),
        name=f"{name}_extended",
    )
    for name in variable_names
]

# ---------------------------------------------------------------------
# 2. Bias modelling: estimate the historical ERA5 correction
# ---------------------------------------------------------------------
for obs, era, extended, name in zip(
    observations, reanalysis, extended_series, variable_names
):
    # Align ERA5 and CERES-ERBE over their common observational period.
    era_overlap = era.sel(time=slice("1985-01", "2019-12"))
    obs_overlap = obs.sel(time=slice("1985-01", "2019-12"))
    era_overlap = era_overlap.assign_coords(time=obs_overlap.time)

    # Calculate the monthly difference between ERA5 and observations.
    bias = (era_overlap - obs_overlap).sortby("time")

    # Reverse the series so the model extrapolates towards earlier dates.
    bias_reversed = bias.isel(time=slice(None, None, -1))

    # Fit a seasonal ARIMA model to the observed bias time series.
    model = auto_arima(
        bias_reversed.values,
        seasonal=True,
        m=12,
        suppress_warnings=True,
        error_action="ignore",
    )

    # Forecast the bias for the 14 years before satellite observations.
    n_months = 14 * 12
    bias_forecast, bias_conf_int = model.predict(
        n_periods=n_months,
        return_conf_int=True,
    )

    # Reverse the forecast to match chronological order.
    bias_historical = bias_forecast[::-1]

    historical_time = pd.date_range(
        "1971-01-01", "1984-12-01", freq="MS"
    ) + pd.Timedelta(days=14)

    era_historical = era.sel(time=slice("1971-01", "1984-12"))
    era_historical = era_historical.assign_coords(time=historical_time)

    # Apply the estimated bias correction to historical ERA5 data.
    extended.loc[{"time": historical_time}] = (
        era_historical.values - bias_historical
    )

    # Retain satellite observations throughout the overlap period.
    extended.loc[{"time": obs_overlap.time.values}] = obs_overlap.values

# ---------------------------------------------------------------------
# 3. Visualisation and output
# ---------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

for obs, era, extended, name, ax in zip(
    observations, reanalysis, extended_series, variable_names, axes
):
    ax.plot(
        extended.time, extended,
        color="tab:blue", label="Reconstructed"
    )
    ax.plot(
        obs.time, obs,
        color="tab:red", label="CERES-ERBE"
    )
    ax.plot(
        era.time, era,
        color="black", alpha=0.7, label="ERA5"
    )

    ax.set_ylabel(f"{name} flux\n(W m$^{{-2}}$)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(frameon=False, ncol=3, loc="best")
axes[-1].set_xlim(
    pd.Timestamp("1971-01-01"),
    pd.Timestamp("2020-01-01"),
)
fig.tight_layout()
plt.show()

# Add metadata and save the reconstructed time series.
ds_out = xr.Dataset()

descriptions = {
    "LW_extended": "Time-extended longwave radiative flux at TOA",
    "SW_extended": "Time-extended shortwave radiative flux at TOA",
    "NET_extended": "Time-extended net radiative flux at TOA",
}

for extended, name in zip(extended_series, descriptions):
    extended.name = name
    extended.attrs["long_name"] = descriptions[name]
    extended.attrs["units"] = "W m-2"
    ds_out[name] = extended

ds_out.to_netcdf(output_dir + "TOA_fluxes_extended.nc", mode="w",)