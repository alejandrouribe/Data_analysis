"""
This script calculates tropical extreme precipitation as the tail of the daily precipitation distribution (99th to 99.99th percentiles)
and the Clausius-clapeyron scalling (CC, lower tropospheric saturation specific humidity change with temperature).
@author: Alejandro UC
"""
#-----------------------------------------------------------------------------
## Libraries
#-----------------------------------------------------------------------------
import numpy as np
import xarray as xr
#-----------------------------------------------------------------------------
## Input, output paths
#-----------------------------------------------------------------------------
input='/DATA/'
input='/home/alejandro/scratch/DATA_2_3/'
exps=['r2b4_CV','r2b4_NOCV','r2b5_CV','r2b5_NOCV','r2b6_CV','r2b6_NOCV','r2b8_NOCV']
#-----------------------------------------------------------------------------
## Looping along simulatons
#-----------------------------------------------------------------------------
for exp in exps:
    #-----------------------------------------------------------------------------
    ## Experiment 'r2b8_NOCV' has different time
    #-----------------------------------------------------------------------------
    if exp == 'r2b8_NOCV':
        dat = '_197902_197906'
    else:
        dat = '_198001_198212'
    #-----------------------------------------------------------------------------
    ## Reading temperature and calculating dT
    #-----------------------------------------------------------------------------
    T  = xr.open_dataset(input+exp+'_ctl_ta_LWtr'+dat+'.nc')#.isel(height=slice(5,7))
    T  = T.ta.mean(dim=('time','lon'))
    T4  = xr.open_dataset(input+exp+'_4k_ta_LWtr'+dat+'.nc')#.isel(height=slice(5,7))
    T4  = T4.ta.mean(dim=('time','lon'))
    dt_trop = T4.sel(lat=slice(-30,30)).mean(dim='lat')-T.sel(lat=slice(-30,30)).mean(dim='lat')
    del T
    del T4
    #-----------------------------------------------------------------------------
    ## Calculating dqs_ctl and dqs_4k
    #-----------------------------------------------------------------------------
    HU = xr.open_dataset(input+exp+'_ctl_hur_LWtr'+dat+'.nc')#.isel(height=slice(5,7))
    HU = HU.hur
    R  = xr.where((HU == 0.0), 0.0001, HU)
    del HU
    R = R.mean(dim=('time','lon')) 
    q = xr.open_dataset(input+exp+'_ctl_hus_LWtr'+dat+'.nc')#.isel(time=slice(t_o,t_f)).isel(height=slice(5,7))
    q = q.hus.mean(dim=('time','lon'))
    qs_t = (q.sel(lat=slice(-30,30))/R.sel(lat=slice(-30,30))).mean(dim='lat')
    del R
    del q
    HU = xr.open_dataset(input+exp+'_4k_hur_LWtr'+dat+'.nc')#.isel(height=slice(5,7))
    HU = HU.hur
    R  = xr.where((HU == 0.0), 0.0001, HU)
    del HU
    R = R.mean(dim=('time','lon')) 
    q = xr.open_dataset(input+exp+'_4k_hus_LWtr'+dat+'.nc')#.isel(height=slice(5,7))
    q = q.hus.mean(dim=('time','lon'))
    qs4_t= (q.sel(lat=slice(-30,30))/R.sel(lat=slice(-30,30))).mean(dim='lat')
    del R,q
    #-----------------------------------------------------------------------------
    ## Calculating CC
    #-----------------------------------------------------------------------------
    CC_trop = ((qs4_t-qs_t)/(dt_trop*qs_t)*100).mean().values
    del qs_t, qs4_t
    #
    #-----------------------------------------------------------------------------
    ## Reading precipitation data
    #-----------------------------------------------------------------------------
    pr = xr.open_dataarray(input+exp+'_ctl_pr_sfc'+dat+'.nc').sel(lat=slice(-30,30))*86400
    per = np.arange(99, 100, 0.001)/100
    #-----------------------------------------------------------------------------
    ## Defining and running fucntions to retireve the distribution tail.
    #-----------------------------------------------------------------------------
    def prct(x): return (pr.where(pr >= pr.quantile(x,interpolation='nearest'))).mean()
    perpr = np.asarray(list(map(prct, per)))
    pr4 = xr.open_dataarray(input+exp+'_4k_pr_sfc'+dat+'.nc').sel(lat=slice(-30,30))*86400
    def prct4(x): return (pr4.where(pr4 >= pr4.quantile(x,interpolation='nearest'))).mean()
    perpr4 = np.asarray(list(map(prct4, per)))
    rapr=(perpr4-perpr)/perpr*100/4
    del pr,pr4
    #-----------------------------------------------------------------------------
    # Saving results in npz format
    #-----------------------------------------------------------------------------
    savez_dict = dict()
    savez_dict['CC_trop'] = CC_trop
    savez_dict['per'] = per
    savez_dict['perpr'] = perpr
    savez_dict['perpr4'] = perpr4
    savez_dict['rapr'] = rapr    
    np.savez(input+'perpr_'+exp+'.npz', **savez_dict)
    print(str(exp)+' DONE!')