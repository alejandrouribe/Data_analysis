"""
Feedbacks are estimated as linear ordinary least squares regression coefficients between TOA flux anomalies and T.
This script calculates feedbacks regressing local TOA flux anomalies against local surface air temperature anomalies 
and against globally averaged surface air temperature anomalies.
@author: Alejandro UC
"""
#
## Libraries
#
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import acf
import glob
from cdo import *
#-----------------------------------------------------------------------------
## Input and output paths
#-----------------------------------------------------------------------------
input='/home/alejandro/CL_feedbacks/data/'
output='/home/alejandro/CL_feedbacks/data_prod/'
#-----------------------------------------------------------------------------
## Function to estimate feedbacks as linear regresion coefficents (confidence intervals (CI) on the regression slope account for time autocorrelation)
#-----------------------------------------------------------------------------
def regression(sst_mean, y_values):
    x = sm.add_constant(sst_mean.values)
    y = y_values.values
    model = sm.OLS(y, x, missing='drop')
    fitted = model.fit()
    #-----------------------------------------------------------------------------
    # Confidence intervals
    #-----------------------------------------------------------------------------
    n = len(x)
    SE = fitted.bse[1]
    #-----------------------------------------------------------------------------
    # Since the data is time-autocorrelated, the samples are not independent, so a normal t-test would understimate CI.
    # To properly calculate CI, the degrees of freedom for non-independent time series must be used in the t-test (dof=N/2te, where te is e-folding decay time of autocorrelation)
    #-----------------------------------------------------------------------------
    for i,acc in enumerate(acf(sst_mean)):
        if acc<1/np.exp(1):
            Te = i
            val_Te = acc
            break
    dof=(len(sst_mean)*1)/(2*Te)
    #-----------------------------------------------------------------------------
    # Two-tailed t-test
    #-----------------------------------------------------------------------------
    Q=1-0.05/2 
    t = stats.t.ppf(q=Q, df=dof)
    CI = t*SE
    IC = np.asarray([fitted.params[1]-CI,fitted.params[1]+CI])
    return(fitted.params[1],IC)
#-----------------------------------------------------------------------------
## Feedbacks calculation
#-----------------------------------------------------------------------------
# Latitudinal band widht in degrees.
#-----------------------------------------------------------------------------
lat_int=range(-90,90,interval)
interval=10 
North=np.arange(0,90,10)
South=np.arange(0,-90,-10)
full=[North,South]
#-----------------------------------------------------------------------------
# Looping along hemispheres.
#-----------------------------------------------------------------------------
for Hem in full: 
    if np.all(Hem == North):
        upp=10
        H='North'
    elif np.all(Hem == South):
        upp=-10
        H='South'
    #-----------------------------------------------------------------------------
    # Looping all-sky, clear-sky and radiative effects fluxes.
    #-----------------------------------------------------------------------------
    for rad in ('all','cs','cre'): 
        if rad == 'all':
            IR='rlut'
            IR_obs='toa_lw_all_mon'
            SR='rsut'
            SR_obs='toa_sw_all_mon'
        elif rad=='cs':
            IR='rlutcs'
            IR_obs='toa_lw_clr_c_mon'
            SR='rsutcs'
            SR_obs='toa_sw_clr_c_mon'
        else:
            IR='rlutcre'
            IR_obs='toa_lw_cre_mon'
            SR='rsutcre' 
            SR_obs='toa_sw_cre_mon'
        #-----------------------------------------------------------------------------
        # Initializing results dictionary.
        #-----------------------------------------------------------------------------
        savez_dict = dict() 
        #-----------------------------------------------------------------------------
        # Calculation of feedbacks based on global and local temperature perturbations.
        #-----------------------------------------------------------------------------
        for l,domain in enumerate(('Global','Local')):
            feed_lw=np.zeros(9)
            feed_sw=np.zeros_like(feed_lw)
            feed=np.zeros_like(feed_lw)
            top_lw_IC=np.zeros_like(feed_lw)
            bot_lw_IC=np.zeros_like(feed_lw)
            top_sw_IC=np.zeros_like(feed_lw)
            bot_sw_IC=np.zeros_like(feed_lw)
            top_IC=np.zeros_like(feed_lw)
            bot_IC=np.zeros_like(feed_lw)
            for j,i, in enumerate(Hem):
                #-----------------------------------------------------------------------------
                ## Observed Feedbacks calculation.
                #-----------------------------------------------------------------------------
                DATA_sst=xr.open_dataset(input+'HadCRUT/detre_desea_HadCRUT.5.0.1.0.mean_2001_2014_mon.nc')
                sst_anom=DATA_sst.tas_mean
                #-----------------------------------------------------------------------------
                # To calculate the zonal band feedbacks, the effect of decreasing grid cell area towards the poles must be taken into account. 
                # Thus, prior to computing the feedbacks, the data is weighted with the cosine of the latitude, which has a proportional value to the size of the grid cells
                #-----------------------------------------------------------------------------
                weights = np.cos(np.deg2rad(DATA_sst.latitude))
                weights.name = "weights"
                if domain=='Local':           
                    sst_anom=sst_anom.sel(latitude=slice(min(i,i+upp),max(i,i+upp)))
                    weights = np.cos(np.deg2rad(sst_anom.latitude))
                    weights.name = "weights"
                sst_mean=sst_anom.weighted(weights).mean(dim=('latitude','longitude'), skipna=True)
                DATA_flux=xr.open_dataset(input+'CERES/detre_desea_flux_anom_200101-201412'+add+'.nc', decode_times=False)#.sel(time=slice(None,'2013-12-15T00:00:00.000000000'))
                sw_flux=DATA_flux[SR_obs].sel(lat=slice(min(i,i+upp),max(i,i+upp)))
                weights = np.cos(np.deg2rad(sw_flux.lat))
                weights.name = "weights"
                lw_flux=DATA_flux[IR_obs].sel(lat=slice(min(i,i+upp),max(i,i+upp)))
                sw_mean=sw_flux.weighted(weights).mean(dim=('lat','lon'))
                lw_mean=lw_flux.weighted(weights).mean(dim=('lat','lon'))
                #------------------------ SW FLUX ------------------------------------
                m_sw,sw_IC=regression(sst_mean,-sw_mean)
                bot_sw_IC[j]=np.min(sw_IC)
                top_sw_IC[j]=np.max(sw_IC)
                feed_sw[j]=m_sw
                #------------------------ LW FLUX ------------------------------------
                m_lw,lw_IC,=regression(sst_mean,-lw_mean)
                bot_lw_IC[j]=np.min(lw_IC)
                top_lw_IC[j]=np.max(lw_IC)
                feed_lw[j]=m_lw
                #------------------------ NET FLUX -----------------------------------
                m,IC=regression(sst_mean,-lw_mean-sw_mean)
                bot_IC[j]=np.min(IC)
                top_IC[j]=np.max(IC)
                feed[j]=m
            #-----------------------------------------------------------------------------
            # Storing results in dictionary.
            #-----------------------------------------------------------------------------
            savez_dict['feed_lw_'+domain] = feed_lw
            savez_dict['top_lw_IC_'+domain] = top_lw_IC
            savez_dict['bot_lw_IC_'+domain] = bot_lw_IC
            savez_dict['feed_sw_'+domain] = feed_sw
            savez_dict['top_sw_IC_'+domain] = top_sw_IC
            savez_dict['bot_sw_IC_'+domain] = bot_sw_IC
            savez_dict['feed_'+domain] = feed
            savez_dict['top_IC_'+domain] = top_IC
            savez_dict['bot_IC_'+domain] = bot_IC
            #-----------------------------------------------------------------------------
            # Looping along model experimets.
            #-----------------------------------------------------------------------------
            for exp in (('CMIP6_historical','AMIP','CMIP6_4xCO2')):
                    models=[]
                    #-----------------------------------------------------------------------------
                    # Retrieving model names from directory
                    #-----------------------------------------------------------------------------
                    for i in glob.glob(input+exp+'/*'):
                        models.append(i.replace('/home/alejandro/CL_feedbacks/data/'+exp,''))
                    #-----------------------------------------------------------------------------
                    # Initializing result variables
                    #-----------------------------------------------------------------------------
                    mean_lw=np.zeros_like(feed_lw)
                    mean_sw=np.zeros_like(feed_lw)
                    mean=np.zeros_like(feed_lw)
                    mod_m_lw=np.zeros_like(feed_lw)
                    mod_m_sw=np.zeros_like(feed_lw)
                    mod_m=np.zeros_like(feed_lw)
                    mod_lw_IC_min=np.zeros_like(feed_lw)
                    mod_sw_IC_min=np.zeros_like(feed_lw)
                    mod_IC_min=np.zeros_like(feed_lw)
                    mod_lw_IC_max=np.zeros_like(feed_lw)
                    mod_sw_IC_max=np.zeros_like(feed_lw)
                    mod_IC_max=np.zeros_like(feed_lw)
                    mean_mod_lw=[]
                    mean_mod_sw=[]
                    mean_mod=[]
                    #-----------------------------------------------------------------------------
                    # Looping along models
                    #-----------------------------------------------------------------------------
                    for mod in models:
                        for j,i, in enumerate(Hem):
                        #-----------------------------------------------------------------
                        # Historical simulations
                        #-----------------------------------------------------------------
                            if  exp=='CMIP6_historical' or exp=='AMIP':
                                LW=xr.open_dataset(input+exp+mod+'/detre_desea_'+IR+'_anom.nc')
                                rlut=LW[IR.replace('cre','')].sel(lat=slice(min(i,i+upp),max(i,i+upp)))
                                weights_mod = np.cos(np.deg2rad(rlut.lat))
                                weights_mod.name = "weights"
                                rlut=rlut.weighted(weights_mod).mean(dim=('lat','lon'))
                                SW=xr.open_dataset(input+exp+mod+'/detre_desea_'+SR+'_anom.nc')
                                rsut=SW[SR.replace('cre','')].sel(lat=slice(min(i,i+upp),max(i,i+upp)))
                                rsut=rsut.weighted(weights_mod).mean(dim=('lat','lon'))
                                TS=xr.open_dataset(input+exp+mod+'/detre_desea_ts_anom.nc')
                                if domain=='Global':
                                    weights_glob = np.cos(np.deg2rad(TS.lat))
                                    weights_glob.name = "weights"
                                    ts=TS.ts.weighted(weights_glob).mean(dim=('lat','lon'))
                                else:
                                    ts=TS.ts.sel(lat=slice(min(i,i+upp),max(i,i+upp)))
                                    ts=ts.weighted(weights_mod).mean(dim=('lat','lon'))
                                #------------------------ SW FLUX ------------------------------------
                                m_sw,sw_IC=regression(ts,-rsut)
                                mod_m_sw[j]=m_sw
                                mod_sw_IC_min[j]=np.min(sw_IC)
                                mod_sw_IC_max[j]=np.max(sw_IC)
                                #------------------------ LW FLUX ------------------------------------
                                m_lw,lw_IC=regression(ts,-rlut)
                                mod_m_lw[j]=m_lw
                                mod_lw_IC_min[j]=np.min(lw_IC)
                                mod_lw_IC_max[j]=np.max(lw_IC)
                                #------------------------ NET FLUX -----------------------------------
                                m,IC=regression(ts,-rlut-rsut)
                                mod_m[j]=m
                                mod_IC_min[j]=np.min(IC)
                                mod_IC_max[j]=np.max(IC)
                            #-----------------------------------------------------------------
                            # abrupt 4XCO2 simulations
                            #-----------------------------------------------------------------
                            else:
                                LW_pi=xr.open_dataset(input+exp+mod+'/ymean_'+IR+'_piControl.nc',use_cftime=False,decode_times=False).isel(time=slice(None,150))
                                LW_ab=xr.open_dataset(input+exp+mod+'/ymean_'+IR+'_abrupt-4xCO2.nc',use_cftime=False,decode_times=False).isel(time=slice(None,150))
                                rlut_pi=LW_pi[IR.replace('cre','')].sel(lat=slice(min(i,i+upp),max(i,i+upp)))
                                rlut_ab=LW_ab[IR.replace('cre','')].sel(lat=slice(min(i,i+upp),max(i,i+upp)))
                                weights_mod = np.cos(np.deg2rad(rlut_pi.lat))
                                weights_mod.name = "weights"
                                rlut=rlut_ab.weighted(weights_mod).mean(dim=('lat','lon'))-rlut_pi.weighted(weights_mod).mean(dim=('lat','lon'))
                                SW_pi=xr.open_dataset(input+exp+mod+'/ymean_'+SR+'_piControl.nc',use_cftime=False,decode_times=False).isel(time=slice(None,150))
                                SW_ab=xr.open_dataset(input+exp+mod+'/ymean_'+SR+'_abrupt-4xCO2.nc',use_cftime=False,decode_times=False).isel(time=slice(None,150))
                                rsut_pi=SW_pi[SR.replace('cre','')].sel(lat=slice(min(i,i+upp),max(i,i+upp)))
                                rsut_ab=SW_ab[SR.replace('cre','')].sel(lat=slice(min(i,i+upp),max(i,i+upp)))
                                rsut=rsut_ab.weighted(weights_mod).mean(dim=('lat','lon'))-rsut_pi.weighted(weights_mod).mean(dim=('lat','lon'))
                                TS_pi=xr.open_dataset(input+exp+mod+'/ymean_ts_piControl.nc',use_cftime=False,decode_times=False).isel(time=slice(None,150))
                                TS_ab=xr.open_dataset(input+exp+mod+'/ymean_ts_abrupt-4xCO2.nc',use_cftime=False,decode_times=False).isel(time=slice(None,150))
                                if domain=='Global':
                                    weights_glob = np.cos(np.deg2rad(TS_pi.lat))
                                    weights_glob.name = "weights"
                                    ts=TS_ab.ts.weighted(weights_glob).mean(dim=('lat','lon'))-TS_pi.ts.weighted(weights_glob).mean(dim=('lat','lon'))
                                else:
                                    ts_pi=TS_pi.ts.sel(lat=slice(min(i,i+upp),max(i,i+upp)))
                                    ts_ab=TS_ab.ts.sel(lat=slice(min(i,i+upp),max(i,i+upp)))
                                    ts=ts_ab.weighted(weights_mod).mean(dim=('lat','lon'))-ts_pi.weighted(weights_mod).mean(dim=('lat','lon'))
                                #------------------------ SW FLUX ------------------------------------
                                m_sw,sw_IC=regression(ts,-rsut)
                                mod_m_sw[j]=m_sw
                                mod_sw_IC_min[j]=np.min(sw_IC)
                                mod_sw_IC_max[j]=np.max(sw_IC)
                                #------------------------ LW FLUX ------------------------------------
                                m_lw,lw_IC=regression(ts,-rlut)
                                mod_m_lw[j]=m_lw
                                mod_lw_IC_min[j]=np.min(lw_IC)
                                mod_lw_IC_max[j]=np.max(lw_IC)
                                #------------------------ NET FLUX -----------------------------------
                                m,IC=regression(ts,-rlut-rsut)
                                mod_m[j]=m
                                mod_IC_min[j]=np.min(IC)
                                mod_IC_max[j]=np.max(IC)
                        mean_mod_lw.append(np.copy(mod_m_lw))
                        mean_mod_sw.append(np.copy(mod_m_sw))
                        savez_dict[exp+'_'+mod.replace('/','')+'_m_lw_'+domain] = np.copy(mod_m_lw)
                        savez_dict[exp+'_'+mod.replace('/','')+'_lw_IC_min_'+domain] = np.copy(mod_lw_IC_min)
                        savez_dict[exp+'_'+mod.replace('/','')+'_lw_IC_max_'+domain] = np.copy(mod_lw_IC_max)
                        savez_dict[exp+'_'+mod.replace('/','')+'_m_sw_'+domain] = np.copy(mod_m_sw)
                        savez_dict[exp+'_'+mod.replace('/','')+'_sw_IC_min_'+domain] = np.copy(mod_sw_IC_min)
                        savez_dict[exp+'_'+mod.replace('/','')+'_sw_IC_max_'+domain] = np.copy(mod_sw_IC_max)
                        savez_dict[exp+'_'+mod.replace('/','')+'_m_'+domain] = np.copy(mod_m)
                        savez_dict[exp+'_'+mod.replace('/','')+'_IC_min_'+domain] = np.copy(mod_IC_min)
                        savez_dict[exp+'_'+mod.replace('/','')+'_IC_max_'+domain] = np.copy(mod_IC_max)
                    savez_dict['mean_mod_lw_'+exp+'_'+domain] = np.mean(mean_mod_lw,axis=0)
                    savez_dict['mean_mod_sw_'+exp+'_'+domain] = np.mean(mean_mod_sw,axis=0)
                    savez_dict['mean_mod_'+exp+'_'+domain] = np.mean(mean_mod,axis=0)
        #-----------------------------------------------------------------------------
        # Saving results in npy format
        #-----------------------------------------------------------------------------
        np.savez(output+'CI_Feed_Ftropics_'+rad+'_NOSI_NOI_'+H+'_Had5_V2.npz', **savez_dict)        