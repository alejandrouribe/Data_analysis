"""
This script calculates a scalling of tropical extreme precipitation changes with global warming 
based on Muller et al. 2011.
@author: Alejandro UC
"""
#-----------------------------------------------------------------------------
## Libraries
#-----------------------------------------------------------------------------
import numpy as np
import xarray as xr
from cdo import *
cdo = Cdo()
#-----------------------------------------------------------------------------
## Input, output paths
#-----------------------------------------------------------------------------
input='/DATA/'
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
    ## Reading data
    #-----------------------------------------------------------------------------
    HU = xr.open_dataset(input+exp+'_ctl_hur_full'+dat+'.nc').sel(lat=slice(-30,30))
    HU = HU.hur
    R  = xr.where((HU == 0.0), 0.0001, HU) 
    del HU
    q = xr.open_dataset(input+exp+'_ctl_hus_full'+dat+'.nc').sel(lat=slice(-30,30))
    q = q.hus
    QS= (q/R)
    del R, q
    HU = xr.open_dataset((input+exp+'_4k_hur_full'+dat+'.nc').sel(lat=slice(-30,30))
    HU = HU.hur
    R  = xr.where((HU == 0.0), 0.0001, HU) 
    del HU
    q = xr.open_dataset(input+exp+'_4k_hus_full'+dat+'.nc').sel(lat=slice(-30,30))
    q = q.hus
    QS4= (q/R)
    del R, q
    #-----------------------------------------------------------------------------
    ## Control scenario
    #-----------------------------------------------------------------------------
    p  = xr.open_dataset(input+exp+'_ctl_pfull_full'+dat+'.nc').sel(lat=slice(-30,30))
    p  = p.pfull
    pdiff=p.diff('height',n=1)
    p_o=xr.zeros_like(p[:,0:1,:])
    del p
    dqdp = xr.concat([xr.zeros_like(QS[:,0:1,:]),QS.diff('height',n=1)/pdiff],dim='height')
    del QS
    dqdp = dqdp.where(dqdp!=0)
    dp = xr.concat([p_o,pdiff],dim='height').mean(dim=('time', 'lat','lon')).values
    del p_o
    del pdiff
    pr = xr.open_dataarray(input+exp+'_ctl_pr_sfc'+dat+'.nc').sel(lat=slice(-30,30))*86400
    #-----------------------------------------------------------------------------
    ## Warming scenario
    #-----------------------------------------------------------------------------   
    p  = xr.open_dataset(input+exp+'_4k_pfull_full'+dat+'.nc').sel(lat=slice(-30,30))
    p  = p.pfull
    pdiff=p.diff('height',n=1)
    dqdp4 = xr.concat([xr.zeros_like(QS4[:,0:1,:]),QS4.diff('height',n=1)/pdiff],dim='height')
    dqdp4 = dqdp4.where(dqdp4!=0)
    del QS4
    del pdiff
    pr4 = xr.open_dataarray(input+exp+'_4k_pr_sfc'+dat+'.nc').sel(lat=slice(-30,30))*86400
    #-----------------------------------------------------------------------------
    ## Reading additional data
    #-----------------------------------------------------------------------------
    data=np.load(input+'scaling/perpr_'+exp+'.npz')
    per = data['per'] 
    perpr = data['perpr']
    perpr4 = data['perpr4']
    CC_global = data['CC_global']
    CC_trop = data['CC_trop']
    ite = list(range(len(per)))
    w  = xr.open_dataset(input+exp+'_ctl_wap_full'+dat+'.nc').sel(lat=slice(-30,30))
    w  = w.wap
    #-----------------------------------------------------------------------------
    ## Retrieving variables where extreme precipitation occurs
    #-----------------------------------------------------------------------------
    def tow(x): return (w.where(pr>perpr[x])).mean(('time','lat','lon'), skipna=True)
    tow=np.asarray(list(map(tow, ite)))
    del w
    def todq(x): return (dqdp.where(pr>perpr[x])).mean(('time','lat','lon'), skipna=True)
    todq=np.asarray(list(map(todq, ite)))
    del pr
    w4  = xr.open_dataset(input+exp+'_4k_wap_full'+dat+'.nc').sel(lat=slice(-30,30))
    w4  = w4.wap    
    def tow4(x): return (w4.where(pr4>perpr4[x])).mean(('time','lat','lon'), skipna=True)
    tow4=np.asarray(list(map(tow4, ite)))
    del w4
    def todq4(x): return (dqdp4.where(pr4>perpr4[x])).mean(('time','lat','lon'), skipna=True)
    todq4=np.asarray(list(map(todq4, ite)))
    del dqdp4
    del pr4
    #-----------------------------------------------------------------------------
    ## Scalling computation
    #-----------------------------------------------------------------------------
    dyn=(np.nansum((tow4*todq*dp),axis=1)/np.nansum((tow*todq*dp),axis=1))-1.
    ther=(np.nansum((tow*todq4*dp),axis=1)/np.nansum((tow*todq*dp),axis=1))-1.
    rapr=(perpr4-perpr)/perpr
    ctl=-np.nansum((tow*todq*dp/9.81),axis=1)*86400.0
    ctl4=-np.nansum((tow4*todq4*dp/9.81),axis=1)*86400.0
    micro=(perpr4/ctl4*ctl/perpr-1)/4.0*100.0
    total=(dyn+ther)/4.0*100.0+micro
    rapr=(perpr4-perpr)/perpr
    #-----------------------------------------------------------------------------
    # Saving results in npz format
    #-----------------------------------------------------------------------------
    savez_dict = dict()
    savez_dict['dp'] = dp
    savez_dict['tow'] = tow
    savez_dict['todq'] = todq
    savez_dict['tow4'] = tow4
    savez_dict['todq4'] = todq4
    savez_dict['perpr'] = perpr
    savez_dict['perpr4'] = perpr4
    savez_dict['rapr'] = rapr
    savez_dict['dyn'] = dyn
    savez_dict['ther'] = ther
    savez_dict['total'] = total
    savez_dict['ctl'] = ctl
    savez_dict['ctl4'] = ctl4
    savez_dict['CC_global'] = CC_global
    savez_dict['CC_trop'] = CC_trop 
    np.savez(input+'lat_lon-OG_scaling'+exp+'.npz', **savez_dict)