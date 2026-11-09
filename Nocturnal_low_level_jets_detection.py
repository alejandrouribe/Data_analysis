"""
This script detects Nocturnal Low Level Jets (NLLJ) from reanalysis data (ERA-5) 
and calculates their speed, height, shear above and virtual temperature gradient. 
@author: Alejandro UC
"""
#-----------------------------------------------------------------------------
## Libraries
#-----------------------------------------------------------------------------
import time
import xarray as xr
import numpy as np
start_time = time.time()
#-----------------------------------------------------------------------------
## Input, output paths
#-----------------------------------------------------------------------------
input='/work/bb1198/m300648/NLLJ/data/'
output='/work/bb1198/m300648/NLLJ/output/'
#
#-----------------------------------------------------------------------------
#-----Reading netCDF data
#-----------------------------------------------------------------------------
year=2000
month=01
day=01
syno_130=xr.open_dataset(input+year+'/'+month+'/E5ml00_6H_'+year+'-'+month+'-130_gpl_'+day+'.nc')
syno_131=xr.open_dataset(input+year+'/'+month+'/E5ml00_6H_'+year+'-'+month+'-131_gpl_'+day+'.nc')
syno_132=xr.open_dataset(input+year+'/'+month+'/E5ml00_6H_'+year+'-'+month+'-132_gpl_'+day+'.nc')
syno_133=xr.open_dataset(input+year+'/'+month+'/E5ml00_6H_'+year+'-'+month+'-133_gpl_'+day+'.nc')
syno_133['lat']=syno_130.lat # Forcing the varaiables to have the same latitudes. Sometimes slightly differ.
slp=xr.open_dataset(input+year+'/'+month+'/E5sf00_6H_'+year+'-'+month+'_134_gpl_'+day+'.nc')
slp['lat']=syno_130.lat # Forcing the varaiables to have the same latitudes. Sometimes slightly differ.
#-----------------------------------------------------------------------------
#----- Declaring variables
#-----------------------------------------------------------------------------
lljwspdid=xr.full_like(slp.var134, np.nan, dtype=None)
lljhid=xr.full_like(slp.var134, np.nan, dtype=None)
wspdshear=xr.full_like(slp.var134, np.nan, dtype=None)
lljvt=xr.full_like(slp.var134, np.nan, dtype=None)
Ri=xr.full_like(slp.var134, np.nan, dtype=None)
#-----------------------------------------------------------------------------
#----- Calculation is made day by day
#-----------------------------------------------------------------------------
for t in syno_130.time:
    P_sfc=slp.var134.sel(time=t) # Surface pressure
    T=syno_130.t.sel(time=t) # Temperature
    q=syno_133.q.sel(time=t) # Specific humidity
    g=-9.81 # Gravity
    P=syno_130.hyam+syno_130.hybm*P_sfc # Mid levels
    P=P.rename({'nhym': 'lev'}) # Renaming nhym coordinate to lev to have consitency in the data
    rho=P/(287*T)*(1+q)/(1+1.609*q) # Density
    theta=T*(100000./P)**(0.286) # Potential temperature
    #-----------------------------------------------------------------------------
    #----- converting P to z (meters) using the hydrostatic approximation
    #-----------------------------------------------------------------------------
    dz=-P.diff('lev',n=1)/(rho.isel(lev=slice(None,-1))*g)
    hgt_rel=xr.zeros_like(P, dtype=None)
    z=xr.zeros_like(P_sfc, dtype=None)
    for i in reversed(dz.lev):
        hgt_rel.loc[dict(lev=int(i))]=z
        z=z+dz.sel(lev=i)               
    hgt_rel.loc[dict(lev=int(i)-1)]=z
    hgt_rel=hgt_rel.assign_coords(lev=np.arange(1,len(P.lev)+1))
    #-----------------------------------------------------------------------------
    #----- calculating virtual tempearture criterion for NLLJ 
    #-----------------------------------------------------------------------------
    e=0.22
    Tv=theta*(1.+q/e)/(1+q) # Virtual potential temperature
    wspd=np.sqrt((syno_131.u.sel(time=t)**2)+(syno_132.v.sel(time=t)**2.)) # Wind speed
    hgt_100=abs(hgt_rel-100).min(dim='lev') # Get the level index of level ~100 m above the sfc
    hgt_1500=abs(hgt_rel-1500).min(dim='lev')# Get the level index of level ~1500 m above the sfc
    dh=hgt_rel.where(abs(hgt_rel-100)==hgt_100,drop=True)-hgt_rel.isel(lev=len(P.lev)-1) # dh btw ~100 m and sfc
    dTv=Tv.where(abs(hgt_rel-100)==hgt_100,drop=True)-Tv.isel(lev=len(P.lev)-1) # Dtv btw ~100 m and sfc 
    Tv_grad_cond=(dTv/dh).isel(lev=0)>0.001 # dTv/dh criterion
    #-----------------------------------------------------------------------------
    # Get the location (lon, lat) with stable stratifed surface
    #-----------------------------------------------------------------------------
    Tv_grad_loc=P_sfc.where(Tv_grad_cond).rename('TV_grad').to_dataframe().dropna().reset_index()[['lon', 'lat']].values
    height_top=1500.
    #-----------------------------------------------------------------------------
    #----- calculating shear criterion where virtual criterion is met. 
    #-----------------------------------------------------------------------------
    for loc in Tv_grad_loc:
        la=loc[1] # latitude
        lo=loc[0] # longitude
        idx_top=(np.abs(hgt_rel.sel(lat=la).sel(lon=lo)-height_top)).argmin() # Get the level index of level ~1500m
        H=hgt_rel.sel(lat=la).sel(lon=lo).isel(lev=slice(idx_top.values,None)) # Select height from sfc to ~1500m  
        WSPD=wspd.sel(lat=la).sel(lon=lo).isel(lev=slice(idx_top.values,None)) # Wind speed of the layer btw sfc and 1500m
        WSPD=WSPD.assign_coords(lev=H) # Change height coordinate form index to meters
        WM_loc=WSPD.where(WSPD==WSPD.max(),drop=True).lev# Height at which the maximum wind is located
        #-----------------------------------------------------------------------------
        # If the jet is located at ~1500, then skip this grid point 
        #-----------------------------------------------------------------------------
        if len(WM_loc)>1:
            WM_loc=WM_loc.isel(lev=0).values
        else:
            WM_loc=WM_loc.values[0]
        if WM_loc==H.max().values:
            continue 
        WSPD_OV=wspd.sel(lat=la).sel(lon=lo).assign_coords(lev=hgt_rel.sel(lat=la).sel(lon=lo)) # Take the WSPD along the column
        WSPD_OV=WSPD_OV.sel(lev=slice(None,WM_loc)) # Take the wind speed btw the jet height to the top
        idx_500=(np.abs(hgt_rel.sel(lat=la).sel(lon=lo)-(WM_loc+500.))).argmin() # Get the level index of level ~500 m above the jet core 
        WSPD_500_loc=hgt_rel.sel(lat=la).sel(lon=lo).sel(lev=idx_500.values)# Get the height of the ~500 m above the jet
        WSPD_500=wspd.sel(lat=la).sel(lon=lo).sel(lev=idx_500.values) # Get the wind speed at ~500 m above the jet
        #-----------------------------------------------------------------------------
        # Shear criterion  
        #-----------------------------------------------------------------------------        
        if (WSPD_500-WSPD.max())/(WSPD_500_loc-WM_loc)<-0.005: 
            lljwspdid.loc[dict(lat=la,lon=lo,time=t)]=WSPD.max() # NLLJ speed
            lljhid.loc[dict(lat=la,lon=lo,time=t)]=WM_loc # NLLJ height
            wspdshear.loc[dict(lat=la,lon=lo,time=t)]=((WSPD_500-WSPD.max())/(WSPD_500_loc-WM_loc))#.isel(lev=0)# Shear above the NLLJ
            lljvt.loc[dict(lat=la,lon=lo,time=t)]=(dTv/dh).where(Tv_grad_cond).isel(lev=0).sel(lat=la,lon=lo)# Virtual temperature gradient NLLJ
            Theta=theta.sel(lat=la,lon=lo).assign_coords(lev=hgt_rel.sel(lat=la,lon=lo))# Take the potential temperature along the column
            RI = (((g/((Theta.sel(lev=WM_loc)+Theta.isel(lev=len(P.lev)-1))/2.))*((Theta.sel(lev=WM_loc)-Theta.isel(lev=len(P.lev)-1))/(WM_loc)))/(((WSPD.max()-WSPD.isel(lev=0))/WM_loc)**2))           
            Ri.loc[dict(lat=la,lon=lo,time=t)]= RI
        else:
            print('No shear found over the Jet layer')
    print(str(t.values)+' Completed.')
#-----------------------------------------------------------------------------
# Saving results in netcdf format
#-----------------------------------------------------------------------------
lljwspdid.name = 'NLLJ speed'
lljwspdid.attrs['long_name'] = 'NLLJ speed'
lljwspdid.attrs['units'] = 'm/s'
#
lljhid.name = 'NLLJ height'
lljhid.attrs['long_name'] = 'NLLJ height'
lljhid.attrs['units'] = 'm'
#
wspdshear.name = 'Shear above the NLLJ'
wspdshear.attrs['long_name'] = 'Shear above the NLLJ in 500m deep layer'
wspdshear.attrs['units'] = 's^(-1)'
#
lljvt.name = 'Virtual temperature gradient NLLJ'
lljvt.attrs['long_name'] = 'Virtual temperature gradient in 100m deep surface layer'
lljvt.attrs['units'] = 'K/m'
#
Ri.name = 'Richardson number'
Ri.attrs['long_name'] = 'Richardson number below NLLJ'
Ri.attrs['units'] = 'dimensionless'
#
ds = lljhid.to_dataset(name = 'lljhid')
ds['lljwspdid'] = lljwspdid
ds['wspdshear'] = wspdshear
ds['lljvt'] = lljvt
ds['Ri'] = Ri
ds.to_netcdf(path=output+str(year)+'/'+str(month)+'/NLLJ_'+str(year)+'_'+str(month)+'_'+day+'.nc', mode='w')
print("--- %s seconds ---" % (time.time() - start_time))
