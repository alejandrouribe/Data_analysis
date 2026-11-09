"""
This script calculates Iorg (Tompkins & Semie, 2017) for subdomains of the native grid of ICON. 
Script used in the German supercomputer Levante for the NextGEMS hackathon Vienna, 2022.
@author: Alejandro UC
"""
#
## Libraries
#
import intake
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import sklearn.neighbors as skl
#-----------------------------------------------------------------------------
## Input, output paths and initial values
#-----------------------------------------------------------------------------
res=10
model='ngc2009'
time_0='2020-01-21'
time_1='2020-01-23'
#
output='/work/bb1153/m300648/NextGEMS/outdata/'
#-----------------------------------------------------------------------------
## Retrieving and preprocessing initial data
#-----------------------------------------------------------------------------
catalog_file="/work/ka1081/Catalogs/dyamond-nextgems.json"
cat=intake.open_esm_datastore(catalog_file).search(experiment_id='nextgems_cycle2')
hits=cat.search(variable_id=['rlut','pr'])
dic=hits.search(simulation_id=[model])
dataset_dict = dic.to_dataset_dict(cdf_kwargs={"chunks": {"time": 1}})
for name, da in dataset_dict.items():
    data=da
del da
grid_file_path = "/pool/data/ICON" + data.grid_file_uri.split(".de")[1]
grid_data = xr.open_dataset(grid_file_path).rename(
            cell="ncells",  # the dimension has different names in the grid file and in the output.
        )
data = xr.merge((data, grid_data))
md = data.sel(time=slice(time_0,time_1))[["rlut","pr"]]
#
data_daily = (md.resample(time="1D", skipna=True).mean())
del md
#-----------------------------------------------------------------------------
## Reading time
#-----------------------------------------------------------------------------
TIME=data_daily.time
#-----------------------------------------------------------------------------
# Initializing results xarrays.
#-----------------------------------------------------------------------------
Iorg=xr.DataArray(data=np.full([len(TIME),6,36], np.nan), dims=["time","lat","lon"],
                  coords=dict(lon=(["lon"],np.arange(-175.,180.,res)),lat=(["lat"], np.arange(-25.,30.,res)),time=TIME,))
Iorg.name = 'Iorg'
Iorg.attrs['standard_name'] = "Iorg"
Iorg.attrs['long_name'] = "Iorg"
#
Iorg.lon.name = 'longitude'
Iorg.lon.attrs['standard_name'] = "longitude"
Iorg.lon.attrs['long_name'] = "longitude"
Iorg.lon.attrs['units'] = "degrees_east"
Iorg.lon.attrs['axis'] = "X"
#
Iorg.lat.name = 'latitude'
Iorg.lat.attrs['standard_name'] = "latitude"
Iorg.lat.attrs['long_name'] = "latitude"
Iorg.lat.attrs['units'] = "degrees_north"
Iorg.lat.attrs['axis'] = "Y"
#
pr_max=xr.full_like(Iorg,np.nan)
pr_max.name = 'Maximun precipitation'
pr_max.attrs['standard_name'] = "Maximun precipitation"
pr_max.attrs['long_name'] = "Maximun precipitation"
pr_max.attrs['units'] = "kg m-2 d-1"
#
pr_mean=xr.full_like(Iorg,np.nan)
pr_mean.name = 'Mean precipitation'
pr_mean.attrs['standard_name'] = "Mean precipitation"
pr_mean.attrs['long_name'] = "Mean precipitation"
pr_mean.attrs['units'] = "kg m-2 d-1"
#-----------------------------------------------------------------------------
## Looping along subdomains
#-----------------------------------------------------------------------------
for latitude in range(-30,30,res):
    for longitude in range(-180,180,res):
        print('Starting for '+str(latitude)+'_'+str(longitude))
        #-----------------------------------------------------------------------------
        ## Subsetting large domain
        #-----------------------------------------------------------------------------
        mask = (
            (data.clat.values > np.deg2rad(latitude))
            & (data.clat.values < np.deg2rad(latitude+res))
            & (data.clon.values > np.deg2rad(longitude))
            & (data.clon.values < np.deg2rad(longitude+res))
        )
        #-----------------------------------------------------------------------------
        ## Looping along time
        #-----------------------------------------------------------------------------
        for tim in range(0,len(TIME)):
            rlut=data_daily.rlut.isel(time=tim)[mask]
            #-----------------------------------------------------------------------------
            ## Retrieving maximum and mean subdomain precipitation values
            #-----------------------------------------------------------------------------
            pr_max.isel(time=tim).loc[dict(lat=(latitude+(latitude+res-latitude)/2), 
                                                       lon=longitude+(longitude+res-longitude)/2)] = data_daily.pr.isel(time=tim)[mask].max().values*86400
            pr_mean.isel(time=tim).loc[dict(lat=(latitude+(latitude+res-latitude)/2), 
                                                       lon=longitude+(longitude+res-longitude)/2)] = data_daily.pr.isel(time=tim)[mask].mean().values*86400
            #-----------------------------------------------------------------------------
            # Converting center longitude and latitude from radians to degrees
            #-----------------------------------------------------------------------------
            clon=np.rad2deg(rlut.clon)
            clat=np.rad2deg(rlut.clat)
            #-----------------------------------------------------------------------------
            # Grouping gridpoints
            #-----------------------------------------------------------------------------
            grid_points= np.array([[clon[i].values,clat[i].values] for i in range(0, len(clon))])
            #-----------------------------------------------------------------------------
            # Finding minimun distance between grid points. Needed for clustering later.
            #-----------------------------------------------------------------------------
            treeKX=skl.KDTree(grid_points,leaf_size=1)
            distances,index=treeKX.query(grid_points, k=2) 
            distances=distances[:,1]
            dX=distances.min()
            #-----------------------------------------------------------------------------
            # Finding convective points using a minimun of 190 Wm-2
            #----------------------------------------------------------------------------- 
            conv_lon=clon.values[~np.isnan(rlut.where(rlut<=190))]
            conv_lat=clat.values[~np.isnan(rlut.where(rlut<=190))]
            conv_loc=np.array([[conv_lon[i],conv_lat[i]] for i in range(0, len(conv_lon))])
            #-----------------------------------------------------------------------------
            # If there are not convective points Iorg is undefined
            #-----------------------------------------------------------------------------
            if len(conv_loc)==0:
                print('no conv')
                Iorg.isel(time=tim).loc[dict(lat=(latitude+(latitude+res-latitude)/2), 
                                             lon=longitude+(longitude+res-longitude)/2)] = np.nan
                pr_max.isel(time=tim).loc[dict(lat=(latitude+(latitude+res-latitude)/2), 
                                           lon=longitude+(longitude+res-longitude)/2)] = np.nan
                pr_mean.isel(time=tim).loc[dict(lat=(latitude+(latitude+res-latitude)/2), 
                                           lon=longitude+(longitude+res-longitude)/2)] = np.nan
                continue
            #-----------------------------------------------------------------------------
            # If there is just one convective point Iorg is one since the convective organization is maximum
            #-----------------------------------------------------------------------------
            elif len(conv_loc)==1: #<=1: # if there is just one convective point
                print('one conv')
                Iorg.isel(time=tim).loc[dict(lat=(latitude+(latitude+res-latitude)/2), 
                                             lon=longitude+(longitude+res-longitude)/2)] = 1 

                continue
            else:
                #-----------------------------------------------------------------------------
                # Machine learning algorithm to cluster
                #-----------------------------------------------------------------------------
                db = DBSCAN(eps=(dX), min_samples=1).fit(conv_loc)
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                #-----------------------------------------------------------------------------
                # If there is just one cluster
                #-----------------------------------------------------------------------------
                if n_clusters_==1:
                    Iorg.isel(time=tim).loc[dict(lat=(latitude+(latitude+res-latitude)/2), 
                    lon=longitude+(longitude+res-longitude)/2)] = 1
                else:                        
                    n_noise_ = list(labels).count(-1)
                    unique_labels = set(labels)
                    colors = [plt.cm.Spectral(each)
                              for each in np.linspace(0, 1, len(unique_labels))]
                    centroids=np.zeros((n_clusters_,2))
                    CLUS=np.zeros(n_clusters_)
                    #-----------------------------------------------------------------------------
                    # Retrieving cluster's centroids.
                    #-----------------------------------------------------------------------------
                    for k, col, i in zip(unique_labels, colors, range(n_clusters_)):
                        class_member_mask = (labels == k)
                        xy = conv_loc[class_member_mask & core_samples_mask]
                        centroids[i]=[np.nanmean(xy[:, 0]),np.nanmean(xy[:,1])]
                        CLUS[i]=len(xy)
                    n_clusters_=len(conv_loc)
                    #-----------------------------------------------------------------------------
                    # Machine learning algorithm to calculate the nearest neighbor distances
                    #-----------------------------------------------------------------------------
                    treeKX=skl.KDTree(centroids,leaf_size=1)
                    distances,index=treeKX.query(centroids, k=2) 
                    distances=distances[:,1]
                    centroids=centroids[:,::-1]
                    distances=np.zeros((index[:,1].shape))
                    #-----------------------------------------------------------------------------
                    # Retrieving distances between cluster's centroids.
                    #-----------------------------------------------------------------------------
                    for i,j in zip(range(len(index)),index[:,1]):
                        x1=centroids[i,0]
                        y1=centroids[i,1]
                        x2=centroids[j,0]
                        y2=centroids[j,1]    
                        distances[i]=np.sqrt((x2-x1)**2+(y2-y1)**2)*111
                    #-----------------------------------------------------------------------------
                    # Estimating Weibull distribution of the distances
                    #-----------------------------------------------------------------------------
                    lambd = (n_clusters_)/(((clat.max()-clat.min())*(clon.max()-clon.min())).values*111**2)
                    weib = 1-np.exp(-lambd*np.pi*(np.sort(distances)**2))
                    #-----------------------------------------------------------------------------
                    # Estimating Cumulative Distribution Function of centroid's distances.
                    #-----------------------------------------------------------------------------
                    cdf=np.zeros(distances.shape)
                    for j,i in enumerate(np.sort(distances)):
                        cdf[j]=len(np.asarray(np.where(distances<=i))[0,:])/len(distances)
                    #-----------------------------------------------------------------------------
                    # Estimating area between CDF and Weibull (Iorg).
                    #-----------------------------------------------------------------------------
                    Iorg.isel(time=tim).loc[dict(lat=(latitude+(latitude+res-latitude)/2), 
                            lon=longitude+(longitude+res-longitude)/2)] = np.trapz(cdf,weib)
        print('Done for '+str(latitude)+'_'+str(longitude))
#-----------------------------------------------------------------------------
# Saving results in netcdf format
#-----------------------------------------------------------------------------
ds = Iorg.to_dataset(name = 'Iorg')
ds['pr_max'] = pr_max
ds['pr_mean'] = pr_mean
ds.to_netcdf(output+model+'/Iorg_'+model+'_'+time_0+'_'+time_1+'.nc')
print('Done for '+model+'_'+time_0+'_'+time_1)