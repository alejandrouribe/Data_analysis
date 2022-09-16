"""
This script calculates Iorg  Tompkins & Semie, 2017. for a subdomain of the native grid of ICON. 
Script used in Levante, German supercomputer for the NextGEMS hackathon Vienna, 2022.
@author: Alejandro UC
"""
#
## Libraries
#
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import sklearn.neighbors as skl
import os
from cdo import *
cdo = Cdo()
#-----------------------------------------------------------------------------
## Input, output paths and initial values
#-----------------------------------------------------------------------------
input='/home/alejandro/scratch/DATA_2_3/'
output='/home/alejandro/scratch/DATA_2_3/'
model=dpp0052
res=10
#-----------------------------------------------------------------------------
## Reading time
#-----------------------------------------------------------------------------
TIME=cdo.selname('rlut', input = input+model+'_pr_rlut.nc',returnXDataset = True).time#.isel(time=slice(1,2))
#-----------------------------------------------------------------------------
# Initializing results xarrays.
#-----------------------------------------------------------------------------
Iorg=xr.DataArray(data=np.full([len(TIME),6,36], np.nan), dims=["time","lat","lon"],
                  coords=dict(lon=(["lon"],np.arange(5.,360.,res)),lat=(["lat"], np.arange(-25.,30.,res)),time=TIME,))
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
## Looping along subdomain
#-----------------------------------------------------------------------------
for latitude in range(-30,30,res):
    for longitude in range(0,360,res):
        print('Starting for '+str(latitude)+'_'+str(longitude))
        #-----------------------------------------------------------------------------
        ## Subsetting large domain
        #-----------------------------------------------------------------------------
        os.system('cdo -sellonlatbox,'+str(longitude)+','+str(longitude+res)+','+str(latitude)+','+str(latitude+res)+' '+input+model+'_pr_rlut.nc '+input+'tmp_'+model+str(latitude)+'_'+str(longitude)+'.nc')
        DATA=xr.open_dataset(input+'tmp_'+model+str(latitude)+'_'+str(longitude)+'.nc')
        #-----------------------------------------------------------------------------
        ## Looping along time
        #-----------------------------------------------------------------------------
        for tim in range(0,len(TIME)):
            rlut=DATA.rlut.isel(time=tim)
            #-----------------------------------------------------------------------------
            ## Retrieving maximum and mean subdomain precipitation values
            #-----------------------------------------------------------------------------
            pr_max.isel(time=tim).loc[dict(lat=(latitude+(latitude+res-latitude)/2), 
                                                       lon=longitude+(longitude+res-longitude)/2)] = DATA.pr.isel(time=tim).max().values*86400
            pr_mean.isel(time=tim).loc[dict(lat=(latitude+(latitude+res-latitude)/2), 
                                                       lon=longitude+(longitude+res-longitude)/2)] = DATA.pr.isel(time=tim).mean().values*86400
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
            conv_lon=clon.values[~np.isnan(rlut.where(rlut<190))]
            conv_lat=clat.values[~np.isnan(rlut.where(rlut<190))]
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
                continue
            #-----------------------------------------------------------------------------
            # If there is just one convective point Iorg is one since the convective organization is maximum
            #-----------------------------------------------------------------------------
            elif len(conv_loc)==1: 
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
        os.system('rm '+input+'tmp_'+model+str(latitude)+'_'+str(longitude)+'.nc')
#-----------------------------------------------------------------------------
# Saving results in netcdf format
#-----------------------------------------------------------------------------
ds = Iorg.to_dataset(name = 'Iorg')
ds['pr_max'] = pr_max
ds['pr_mean'] = pr_mean
ds.to_netcdf(output+'Iorg_'+model+'_NG.nc')
print('Done for '+model)