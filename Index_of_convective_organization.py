"""
This script calculates a simple organization index that permits to classify a field as regular, random or clustered (Iorg, Tompkins & Semie, 2017). 
We use a threshold of vertical velocity (omega) higher or equal to the mean subsiding omega at the level of 500 hPa to distinguish convective grid cells. 
Then, to identify convective grid cells that are part of the same cluster, eight point connectivity is employed.
Unsupervised machine learning is used to create convective cluseters and fined the nearest neighbors among convective clusters.
@author: Alejandro UC
"""
#
## Libraries
#
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import sklearn.neighbors as skl
from sklearn.neighbors import NearestNeighbors
import sys
from cdo import *   
cdo = Cdo()
#-----------------------------------------------------------------------------
## Input and output paths
#-----------------------------------------------------------------------------
input='/home/alejandro/scratch/DATA_2_3/'
output='/home/alejandro/scratch/DATA_2_3/'
#-----------------------------------------------------------------------------
## Function to calculate Iorg
#-----------------------------------------------------------------------------
def Iorg_calculation(data):
    DATA=xr.open_dataset(input+data).sel(lat=slice(-30,30))
    wap=DATA.wap
    #-----------------------------------------------------------------------------
    # Initializing results dictionary.
    #-----------------------------------------------------------------------------
    Iorg=np.zeros(len(range(0,len(wap.time))))
    DIST=[]
    CLUSTERS=[]
    #-----------------------------------------------------------------------------
    # Looping along time.
    #-----------------------------------------------------------------------------
    for l,tim in enumerate(range(0,len(wap.time))):
        W=wap.isel(time=tim).isel(height=0)
        W=W.where(W<0)
        W_high=(W.where(W<W.mean()))
        #-----------------------------------------------------------------------------
        # To extract the coordinates where deep convection occurs 
        #-----------------------------------------------------------------------------
        conv_loc=W_high.to_dataframe().dropna().reset_index()[['lon', 'lat']].values
        #-----------------------------------------------------------------------------
        # Machine learning algorithm to cluster
        #-----------------------------------------------------------------------------
        dx=DATA.lon.diff(dim='lon').mean().values
        dy=DATA.lat.diff(dim='lat').mean().values
        #-----------------------------------------------------------------------------
        # The miminum distance to cluster is that of the hypotenuse of the gird cells
        #-----------------------------------------------------------------------------
        db = DBSCAN(eps=(np.sqrt(dx**2+dy**2)), min_samples=1).fit(conv_loc)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        #-----------------------------------------------------------------------------
        # Initializing clusters array.
        #-----------------------------------------------------------------------------
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
        #-----------------------------------------------------------------------------
        # Machine learning algorithm to calculate the nearest neighbor distances
        #-----------------------------------------------------------------------------
        neigh = NearestNeighbors(n_neighbors=2, radius=1.0)
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
        lambd = (n_clusters_)/(((W.lat.max()-W.lat.min())*(W.lon.max()-W.lon.min())).values*111**2)
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
        Iorg[l]=np.trapz(cdf,weib)
        CLUSTERS.append(CLUS)
        DIST.append(distances)
    return(Iorg)
#-----------------------------------------------------------------------------
# Testing for lat-long model experiment (ICOsahedral Nonhydrostatic Atmospheric general circulation model)
# with resolution ~ 40 km
#-----------------------------------------------------------------------------
experiment='r2b6_CV_ctl_'
Iorg_ctl=Iorg_calculation(experiment+'wap_500hpa_198001_198212.nc')
experiment='r2b6_CV_4k_'
Iorg_4k=Iorg_calculation(experiment+'wap_500hpa_198001_198212.nc')
#-----------------------------------------------------------------------------
# Saving results in npy format
#-----------------------------------------------------------------------------
np.savez(output+'IORG_r2b6CV_Cluster_corr.npz', Iorg_ctl=Iorg_ctl, Iorg_4k=Iorg_4k)
