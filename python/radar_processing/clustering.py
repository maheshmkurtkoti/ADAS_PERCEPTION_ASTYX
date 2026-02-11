import numpy as np
from sklearn.cluster import DBSCAN

def cluster_radar_points(xyz,eps=1.0,min_samples = 5):
    """
    xyz: NX3 radar point cloud
    returns: cluster labels
    """
    clustering= DBSCAN(eps=eps,min_samples=min_samples).fit(xyz)
    return clustering.labels_