#
# Copyright (c) 2015, Yarpiz (www.yarpiz.com)
# All rights reserved. Please read the "license.txt" for license terms.
#
# Project Code: YPML110
# Project Title: Implementation of DBSCAN Clustering in MATLAB
# Publisher: Yarpiz (www.yarpiz.com)
#
# Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
#
# Contact Info: sm.kalami@gmail.com, info@yarpiz.com
#

import numpy as np
    
def DBSCAN(X = None,epsilon = None,MinPts = None): 
    C = 0
    n = X.shape[1-1]
    IDX = np.zeros((n,1))
    D = pdist2(X,X)
    visited = False(n,1)
    isnoise = False(n,1)
    for i in np.arange(1,n+1).reshape(-1):
        if not visited(i) :
            visited[i] = True
            Neighbors = RegionQuery(i)
            if np.asarray(Neighbors).size < MinPts:
                # X(i,:) is NOISE
                isnoise[i] = True
            else:
                C = C + 1
                ExpandCluster(i,Neighbors,C)
    
    
def ExpandCluster(i = None,Neighbors = None,C = None): 
    IDX[i] = C
    k = 1
    while True:

        j = Neighbors(k)
        if not visited(j) :
            visited[j] = True
            Neighbors2 = RegionQuery(j)
            if np.asarray(Neighbors2).size >= MinPts:
                Neighbors = np.array([Neighbors,Neighbors2])
        if IDX(j) == 0:
            IDX[j] = C
        k = k + 1
        if k > np.asarray(Neighbors).size:
            break

    
    return
    
    
def RegionQuery(i = None): 
    Neighbors = find(D(i,:) <= epsilon)
    return Neighbors
    
    return Neighbors
    
    return IDX,isnoise