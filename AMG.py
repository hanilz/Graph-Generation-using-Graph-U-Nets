import numpy as np
from scipy.sparse import spdiags, find, issparse


def AMG(fW, beta, NS):
    
    # Graph coarsening using Algebraic MultiGrid (AMG)
    
    # Usage:
#   [P R W c] = AMG(fW, beta, NS)
    
    # Inputs:
#   fW      - affinity matrix at the finest scale.
#             Matrix must be symmetric, sparse and have non-negative
#             entries.
#   beta    - coarsening factor (typical value ~.2)
#   NS      - number of scales / levels
    
    # Outputs
#   P       - interp matrices (from coarse to fine)
#   R       - restriction matrices (from fine to coarse)
#   W       - affective affinity matrix at each scale (W{1}=fW)
#   c       - coarse nodes selected at each scale
    
    # Example:
#   Image super-pixels
    
    #   img = im2double(imread('football.jpg'));
#   sz = size(img);
#   # forming 4-connected graph over image grid
#   [ii jj] = sparse_adj_matrix(sz(1:2), 1, 1, 1);
#   fimg = reshape(img,[],sz(3));
#   wij = fimg(ii,:)-fimg(jj,:);
#   wij = sum(wij.^2, 2);
#   wij = exp(-wij./(2*mean(wij)));
#   fW = sparse(ii, jj, wij, prod(sz(1:2)), prod(sz(1:2)));
    
    #   # forming AMG
#   [P R W c] = AMG(fW, .2, 5);
    
    #   # super pixels
#   [~, sp] = max( P{1}*(P{2}*(P{3}*(P{4}*P{5}))), [], 2);
    
    #   figure;
#   subplot(121);imshow(img);title('input image');
#   subplot(122);imagesc(reshape(sp,sz(1:2)));
#   axis image;colormap(rand(numel(c{5}),3));
#   title('super pixels');
    
    
    #  Copyright (c) Bagon Shai
#  Many thanks to Meirav Galun.
#  Department of Computer Science and Applied Mathmatics
#  Wiezmann Institute of Science
#  http://www.wisdom.weizmann.ac.il/
    
    #  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, subject to the following conditions:
    
    #  1. The above copyright notice and this permission notice shall be included in
#      all copies or substantial portions of the Software.
#  2. No commercial use will be done with this software.
#  3. If used in an academic framework - a proper citation must be included.
    
    # @ELECTRONIC{bagon2012,
#  author = {Shai Bagon},
#  title = {Matlab implementation for AMG graph coarsening},
#  url = {http://www.wisdom.weizmann.ac.il/~bagon/matlab.html#AMG},
#  owner = {bagon},
#  version = {}, <-please insert version number from VERSION.txt
# }
    
    
    #  The Software is provided "as is", without warranty of any kind.
    
    #  May 2011
    
    n = fW.shape[0]
    P = [None] * NS
    c = [None] * NS
    W = [None] * NS

    # make sure diagonal is zero
    W[0] = fW - spdiags(fW.diagonal(), 0, n, n, format='csc')
    fine = np.arange(n)
    for si in np.arange(NS):
        tmp_c, P[si] = fine2coarse(W[si],beta)
        c[si] = fine(tmp_c)
        if si < NS:
            W[si + 1] = np.transpose(P[si]) * W[si] * P[si]
            W[si + 1] = W[si + 1] - spdiags(spdiags(W[si + 1],0),0,W[si + 1].shape[1-1],W[si + 1].shape[2-1])
            fine = c[si]
    
    # restriction matrices
    R = cellfun(lambda x = None: spmtimesd(np.transpose(x),1.0 / np.sum(x, 1-1),[]),P,'UniformOutput',False)
    #-------------------------------------------------------------------------#
    
def fine2coarse(W = None,beta = None):
    # Performs one step of coarsening
    n = W.shape[0]
    # weight normalization
    nW = W.multiply(1.0 / np.sum(W, axis=1))
    c = ChooseCoarseGreedy_mex(nW, np.random.permutation(n), beta)
    
    # compute the interp matrix
    ci = np.where(c)[0]
    P = W[:, ci]
    P = spmtimesd(P,1.0 / full(np.sum(P, 2-1)),[])
    # make sure coarse points are directly connected to their fine counterparts
    jj,ii,pji = find(np.transpose(P))
    sel = not c(ii) 
    
    mycat = lambda x = None,y = None: vertcat(x,y)
    P = np.transpose(sparse(mycat(jj(sel),np.arange(1,sum(c)+1)),mycat(ii(sel),ci),mycat(pji(sel),np.ones((1,sum(c)))),P.shape[2-1],P.shape[1-1]))
    return P,R,W,c


def ChooseCoarseGreedy_mex(nC, ord, beta):
    if len(nC.shape) != 2 or nC.shape[0] != nC.shape[1]:
        raise ValueError('nC must be a square matrix')

    m, n = nC.shape

    if not issparse(nC) or nC.dtype.kind != 'f':
        raise ValueError('nC must be a sparse float matrix')

    if not isinstance(ord, np.ndarray) or len(ord) != n:
        raise ValueError(f'ord must be a float vector with {n} elements')

    if not np.isscalar(beta) or not isinstance(beta, (float, np.floating)) or beta <= 0 or beta >= 1:
        raise ValueError('beta must be a scalar in the range (0,1)')

    # allocate space for sum_jc
    sum_jc = np.zeros(n)

    # allocate space for indicator vector c
    c = np.zeros(n, dtype=bool)

    # python_variable = CPP_variable/meaning
    # ir = pir/row indices | jc = pjc/col indices | pr = pr/data
    ir, jc, pr = find(nC)

    for current in ord:
        if sum_jc[current] <= beta:
            # add current to coarse
            c[current] = True

            # update sum_jc accordingly
            for jj in range(jc[current], jc[current + 1]):
                row = ir[jj]
                sum_jc[row] = sum_jc[row] + pr[jj]

    return c
