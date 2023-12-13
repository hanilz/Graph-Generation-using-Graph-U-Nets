# clear all
# a = importdata('cora.txt');
# n = max(max(a))+ 1;
# A = zeros(n,n);
# for i = 1: size(a,1)
#     A(a(i,1)+1,a(i,2)+1)=1;
# end
# A = remove_zero_row(A,n);
# [P R W c] = AMG(sparse(A), 0.2, 5);
#
# epsilon=0.5;
# MinPts=10;
# IDX=DBSCAN(full(A),epsilon,MinPts);
# edges = supernode(A,c);
# save('./data/cora')

import numpy as np
from scipy.sparse import spdiags, csr_matrix, find, issparse, coo_matrix
from scipy.sparse.linalg import spilu
from scipy.io import savemat
from sklearn.cluster import DBSCAN


def supernode(A, c):
    edges = np.zeros(len(c), dtype=int)
    for i, current_c in enumerate(c):
        F = A.copy()
        n = F.shape[0]
        m = len(current_c)
        E = np.zeros((m, n), dtype=int)
        prev = 0
        for j in range(m):
            later = current_c[j] + 1
            for k in range(prev, later):
                E[j, :] += F[k, :]
            prev = later
        F = np.zeros((m, m), dtype=int)
        prev = 0
        for j in range(m):
            later = current_c[j] + 1
            for k in range(prev, later):
                F[:, j] += E[:, k]
            prev = later
        F = F != 0
        edges[i] = np.sum(F)
    return edges


def remove_zero_row(A, n):
    # Check if any columns are zero columns
    B = np.sum(A, axis=0) == 0
    indices = [i for i in range(n) if not B[i]]
    A = A[indices][:, indices]
    return A


def AMG(fW, beta, NS):
    def fine2coarse(W, beta):
        n = W.shape[0]
        nW = W.multiply(1.0 / np.sum(W, axis=1))
        # Select coarse nodes (using ChooseCoarseGreedy_mex)
        c = ChooseCoarseGreedy_mex(nW, np.random.permutation(n), beta)

        # Compute the interp matrix
        ci = np.where(c)[0]
        P = W[:, ci]
        P = P.multiply(1.0 / np.sum(P, axis=1))

        # Make sure coarse points are directly connected to their fine counterparts
        jj, ii, pji = find(P.transpose())
        # jj, ii, pji = coo_matrix(P.transpose())
        sel = ~c[ii]
        # P = csr_matrix((np.concatenate((pji[sel], np.ones(sum(c)))), (np.concatenate((jj[sel], np.arange(sum(c)))), np.concatenate((ii[sel], ci)))), shape=(len(c), len(ci)))

        # Define mycat function
        def mycat(x, y):
            return np.concatenate((np.ravel(x), np.ravel(y)))

        # Create sparse matrix P
        rows = mycat(jj[sel], np.arange(np.sum(c)))
        cols = mycat(ii[sel], ci)
        values = mycat(pji[sel], np.ones(np.sum(c)))

        # Calculate the size of the sparse matrix P
        num_rows = np.max(rows) + 1
        num_cols = np.max(cols) + 1

        # Create a sparse matrix in CSR format
        P = csr_matrix((values, (rows, cols)), shape=(num_rows, num_cols)).transpose()

        return c, P
    n = fW.shape[0]

    P = [None] * NS
    c = [None] * NS
    W = [None] * NS

    # Make sure diagonal is zero
    W[0] = (fW - spdiags(fW.diagonal(), 0, n, n))
    fine = np.arange(n)

    for si in range(NS):
        tmp_c, P[si] = fine2coarse(W[si], beta)
        c[si] = fine[tmp_c]

        if si < NS - 1:
            W[si + 1] = csr_matrix(P[si].transpose().dot(W[si]).dot(P[si]).transpose()).transpose()

            W[si + 1] = W[si + 1] - spdiags(W[si + 1].diagonal(), [0], W[si + 1].shape[0], W[si + 1].shape[1], format='csc')
            fine = c[si]

    # Restriction matrices
    def spmtimesd(x, weights):
        return csr_matrix(x.transpose() * weights)

    R = [None] * NS
    # Apply the function to each element of P using list comprehension
    # R = [spmtimesd(x, 1. / np.sum(x, axis=1)) for x in P]
    for i, x in enumerate(P):
        x_transpose = x.transpose()
        column_sum = np.sum(x, axis=0)
        inverse_sum = 1.0 / column_sum
        result = x_transpose.multiply(inverse_sum.transpose())
        result_sparse = csr_matrix(result.transpose()).transpose()
        R[i] = result_sparse
    # R = [csr_matrix((x.transpose()).multiply(1.0 / np.sum(x, axis=0))) for x in P]
    return P, R, W, c


def ChooseCoarseGreedy_mex(nC, ord, beta):
    n = nC.shape[0]

    # allocate space for sum_jc
    sum_jc = np.zeros(n)

    # allocate space for indicator vector c
    c = np.zeros(n, dtype=bool)

    for current in ord:
        if sum_jc[current] <= beta:
            # add current to coarse and update sum_jc accordingly
            c[current] = True
            sum_jc = sum_jc + nC.getcol(current).toarray().flatten()
    return c


def DBSCAN(X, epsilon, MinPts):
    C = 0
    n = X.shape[0]
    IDX = np.zeros(n, dtype=int)

    D = np.linalg.norm(X - X[:, np.newaxis], axis=-1)

    visited = np.zeros(n, dtype=bool)
    isnoise = np.zeros(n, dtype=bool)

    def RegionQuery(i):
        return np.where(D[i, :] <= epsilon)[0]

    def ExpandCluster(i, neighbors, C):
        IDX[i] = C

        k = 0
        while k < len(neighbors):
            j = neighbors[k]

            if not visited[j]:
                visited[j] = True
                neighbors2 = RegionQuery(j)
                if len(neighbors2) >= MinPts:
                    neighbors = np.concatenate((neighbors, neighbors2))
            if IDX[j] == 0:
                IDX[j] = C

            k += 1

    for i in range(n):
        if not visited[i]:
            visited[i] = True

            Neighbors = RegionQuery(i)
            if len(Neighbors) < MinPts:
                # X[i, :] is NOISE
                isnoise[i] = True
            else:
                C += 1
                ExpandCluster(i, Neighbors, C)

    return IDX, isnoise


if __name__ == '__main__':
    # input_file_path = r'C:\Users\Hanil\Coding\Miscgan\Dataset\email-Eu-core.txt'
    input_file_path = r'C:\Users\Hanil\Coding\Miscgan\Dataset\email-Eu-core2.txt'

    a = np.loadtxt(input_file_path, dtype=int)  # Read the edges from the text file
    n = np.max(a) + 1  # Find the maximum element in the entire array and add 1
    A = np.zeros((n, n), dtype=int)

    for i in range(a.shape[0]):
        # A[a[i, 0] + 1, a[i, 1] + 1] = 1
        A[a[i, 0], a[i, 1]] = 1
    A = remove_zero_row(A, n)

    # Convert the adjacency matrix to a sparse matrix
    A_sparse = csr_matrix(A.transpose()).transpose()
    # TODO: change NS=5
    P, R, W, c = AMG(fW=A_sparse, beta=0.2, NS=5)

    # Apply AMG clustering
    epsilon = 0.5
    MinPts = 10
    indices = DBSCAN(X=A, epsilon=epsilon, MinPts=MinPts)
    # labels = dbscan.fit_predict(pairwise_distances(A_sparse, metric='l2'))

    # Extract edges between supernodes
    edges = supernode(A, c)
    c = [[x + 1 for x in a] for a in c]
    # data = {'A': A, 'P': P, 'W': W, 'R': R, 'IDX': indices, 'edges': edges, 'c': c}
    data = {'A': A, 'P': P, 'W': W, 'R': R, 'IDX': indices, 'edges': edges, 'c': c}
    savemat(r'C:\Users\Hanil\Coding\Miscgan\data\email-python.mat', data)
