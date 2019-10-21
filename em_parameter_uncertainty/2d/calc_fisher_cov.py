import numpy as np
import numpy.random as npr
from scipy.stats import multivariate_normal
from numpy.linalg import inv, det
from calc_gaussians_2d import calc_gaussians_2d
import sys

def calc_fisher_cov(X, pi, mu, Sigma):
    '''
    Input:
    X: (N, C), data sample
    pi: (C), prior
    mu: (C, 2), mean
    Sigma: (C, 2, 2), covariance

    Return:
    fisher: (6*C, 6*C), Fisher matrix [pi_1, ..., pi_C, m1_1, m2_1, S1_1, S2_1, S12_1, ..., m1_C, m2_C, S1_C, S2_C, S12_C]
            (assuming sum of prior is one so one of pi is not free)
    '''

    if (X.shape[1]!=2) or (mu.shape[1]!=2) or (Sigma.shape[1]!=2) or (Sigma.shape[2]!=2):
        print('Error! The dimension of the input data is not 2!')
        sys.exit()

    N = X.shape[0]
    C = pi.shape[0]
    if (mu.shape[0]!=C) or (Sigma.shape[0]!=C):
        print('Error! The input parameters do not agree with three clusters!')
        sys.exit()

    G = []
    dG = []
    ddG = []
    for c in range(C):
        G_temp, dG_temp, ddG_temp = calc_gaussians_2d(X, mu[c], Sigma[c])

        G.append(G_temp)
        dG.append(dG_temp)
        ddG.append(ddG_temp)

    G = np.array(G)
    dG = np.array(dG)
    ddG = np.array(ddG)

    #print(G.shape, dG.shape, ddG.shape)

    fisher = np.zeros((6*C, 6*C))
    # pi_1, ..., pi_C, (m1_1, m2_1, S1_1, S2_1, S12_1), ..., (m1_C, m2_C, S1_C, S2_C, S12_C)

    M = np.einsum('i, ij->j', pi, G)

    for c1 in range(C):
        # dpi_{c1} dpi_{c}
        fisher[c1, c1:C] = np.sum(G[c1]*G[c1:C]/M**2, axis=1)
        for c2 in range(C):
            c = C+c2*5
            if c1==c2:
                # dpi_{c1} dp_{c1, a}
                fisher[c1, c:c+5] = np.sum((pi[c2]*dG[c2]*G[c1]-dG[c2]*M)/M**2, axis=1)
            else:
                # dpi_{c1} dp_{c2, a}
                fisher[c1, c:c+5] = np.sum(pi[c2]*dG[c2]*G[c1]/M**2, axis=1)

        for c2 in range(c1, C):
            for a in range(5):
                c_a = C+c1*5+a
                if c1==c2:
                    # dp_{c1, a} dp_{c1, b}
                    fisher[c_a, c_a:C+(c1+1)*5] = np.sum((pi[c1]**2*dG[c1, a]*dG[c1, a:]-pi[c1]*ddG[c1, a, a:]*M)/M**2, axis=1)
                else:
                    # dp_{c1, a} dp_{c2, b}
                    c_b = C+c2*5
                    fisher[c_a, c_b:c_b+5] = np.sum(pi[c1]*pi[c2]*dG[c1, a]*dG[c2]/M**2, axis=1)

    #np.set_printoptions(precision=2, linewidth=200)
    #print(fisher[C:, C:])

    fisher = fisher+fisher.T-np.diag(np.diagonal(fisher))
    cov = inv(fisher)

    return fisher, cov
