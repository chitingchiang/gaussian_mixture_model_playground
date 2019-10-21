import numpy as np
import numpy.random as npr
from scipy.stats import multivariate_normal

def E_step(X, pi, mu, Sigma):
    """
    Performs E-step on GMM model
    Inputs:
    X: (N, d), data points
    pi: (C), mixture component weights 
    mu: (C, d), mixture component means
    Sigma: (C, d, d), mixture component covariance matrices
    
    Return:
    q: (N, C), probabilities of clusters for objects
    """
    N = X.shape[0] # number of objects
    C = pi.shape[0] # number of clusters

    q = np.zeros((N, C))
    for c in range(C):
        q[:, c] = multivariate_normal.pdf(x=X, mean=mu[c, :], cov=Sigma[c, :, :])*pi[c]
    q = np.transpose(q.T/np.sum(q, axis=1))

    return q

def M_step(X, q):
    """
    Performs M-step on GMM model
    Inputs:
    X: (N, d), data points
    q: (N, C), distribution q(T)  
    
    Returns:
    pi: (C)
    mu: (C, d)
    Sigma: (C, d, d)
    """
    C = q.shape[1] # number of clusters
    d = X.shape[1] # dimension of each object

    mu = np.zeros((C, d))
    for c in range(C):
        mu[c, :] = np.sum((X.T)*q[:, c], axis=1)/np.sum(q[:, c])
        
    Sigma = np.zeros((C, d, d))
    for c in range(C):
        diff = X-mu[c, :]
        Sigma[c, :, :] = np.dot(np.sum(np.einsum('ij, ik->ijk', diff, diff).T*q[:, c], axis=-1),
                                np.eye(d)/np.sum(q[:, c]))
    
    pi = np.sum(q, axis=0)
    pi = pi/np.sum(pi)

    return pi, mu, Sigma

def compute_vlb(X, pi, mu, Sigma, q):
    """
    Inputs:
    X: (N, d), data points
    q: (N, C), distribution q(T)  
    pi: (C)
    mu: (C, d)
    Sigma: (C, d, d)
    
    Return:
    vlb: scalar, variational lower bound
    """
    C = q.shape[1] # number of clusters

    vlb = 0
    for c in range(C):
        try:
            vlb += np.sum(q[:, c]*(np.log(pi[c])+multivariate_normal.logpdf(x=X, mean=mu[c, :], cov=Sigma[c, :, :])
                                  -np.log(q[:, c])))
        except:
            vlb = -1e10
            break
    return vlb

def calc_gaussian_mixture(X, C, rtol=1e-3, max_iter=100, restarts=10):
    '''
    Starts with random initialization *restarts* times
    Runs optimization until saturation with *rtol* reached
    or *max_iter* iterations were made.

    Inputs:
    X: (N, d), data points
    C: int, number of clusters

    Returns:
    vlb_best: scalar, maximized variational lower bound
    pi_best: (C), optimized prior
    mu_best: (C, d), optimized mean
    Sigma_best: (C, d, d), optimized covariance matrix
    '''
    d = X.shape[1] # dimension of each object
    
    vlb_best = -1e10
    pi_best = np.ones(C)/C
    mu_best = np.random.rand(C, d)
    Sigma_best = np.zeros((C, d, d))

    for _ in range(restarts):
        try:
            # random initialization of the parameters
            vlb_last = -1e10
            pi = np.ones(C)/C
            mu = np.random.rand(C, d)
            Sigma = np.zeros((C, d, d))
            for c in range(C):
                var = npr.rand(d)+0.5
                corr = npr.rand(d, d)-0.5
                corr = 0.5*(corr+corr.T)
                for i in range(d):
                    for j in range(d):
                        if i==j:
                            Sigma[c, i, j] = var[i]
                        else:
                            Sigma[c, i, j] = corr[i, j]*np.sqrt(var[i]*var[j])

            # main loop
            for _ in range(max_iter):
                q = E_step(X, pi, mu, Sigma)
                pi, mu, Sigma = M_step(X, q)
                vlb = compute_vlb(X, pi, mu, Sigma, q)
                if np.abs((vlb-vlb_last)/vlb_last)<=rtol:
                    break
                vlb_last = vlb

            # record the best result
            if vlb>vlb_best:
                vlb_best = vlb
                pi_best = pi
                mu_best = mu
                Sigma_best = Sigma

        except np.linalg.LinAlgError:
#            print("Singular matrix: components collapsed")
            pass

    # order data by mu_y
    order = np.argsort(mu_best[:, 1])
    pi_best = pi_best[order]
    mu_best = mu_best[order]
    Sigma_best = Sigma_best[order]

    return vlb_best, pi_best, mu_best, Sigma_best
