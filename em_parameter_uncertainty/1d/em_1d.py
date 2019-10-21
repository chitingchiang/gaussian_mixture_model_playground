import numpy as np
import numpy.random as npr
from scipy.stats import multivariate_normal

def E_step(X, pi, mu, Sigma):
    '''
    Performs E-step on GMM model

    Inputs are numpy arrays:
    X: (N), data points
    pi: (C), mixture component weights 
    mu: (C), mixture component means
    Sigma: (C), mixture component covariance matrices
    
    Returns:
    q: (N x C), probabilities of clusters for objects
    '''

    N = X.shape[0]
    C = mu.shape[0]

    q = np.zeros((N, C))
    for c in range(C):
        q[:, c] = multivariate_normal.pdf(x=X, mean=mu[c], cov=Sigma[c])*pi[c]
    q = np.transpose(q.T/np.sum(q, axis=1))
    
    return q

def M_step(X, q):
    '''
    Performs M-step on GMM model

    Inputs are numpy arrays:
    X: (N), data points
    q: (N x C), distribution q(T)  
    
    Returns:
    pi: (C)
    mu: (C)
    Sigma: (C)
    '''

    N = X.shape[0]
    C = q.shape[1]

    mu = np.zeros(C)
    for c in range(C):
        mu[c] = np.sum(X*q[:, c])/np.sum(q[:, c])

    Sigma = np.zeros(C)
    for c in range(C):
        Sigma[c] = np.sum(q[:, c]*((X-mu[c])**2))/np.sum(q[:, c])
    
    pi = np.sum(q, axis=0)
    pi = pi/np.sum(pi)

    return pi, mu, Sigma

def compute_vlb(X, pi, mu, Sigma, q):
    '''
    Compute the variational lower bound (negative log KL divergence)

    Inputs are numpy arrays:
    X: (N), data points
    q: (N x C), distribution q(T)  
    pi: (C)
    mu: (C)
    Sigma: (C)
    
    Return:
    vlb: float scalar
    '''
    C = q.shape[1] # number of clusters

    vlb = 0
    for c in range(C):
        vlb = vlb + np.sum(q[:, c]*(np.log(pi[c])+multivariate_normal.logpdf(x=X, mean=mu[c], cov=Sigma[c])-np.log(q[:, c])))
    return vlb

def calc_gaussian_mixture(X, C, rtol=1e-3, max_iter=100, restarts=10):#, seed=5432):
    '''
    Starts with random initialization *restarts* times
    Runs optimization (maximization of variational lower bound)
    until saturation with *rtol* reached or *max_iter*
    iterations were made.
    
    Inputs are numpy arrays:
    X: (N), data points
    C: integer scalar, number of clusters

    Returns:
    vlb_best: float scalar, maximized variational lower bound
    pi_best: (C), optimized prior
    mu_best: (C), optimized mu
    sigma_best: (C), potimized sigma
    '''

    #npr.seed(seed)

    vlb_best = -1e10
    pi_best = np.zeros(C)
    mu_best = np.zeros(C)
    Sigma_best = np.zeros(C)

    for _ in range(restarts):
        try:
            # random initialization of the parameters
            vlb_last = 0
            pi = np.ones(C)/C
            mu = npr.rand(C)
            Sigma = np.rand(C)+0.5

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
                pi_best = pi[:]
                mu_best = mu[:]
                Sigma_best = Sigma[:]

        except np.linalg.LinAlgError:
            print("Singular matrix: components collapsed")
            pass

    order = np.argsort(mu_best)
    pi_best = pi_best[order]
    mu_best = mu_best[order]
    sigma_best = np.sqrt(Sigma_best[order])

    return vlb_best, pi_best, mu_best, sigma_best
