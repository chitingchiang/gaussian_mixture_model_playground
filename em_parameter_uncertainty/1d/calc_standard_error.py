import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import inv

'''
Compare the best-fit and the mean of the bootstrap resampled data.
'''

pi_best = np.load('pi_best.npy')
mu_best = np.load('mu_best.npy')
sigma_best = np.load('sigma_best.npy')
Sigma_best = sigma_best**2

print('best pi:', pi_best)
print('best mu:', mu_best)
print('best sigma:', sigma_best)

pi_bs = np.load('pi_bs.npy')
mu_bs = np.load('mu_bs.npy')
sigma_bs = np.load('sigma_bs.npy')
Sigma_bs = sigma_bs**2

print('bootstrap mean pi:', np.mean(pi_bs, axis=0))
print('bootstrap mean mu:', np.mean(mu_bs, axis=0))
print('bootstrap mean sigma:', np.mean(sigma_bs, axis=0))

'''
Load the data to compute the Fisher information matrix.
'''

X1 = np.load('data1_true.npy')
X2 = np.load('data2_true.npy')
X = np.concatenate((X1, X2))

'''
The total data probability of the Gaussian mixture model is

P(X) = \prod_{i=1}^{N} \sum_{c=1}^{C} pi_c N(X_i; mu_c, Sigma_c)

where pi_c is the prior and N is the Gaussian distribution.

The parameters of the Gaussian mixture model are pi_c, mu_c, and Sigma_c = sigma_c^2.

The Fisher information matrix is given by F_{ij} = - d^2 ln P(x) / d p_i d p_j ,

where p_i and p_j are parameters of interests.

For our example we consider two clusters, so C=2 and naively the parameters

are (pi_1, mu_1, Sigma_1, pi_2, mu_2, Sigma_2). However, the priors have

the constraints that pi_1 + pi_2 = 1, so in practice we have only five

parameters (pi_1, mu_1, Sigma_1, mu_2, Sigma_2)!

To proceed we let

M_i = \sum_{c=1}^{C} pi_c N(X_i; mu_c, Sigma_c)

    = pi_1 N(X_i; mu_1, Sigma_1) + (1-pi_1) N(X_i; mu_2, Sigma_2) ,

and so

ln P(X) = \sum_{i=1}^N ln M_i.

With this, we can then compute the second derivatives straightforwardly

(and painfully...).

'''

G1 = multivariate_normal.pdf(x=X, mean=mu_best[0], cov=Sigma_best[0])
G2 = multivariate_normal.pdf(x=X, mean=mu_best[1], cov=Sigma_best[1])
diff1 = X-mu_best[0]
diff2 = X-mu_best[1]
diff1_norm = (X-mu_best[0])/sigma_best[0]
diff2_norm = (X-mu_best[1])/sigma_best[1]
dG1dm1 = G1*diff1/Sigma_best[0]
dG2dm2 = G2*diff2/Sigma_best[1]
dG1dS1 = G1/2/Sigma_best[0]*(diff1_norm**2-1)
dG2dS2 = G2/2/Sigma_best[1]*(diff2_norm**2-1)
ddG1dm1dm1 = G1/Sigma_best[0]*(diff1_norm**2-1)
ddG2dm2dm2 = G2/Sigma_best[1]*(diff2_norm**2-1)
ddG1dS1dS1 = G1/4/Sigma_best[0]**2*(diff1_norm**4-6*diff1_norm**2+3)
ddG2dS2dS2 = G2/4/Sigma_best[1]**2*(diff2_norm**4-6*diff2_norm**2+3)
ddG1dm1dS1 = G1*diff1/2/Sigma_best[0]**2*(diff1_norm**2-3)
ddG2dm2dS2 = G2*diff2/2/Sigma_best[1]**2*(diff2_norm**2-3)

M = pi_best[0]*G1+pi_best[1]*G2
dMdpi = G1-G2
dMdm1 = pi_best[0]*dG1dm1
dMdS1 = pi_best[0]*dG1dS1
dMdm2 = pi_best[1]*dG2dm2
dMdS2 = pi_best[1]*dG2dS2
ddMdpidpi = 0
ddMdpidm1 = dG1dm1
ddMdpidS1 = dG1dS1
ddMdpidm2 = -dG2dm2
ddMdpidS2 = -dG2dm2
ddMdm1dm1 = pi_best[0]*ddG1dm1dm1
ddMdm1dS1 = pi_best[0]*ddG1dm1dS1
ddMdm1dm2 = 0
ddMdm1dS2 = 0
ddMdS1dS1 = pi_best[0]*ddG1dS1dS1
ddMdS1dm2 = 0
ddMdS1dS2 = 0
ddMdm2dm2 = pi_best[1]*ddG2dm2dm2
ddMdm2dS2 = pi_best[1]*ddG2dm2dS2
ddMdS2dS2 = pi_best[1]*ddG2dS2dS2

fisher = np.zeros((5, 5))
fisher[0, 0] = np.sum((dMdpi*dMdpi-ddMdpidpi*M)/M**2)
fisher[0, 1] = np.sum((dMdpi*dMdm1-ddMdpidm1*M)/M**2)
fisher[0, 2] = np.sum((dMdpi*dMdS1-ddMdpidS1*M)/M**2)
fisher[0, 3] = np.sum((dMdpi*dMdm2-ddMdpidm2*M)/M**2)
fisher[0, 4] = np.sum((dMdpi*dMdS2-ddMdpidS2*M)/M**2)
fisher[1, 1] = np.sum((dMdm1*dMdm1-ddMdm1dm1*M)/M**2)
fisher[1, 2] = np.sum((dMdm1*dMdS1-ddMdm1dS1*M)/M**2)
fisher[1, 3] = np.sum((dMdm1*dMdm2-ddMdm1dm2*M)/M**2)
fisher[1, 4] = np.sum((dMdm1*dMdS2-ddMdm1dS2*M)/M**2)
fisher[2, 2] = np.sum((dMdS1*dMdS1-ddMdS1dS1*M)/M**2)
fisher[2, 3] = np.sum((dMdS1*dMdm2-ddMdS1dm2*M)/M**2)
fisher[2, 4] = np.sum((dMdS1*dMdS2-ddMdS1dS2*M)/M**2)
fisher[3, 3] = np.sum((dMdm2*dMdm2-ddMdm2dm2*M)/M**2)
fisher[3, 4] = np.sum((dMdm2*dMdS2-ddMdm2dS2*M)/M**2)
fisher[4, 4] = np.sum((dMdS2*dMdS2-ddMdS2dS2*M)/M**2)

'''
Symmetrize the Fisher matrix.
'''

fisher = fisher+fisher.T-np.diag(np.diagonal(fisher))


'''
Check that the off-diagonal elements of the normalized Fisher matrix
are between -1 and 1.

fisher_check = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        fisher_check[i, j] = fisher[i, j]/np.sqrt(fisher[i, i]*fisher[j, j])

np.set_printoptions(linewidth=200)
print(fisher_check)
'''

'''
Invert the Fisher matrix to get the covariance matrix, and the diagonal
elements are the expected constraints on the parameters. Compare the
parameter uncertainties between Fisher matrix calculation and bootstrap
resampling.
'''

ifisher = inv(fisher)

var = np.diagonal(ifisher)
print('[fisher standard error]    pi: %.4e, mu1: %.4e, Sigma1: %.4e, mu2: %.4e, Sigma2: %.4e'%(var[0], var[1], var[2], var[3], var[4]))

pi_bs_var = np.var(pi_bs, axis=0, ddof=1)
mu_bs_var = np.var(mu_bs, axis=0, ddof=1)
Sigma_bs_var = np.var(Sigma_bs, axis=0, ddof=1)

print('[bootstrap standard error] pi: %.4e, mu1: %.4e, Sigma1: %.4e, mu2: %.4e, Sigma2: %.4e'%(pi_bs_var[0], mu_bs_var[0], Sigma_bs_var[0], mu_bs_var[1], Sigma_bs_var[1]))
