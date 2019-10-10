import numpy as np
import numpy.random as npr
from numpy.linalg import inv

N = 500
mu = 0
sigma = 1

a = npr.randn(N)*sigma+mu
a_mean = np.mean(a)
a_var = np.var(a, ddof=1)

N_bs = 10000
a_bs_mean = np.zeros(N_bs)
a_bs_var = np.zeros(N_bs)

for i in range(N_bs):
    a_bs = a[npr.choice(N, N)]
    a_bs_mean[i] = np.mean(a_bs)
    a_bs_var[i] = np.var(a_bs, ddof=1)

a_bs_mean_var = np.var(a_bs_mean, ddof=1)
a_bs_var_var = np.var(a_bs_var, ddof=1)

cov1 = np.zeros((2, 2))
cov1[0, 0] = N/sigma**2
cov1[0, 1] = np.sum((a-mu)/sigma**4)
cov1[1, 0] = np.sum((a-mu)/sigma**4)
cov1[1, 1] = np.sum((a-mu)**2/sigma**6-1/sigma**4/2)

print(cov1)

icov1 = inv(cov1)

print('mean and variance: ', a_mean, a_var)
print('error on mean and variance (bootstrap): ', a_bs_mean_var, a_bs_var_var)
print('error on mean and variance (theory): ', (sigma**2)/N, 2*(sigma**2)**2/N)
print('error on mean and variance (covariance with true mean and variance): ', icov1[0, 0], icov1[1, 1])
print('error on mean and variance (covaraince with sample mean and variance): ', a_var/N, 2*(a_var)**2/N)
