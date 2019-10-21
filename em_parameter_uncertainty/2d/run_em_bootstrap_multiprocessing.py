import numpy as np
import numpy.random as npr
from scipy.stats import multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as pt
from em_2d import calc_gaussian_mixture
from multiprocessing import Pool
import time

'''
Set up the parameters (mu and sigma) for two Gaussian distributions
and generate N1_true and N2_true data points for the two distributions.
'''

'''
npr.seed(1234567)

mu1_true = np.array([2, -3])
Sigma1_true = np.array([[0.64, -0.2], [-0.2, 0.25]])
N1_true = 120

mu2_true = np.array([-2, 0])
Sigma2_true = np.array([[0.36, -0.1], [-0.1, 1]])
N2_true = 60

mu3_true = np.array([2, 2])
Sigma3_true = np.array([[0.49, 0.36], [0.36 ,0.64]])
N3_true = 120
'''

npr.seed(7654321)

mu1_true = np.array([1, -2])
Sigma1_true = np.array([[1.2**2, -0.4], [-0.4, 0.8**2]])
N1_true = 150

mu2_true = np.array([-1, 0])
Sigma2_true = np.array([[0.9**2, -0.1], [-0.1, 1.5**2]])
N2_true = 50

mu3_true = np.array([1.5, 1.5])
Sigma3_true = np.array([[0.7**2, 0.45], [0.45 ,1**2]])
N3_true = 100

data1 = multivariate_normal.rvs(mean=mu1_true, cov=Sigma1_true, size=N1_true)
data2 = multivariate_normal.rvs(mean=mu2_true, cov=Sigma2_true, size=N2_true)
data3 = multivariate_normal.rvs(mean=mu3_true, cov=Sigma3_true, size=N3_true)
data = np.concatenate((data1, data2, data3))

np.save('data1', data1)
np.save('data2', data2)
np.save('data3', data3)

N = N1_true+N2_true+N3_true
pi1_true = N1_true/N
pi2_true = N2_true/N
pi3_true = N3_true/N
pi_true = np.array([pi1_true, pi2_true, pi3_true])

C = 3
rtol = 1e-4
max_iter = 1000
restarts = 20
base_seed = 0

def generate_results(i_trial):
    npr.seed(base_seed+i_trial)
    data_bs = data[npr.choice(N, N)]

    vlb_best, pi_bs, mu_bs, Sigma_bs \
    = calc_gaussian_mixture(data_bs, C, rtol=rtol, max_iter=max_iter, restarts=restarts)

    return pi_bs, mu_bs, Sigma_bs

time1 = time.time()

n_trial = 10000

pool = Pool()
outputs = list(tqdm(pool.imap(generate_results, np.arange(n_trial, dtype=np.int32)), total=n_trial))
pool.close()
pool.join()
outputs = list(zip(*outputs))
pi_bs = np.array(outputs[0])
mu_bs = np.array(outputs[1])
Sigma_bs = np.array(outputs[2])

time2 = time.time()

print(time2-time1)

np.save('pi_bs', pi_bs)
np.save('mu_bs', mu_bs)
np.save('Sigma_bs', Sigma_bs)
