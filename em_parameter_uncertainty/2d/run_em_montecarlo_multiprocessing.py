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
base_seed = 0

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

base_seed = 50000

mu1_true = np.array([1, -2])
Sigma1_true = np.array([[1.2**2, -0.4], [-0.4, 0.8**2]])
N1_true = 150

mu2_true = np.array([-1, 0])
Sigma2_true = np.array([[0.9**2, -0.1], [-0.1, 1.5**2]])
N2_true = 50

mu3_true = np.array([1.5, 1.5])
Sigma3_true = np.array([[0.7**2, 0.45], [0.45 ,1**2]])
N3_true = 100

N = N1_true+N2_true+N3_true
pi1_true = N1_true/N
pi2_true = N2_true/N
pi3_true = N3_true/N
pi_true = np.array([pi1_true, pi2_true, pi3_true])

C = 3
rtol = 1e-4
max_iter = 1000
restarts = 20

def generate_results(i_trial):
    npr.seed(base_seed+i_trial)

    data1 = multivariate_normal.rvs(mean=mu1_true, cov=Sigma1_true, size=N1_true)
    data2 = multivariate_normal.rvs(mean=mu2_true, cov=Sigma2_true, size=N2_true)
    data3 = multivariate_normal.rvs(mean=mu3_true, cov=Sigma3_true, size=N3_true)
    data = np.concatenate((data1, data2, data3))

    mu1_sample = np.mean(data1, axis=0)
    mu2_sample = np.mean(data2, axis=0)
    mu3_sample = np.mean(data3, axis=0)
    Sigma1_sample = np.cov(data1.T, ddof=1)
    Sigma2_sample = np.cov(data2.T, ddof=1)
    Sigma3_sample = np.cov(data3.T, ddof=1)

    mu_sample = np.array([mu1_sample, mu2_sample, mu3_sample])
    Sigma_sample = np.array([Sigma1_sample, Sigma2_sample, Sigma3_sample])

    vlb_best, pi_em, mu_em, Sigma_em \
    = calc_gaussian_mixture(data, C, rtol=rtol, max_iter=max_iter, restarts=restarts)

    return mu_sample, Sigma_sample, pi_em, mu_em, Sigma_em

time1 = time.time()

n_trial = 10000

pool = Pool()
outputs = list(tqdm(pool.imap(generate_results, np.arange(n_trial, dtype=np.int32)), total=n_trial))
pool.close()
pool.join()
outputs = list(zip(*outputs))
mu_sample = np.array(outputs[0])
Sigma_sample = np.array(outputs[1])
pi_em = np.array(outputs[2])
mu_em = np.array(outputs[3])
Sigma_em = np.array(outputs[4])

time2 = time.time()

print(time2-time1)

np.save('mu_sample', mu_sample)
np.save('Sigma_sample', Sigma_sample)
np.save('pi_em', pi_em)
np.save('mu_em', mu_em)
np.save('Sigma_em', Sigma_em)
