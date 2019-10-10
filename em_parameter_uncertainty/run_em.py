import numpy as np
import numpy.random as npr
from tqdm import tqdm
import matplotlib.pyplot as pt
from em_1d import calc_gaussian_mixture

npr.seed(5678)

'''
Set up the parameters (mu and sigma) for two Gaussian distributions
and generate N1_true and N2_true data points for the two distributions.
'''
mu1_true = 1
sigma1_true = 0.5
N1_true = 100
data1_true = npr.randn(N1_true)*sigma1_true+mu1_true

mu2_true = 10
sigma2_true = 0.4
N2_true = 100
data2_true = npr.randn(N2_true)*sigma2_true+mu2_true

'''
Concatenate the two sets of data points into one and save them to
numpy arrays for future use.
'''

data = np.concatenate((data1_true, data2_true))

np.save('data1_true', data1_true)
np.save('data2_true', data2_true)

'''
Compute the true priors as pi1_true = N1_true/N and pi2_true = N2_true/N.
'''

N = N1_true+N2_true
pi1_true = N1_true/N
pi2_true = N2_true/N


'''
Visualize the data by computing the histograms.

x_min = -2
x_max = 5
n_bin = 20

hist1_true, bins = np.histogram(data1_true, bins=n_bin, range=(x_min, x_max))
hist2_true, bins = np.histogram(data2_true, bins=n_bin, range=(x_min, x_max))
hist, bins = np.histogram(data, bins=n_bin, range=(x_min, x_max))

bins = 0.5*(bins[:-1]+bins[1:])

pt.figure(1)
pt.plot(bins, hist1, 'r-')
pt.plot(bins, hist2, 'b-')
pt.plot(bins, hist, 'k--')
pt.show()
'''

'''
Set up the parameters for running the EM optimization for the Gaussian
mixture model and save the results (pi_best, mu_best, sigma_best) to
numpy arrays.
'''

C = 2
rtol = 1e-4
max_iter = 1000
restarts = 20

vlb_best, pi_best, mu_best, sigma_best = calc_gaussian_mixture(data, C, rtol=rtol, max_iter=max_iter, restarts=restarts)

np.save('pi_best', pi_best)
np.save('mu_best', mu_best)
np.save('sigma_best', sigma_best)

'''
Run the bootstrap resampling N_bs times and recored the results from
EM optimization to access the uncertainty of the model parameters
(pi, mu, sigma). Save the results to numpy arrays for later analysis.
'''

N_bs = 5000
vlb_bs = np.zeros(N_bs)
pi_bs = np.zeros((N_bs, C))
mu_bs = np.zeros((N_bs, C))
sigma_bs = np.zeros((N_bs, C))

for i in tqdm(range(N_bs)):
    X_bs = data[npr.choice(N, N)]
    vlb_bs[i], pi_bs[i, :], mu_bs[i, :], sigma_bs[i, :] = calc_gaussian_mixture(X_bs, C, rtol=rtol, max_iter=max_iter, restarts=restarts)

np.save('pi_bs', pi_bs)
np.save('mu_bs', mu_bs)
np.save('sigma_bs', sigma_bs)
