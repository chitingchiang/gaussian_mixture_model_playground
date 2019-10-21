import numpy as np
import numpy.random as npr
from scipy.stats import multivariate_normal
import matplotlib.pyplot as pt
import matplotlib.style
matplotlib.style.use('classic')
import sys
sys.path.insert(1, '/gpfs02/astro/workarea/cchiang/codes/expectation_maximization/2d')
from calc_fisher_cov import calc_fisher_cov

if __name__=='__main__':

    seed = np.int32(input('enter seed for generating data: '))

    mu1_true = np.array([2, -3])
    Sigma1_true = np.array([[0.64, -0.2], [-0.2, 0.25]])
    N1_true = 120

    mu2_true = np.array([-2, 0])
    Sigma2_true = np.array([[0.36, -0.1], [-0.1, 1]])
    N2_true = 60

    mu3_true = np.array([2, 2])
    Sigma3_true = np.array([[0.49, 0.36], [0.36 ,0.64]])
    N3_true = 120

    N = N1_true+N2_true+N3_true
    pi1_true = N1_true/N
    pi2_true = N2_true/N
    pi3_true = N3_true/N

    npr.seed(seed)
    data1 = multivariate_normal.rvs(mean=mu1_true, cov=Sigma1_true, size=N1_true)
    data2 = multivariate_normal.rvs(mean=mu2_true, cov=Sigma2_true, size=N2_true)
    data3 = multivariate_normal.rvs(mean=mu3_true, cov=Sigma3_true, size=N3_true)
    data = np.concatenate((data1, data2, data3))

    pi_em = np.load('pi_em.npy')
    mu_em = np.load('mu_em.npy')
    Sigma_em = np.load('Sigma_em.npy')

    mu_sample = np.load('mu_sample.npy')
    Sigma_sample = np.load('Sigma_sample.npy')

    pi_em_var = np.var(pi_em, axis=0)
    mu_em_var = np.var(mu_em, axis=0)
    Sigma_em_var = np.var(Sigma_em, axis=0)

    var_measure = pi_em_var[:-1]
    for c in range(3):
        var_measure = np.append(var_measure,
                               [mu_em_var[c, 0], mu_em_var[c, 1], Sigma_em_var[c, 0, 0], Sigma_em_var[c, 1, 1], Sigma_em_var[c, 0, 1]])


    fisher, cov = calc_fisher_cov(data, np.mean(pi_em, axis=0), np.mean(mu_em, axis=0), np.mean(Sigma_em, axis=0))
    var_theory = np.diagonal(cov)

    print('measure, theory')
    for i in range(17):
        if i<2:
            print('%.8e %.8e'%(var_measure[i], var_theory[i]))
        else:
            print('%.8e %.8e'%(var_measure[i], var_theory[i+1]))

    pi_1_em_hist, pi_1_em_bin = np.histogram(pi_em[:, 0], bins=20, density=True)
    mx_1_em_hist, mx_1_em_bin = np.histogram(mu_em[:, 0, 0], bins=100, density=True)
    my_1_em_hist, my_1_em_bin = np.histogram(mu_em[:, 0, 1], bins=100, density=True)
    Sxx_1_em_hist, Sxx_1_em_bin = np.histogram(Sigma_em[:, 0, 0, 0], bins=100, density=True)
    Syy_1_em_hist, Syy_1_em_bin = np.histogram(Sigma_em[:, 0, 1, 1], bins=100, density=True)
    Sxy_1_em_hist, Sxy_1_em_bin = np.histogram(Sigma_em[:, 0, 0, 1], bins=100, density=True)

    pi_2_em_hist, pi_2_em_bin = np.histogram(pi_em[:, 1], bins=20, density=True)
    mx_2_em_hist, mx_2_em_bin = np.histogram(mu_em[:, 1, 0], bins=100, density=True)
    my_2_em_hist, my_2_em_bin = np.histogram(mu_em[:, 1, 1], bins=100, density=True)
    Sxx_2_em_hist, Sxx_2_em_bin = np.histogram(Sigma_em[:, 1, 0, 0], bins=100, density=True)
    Syy_2_em_hist, Syy_2_em_bin = np.histogram(Sigma_em[:, 1, 1, 1], bins=100, density=True)
    Sxy_2_em_hist, Sxy_2_em_bin = np.histogram(Sigma_em[:, 1, 0, 1], bins=100, density=True)

    pi_3_em_hist, pi_3_em_bin = np.histogram(pi_em[:, 2], bins=20, density=True)
    mx_3_em_hist, mx_3_em_bin = np.histogram(mu_em[:, 2, 0], bins=100, density=True)
    my_3_em_hist, my_3_em_bin = np.histogram(mu_em[:, 2, 1], bins=100, density=True)
    Sxx_3_em_hist, Sxx_3_em_bin = np.histogram(Sigma_em[:, 2, 0, 0], bins=100, density=True)
    Syy_3_em_hist, Syy_3_em_bin = np.histogram(Sigma_em[:, 2, 1, 1], bins=100, density=True)
    Sxy_3_em_hist, Sxy_3_em_bin = np.histogram(Sigma_em[:, 2, 0, 1], bins=100, density=True)

    pi_1_em_bin = 0.5*(pi_1_em_bin[:-1]+pi_1_em_bin[1:])
    pi_2_em_bin = 0.5*(pi_2_em_bin[:-1]+pi_2_em_bin[1:])
    pi_3_em_bin = 0.5*(pi_3_em_bin[:-1]+pi_3_em_bin[1:])

    mx_1_em_bin = 0.5*(mx_1_em_bin[:-1]+mx_1_em_bin[1:])
    mx_2_em_bin = 0.5*(mx_2_em_bin[:-1]+mx_2_em_bin[1:])
    mx_3_em_bin = 0.5*(mx_3_em_bin[:-1]+mx_3_em_bin[1:])

    my_1_em_bin = 0.5*(my_1_em_bin[:-1]+my_1_em_bin[1:])
    my_2_em_bin = 0.5*(my_2_em_bin[:-1]+my_2_em_bin[1:])
    my_3_em_bin = 0.5*(my_3_em_bin[:-1]+my_3_em_bin[1:])

    Sxx_1_em_bin = 0.5*(Sxx_1_em_bin[:-1]+Sxx_1_em_bin[1:])
    Sxx_2_em_bin = 0.5*(Sxx_2_em_bin[:-1]+Sxx_2_em_bin[1:])
    Sxx_3_em_bin = 0.5*(Sxx_3_em_bin[:-1]+Sxx_3_em_bin[1:])

    Syy_1_em_bin = 0.5*(Syy_1_em_bin[:-1]+Syy_1_em_bin[1:])
    Syy_2_em_bin = 0.5*(Syy_2_em_bin[:-1]+Syy_2_em_bin[1:])
    Syy_3_em_bin = 0.5*(Syy_3_em_bin[:-1]+Syy_3_em_bin[1:])

    Sxy_1_em_bin = 0.5*(Sxy_1_em_bin[:-1]+Sxy_1_em_bin[1:])
    Sxy_2_em_bin = 0.5*(Sxy_2_em_bin[:-1]+Sxy_2_em_bin[1:])
    Sxy_3_em_bin = 0.5*(Sxy_3_em_bin[:-1]+Sxy_3_em_bin[1:])

    pt.figure(1, figsize=(14, 8))
    pt.subplot(2, 3, 1)
    pt.xlabel(r'$\pi$')
    pt.plot(pi_1_em_bin, pi_1_em_hist, 'r-', label=r'${\rm cluster\ 1}$')
    pt.plot(pi_2_em_bin, pi_2_em_hist, 'g-', label=r'${\rm cluster\ 2}$')
    pt.plot(pi_3_em_bin, pi_3_em_hist, 'c-', label=r'${\rm cluster\ 3}$')
    pt.plot(pi_1_em_bin, multivariate_normal.pdf(x=pi_1_em_bin, mean=pi1_true, cov=pi_em_var[0]), 'k--')
    pt.plot(pi_2_em_bin, multivariate_normal.pdf(x=pi_2_em_bin, mean=pi2_true, cov=pi_em_var[1]), 'k--')
    pt.plot(pi_3_em_bin, multivariate_normal.pdf(x=pi_3_em_bin, mean=pi3_true, cov=pi_em_var[2]), 'k--')
    pt.subplot(2, 3, 2)
    pt.xlabel(r'$\mu_x$')
    pt.plot(mx_1_em_bin, mx_1_em_hist, 'r-', label=r'${\rm cluster\ 1}$')
    pt.plot(mx_2_em_bin, mx_2_em_hist, 'g-', label=r'${\rm cluster\ 2}$')
    pt.plot(mx_3_em_bin, mx_3_em_hist, 'c-', label=r'${\rm cluster\ 3}$')
    pt.plot(mx_1_em_bin, multivariate_normal.pdf(x=mx_1_em_bin, mean=mu1_true[0], cov=mu_em_var[0, 0]), 'k--')
    pt.plot(mx_2_em_bin, multivariate_normal.pdf(x=mx_2_em_bin, mean=mu2_true[0], cov=mu_em_var[1, 0]), 'k--')
    pt.plot(mx_3_em_bin, multivariate_normal.pdf(x=mx_3_em_bin, mean=mu3_true[0], cov=mu_em_var[2, 0]), 'k--')
    pt.subplot(2, 3, 3)
    pt.xlabel(r'$\mu_y$')
    pt.plot(my_1_em_bin, my_1_em_hist, 'r-', label=r'${\rm cluster\ 1}$')
    pt.plot(my_2_em_bin, my_2_em_hist, 'g-', label=r'${\rm cluster\ 2}$')
    pt.plot(my_3_em_bin, my_3_em_hist, 'c-', label=r'${\rm cluster\ 3}$')
    pt.plot(my_1_em_bin, multivariate_normal.pdf(x=my_1_em_bin, mean=mu1_true[1], cov=mu_em_var[0, 1]), 'k--')
    pt.plot(my_2_em_bin, multivariate_normal.pdf(x=my_2_em_bin, mean=mu2_true[1], cov=mu_em_var[1, 1]), 'k--')
    pt.plot(my_3_em_bin, multivariate_normal.pdf(x=my_3_em_bin, mean=mu3_true[1], cov=mu_em_var[2, 1]), 'k--')
    pt.subplot(2, 3, 4)
    pt.xlabel(r'$\Sigma_{xx}$')
    pt.plot(Sxx_1_em_bin, Sxx_1_em_hist, 'r-', label=r'${\rm cluster\ 1}$')
    pt.plot(Sxx_2_em_bin, Sxx_2_em_hist, 'g-', label=r'${\rm cluster\ 2}$')
    pt.plot(Sxx_3_em_bin, Sxx_3_em_hist, 'c-', label=r'${\rm cluster\ 3}$')
    pt.plot(Sxx_1_em_bin, multivariate_normal.pdf(x=Sxx_1_em_bin, mean=Sigma1_true[0, 0], cov=Sigma_em_var[0, 0, 0]), 'k--')
    pt.plot(Sxx_2_em_bin, multivariate_normal.pdf(x=Sxx_2_em_bin, mean=Sigma2_true[0, 0], cov=Sigma_em_var[1, 0, 0]), 'k--')
    pt.plot(Sxx_3_em_bin, multivariate_normal.pdf(x=Sxx_3_em_bin, mean=Sigma3_true[0, 0], cov=Sigma_em_var[2, 0, 0]), 'k--')
    pt.subplot(2, 3, 5)
    pt.xlabel(r'$\Sigma_{yy}$')
    pt.plot(Syy_1_em_bin, Syy_1_em_hist, 'r-', label=r'${\rm cluster\ 1}$')
    pt.plot(Syy_2_em_bin, Syy_2_em_hist, 'g-', label=r'${\rm cluster\ 2}$')
    pt.plot(Syy_3_em_bin, Syy_3_em_hist, 'c-', label=r'${\rm cluster\ 3}$')
    pt.plot(Syy_1_em_bin, multivariate_normal.pdf(x=Syy_1_em_bin, mean=Sigma1_true[1, 1], cov=Sigma_em_var[0, 1, 1]), 'k--')
    pt.plot(Syy_2_em_bin, multivariate_normal.pdf(x=Syy_2_em_bin, mean=Sigma2_true[1, 1], cov=Sigma_em_var[1, 1, 1]), 'k--')
    pt.plot(Syy_3_em_bin, multivariate_normal.pdf(x=Syy_3_em_bin, mean=Sigma3_true[1, 1], cov=Sigma_em_var[2, 1, 1]), 'k--')
    pt.subplot(2, 3, 6)
    pt.xlabel(r'$\Sigma_{xy}$')
    pt.plot(Sxy_1_em_bin, Sxy_1_em_hist, 'r-', label=r'${\rm cluster\ 1}$')
    pt.plot(Sxy_2_em_bin, Sxy_2_em_hist, 'g-', label=r'${\rm cluster\ 2}$')
    pt.plot(Sxy_3_em_bin, Sxy_3_em_hist, 'c-', label=r'${\rm cluster\ 3}$')
    pt.plot(Sxy_1_em_bin, multivariate_normal.pdf(x=Sxy_1_em_bin, mean=Sigma1_true[0, 1], cov=Sigma_em_var[0, 0, 1]), 'k--')
    pt.plot(Sxy_2_em_bin, multivariate_normal.pdf(x=Sxy_2_em_bin, mean=Sigma2_true[0, 1], cov=Sigma_em_var[1, 0, 1]), 'k--')
    pt.plot(Sxy_3_em_bin, multivariate_normal.pdf(x=Sxy_3_em_bin, mean=Sigma3_true[0, 1], cov=Sigma_em_var[2, 0, 1]), 'k--')
    pt.tight_layout()
    #pt.show()
    pt.savefig('param_histogram_montecarlo.png')

