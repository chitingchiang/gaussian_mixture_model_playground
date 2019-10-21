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

    data1 = np.load('data1.npy')
    data2 = np.load('data2.npy')
    data3 = np.load('data3.npy')
    data = np.concatenate((data1, data2, data3))

    pi_bs = np.load('pi_bs.npy')
    mu_bs = np.load('mu_bs.npy')
    Sigma_bs = np.load('Sigma_bs.npy')

    pi_bs_mean = np.mean(pi_bs, axis=0)
    mu_bs_mean = np.mean(mu_bs, axis=0)
    Sigma_bs_mean = np.mean(Sigma_bs, axis=0)

    pi_bs_var = np.var(pi_bs, axis=0)
    mu_bs_var = np.var(mu_bs, axis=0)
    Sigma_bs_var = np.var(Sigma_bs, axis=0)

    var_measure = pi_bs_var[:-1]
    for c in range(3):
        var_measure = np.append(var_measure,
                               [mu_bs_var[c, 0], mu_bs_var[c, 1], Sigma_bs_var[c, 0, 0], Sigma_bs_var[c, 1, 1], Sigma_bs_var[c, 0, 1]])


    fisher, cov = calc_fisher_cov(data, np.mean(pi_bs, axis=0), np.mean(mu_bs, axis=0), np.mean(Sigma_bs, axis=0))
    var_theory = np.diagonal(cov)

    print('measure, theory')
    for i in range(17):
        if i<2:
            print('%.8e %.8e'%(var_measure[i], var_theory[i]))
        else:
            print('%.8e %.8e'%(var_measure[i], var_theory[i+1]))

    pi_1_bs_hist, pi_1_bs_bin = np.histogram(pi_bs[:, 0], bins=20, density=True)
    mx_1_bs_hist, mx_1_bs_bin = np.histogram(mu_bs[:, 0, 0], bins=100, density=True)
    my_1_bs_hist, my_1_bs_bin = np.histogram(mu_bs[:, 0, 1], bins=100, density=True)
    Sxx_1_bs_hist, Sxx_1_bs_bin = np.histogram(Sigma_bs[:, 0, 0, 0], bins=100, density=True)
    Syy_1_bs_hist, Syy_1_bs_bin = np.histogram(Sigma_bs[:, 0, 1, 1], bins=100, density=True)
    Sxy_1_bs_hist, Sxy_1_bs_bin = np.histogram(Sigma_bs[:, 0, 0, 1], bins=100, density=True)

    pi_2_bs_hist, pi_2_bs_bin = np.histogram(pi_bs[:, 1], bins=20, density=True)
    mx_2_bs_hist, mx_2_bs_bin = np.histogram(mu_bs[:, 1, 0], bins=100, density=True)
    my_2_bs_hist, my_2_bs_bin = np.histogram(mu_bs[:, 1, 1], bins=100, density=True)
    Sxx_2_bs_hist, Sxx_2_bs_bin = np.histogram(Sigma_bs[:, 1, 0, 0], bins=100, density=True)
    Syy_2_bs_hist, Syy_2_bs_bin = np.histogram(Sigma_bs[:, 1, 1, 1], bins=100, density=True)
    Sxy_2_bs_hist, Sxy_2_bs_bin = np.histogram(Sigma_bs[:, 1, 0, 1], bins=100, density=True)

    pi_3_bs_hist, pi_3_bs_bin = np.histogram(pi_bs[:, 2], bins=20, density=True)
    mx_3_bs_hist, mx_3_bs_bin = np.histogram(mu_bs[:, 2, 0], bins=100, density=True)
    my_3_bs_hist, my_3_bs_bin = np.histogram(mu_bs[:, 2, 1], bins=100, density=True)
    Sxx_3_bs_hist, Sxx_3_bs_bin = np.histogram(Sigma_bs[:, 2, 0, 0], bins=100, density=True)
    Syy_3_bs_hist, Syy_3_bs_bin = np.histogram(Sigma_bs[:, 2, 1, 1], bins=100, density=True)
    Sxy_3_bs_hist, Sxy_3_bs_bin = np.histogram(Sigma_bs[:, 2, 0, 1], bins=100, density=True)

    pi_1_bs_bin = 0.5*(pi_1_bs_bin[:-1]+pi_1_bs_bin[1:])
    pi_2_bs_bin = 0.5*(pi_2_bs_bin[:-1]+pi_2_bs_bin[1:])
    pi_3_bs_bin = 0.5*(pi_3_bs_bin[:-1]+pi_3_bs_bin[1:])

    mx_1_bs_bin = 0.5*(mx_1_bs_bin[:-1]+mx_1_bs_bin[1:])
    mx_2_bs_bin = 0.5*(mx_2_bs_bin[:-1]+mx_2_bs_bin[1:])
    mx_3_bs_bin = 0.5*(mx_3_bs_bin[:-1]+mx_3_bs_bin[1:])

    my_1_bs_bin = 0.5*(my_1_bs_bin[:-1]+my_1_bs_bin[1:])
    my_2_bs_bin = 0.5*(my_2_bs_bin[:-1]+my_2_bs_bin[1:])
    my_3_bs_bin = 0.5*(my_3_bs_bin[:-1]+my_3_bs_bin[1:])

    Sxx_1_bs_bin = 0.5*(Sxx_1_bs_bin[:-1]+Sxx_1_bs_bin[1:])
    Sxx_2_bs_bin = 0.5*(Sxx_2_bs_bin[:-1]+Sxx_2_bs_bin[1:])
    Sxx_3_bs_bin = 0.5*(Sxx_3_bs_bin[:-1]+Sxx_3_bs_bin[1:])

    Syy_1_bs_bin = 0.5*(Syy_1_bs_bin[:-1]+Syy_1_bs_bin[1:])
    Syy_2_bs_bin = 0.5*(Syy_2_bs_bin[:-1]+Syy_2_bs_bin[1:])
    Syy_3_bs_bin = 0.5*(Syy_3_bs_bin[:-1]+Syy_3_bs_bin[1:])

    Sxy_1_bs_bin = 0.5*(Sxy_1_bs_bin[:-1]+Sxy_1_bs_bin[1:])
    Sxy_2_bs_bin = 0.5*(Sxy_2_bs_bin[:-1]+Sxy_2_bs_bin[1:])
    Sxy_3_bs_bin = 0.5*(Sxy_3_bs_bin[:-1]+Sxy_3_bs_bin[1:])

    pt.figure(1, figsize=(14, 8))
    pt.subplot(2, 3, 1)
    pt.xlabel(r'$\pi$')
    pt.plot(pi_1_bs_bin, pi_1_bs_hist, 'r-', label=r'${\rm cluster\ 1}$')
    pt.plot(pi_2_bs_bin, pi_2_bs_hist, 'g-', label=r'${\rm cluster\ 2}$')
    pt.plot(pi_3_bs_bin, pi_3_bs_hist, 'c-', label=r'${\rm cluster\ 3}$')
    pt.plot(pi_1_bs_bin, multivariate_normal.pdf(x=pi_1_bs_bin, mean=pi_bs_mean[0], cov=pi_bs_var[0]), 'k--')
    pt.plot(pi_2_bs_bin, multivariate_normal.pdf(x=pi_2_bs_bin, mean=pi_bs_mean[1], cov=pi_bs_var[1]), 'k--')
    pt.plot(pi_3_bs_bin, multivariate_normal.pdf(x=pi_3_bs_bin, mean=pi_bs_mean[2], cov=pi_bs_var[2]), 'k--')
    pt.subplot(2, 3, 2)
    pt.xlabel(r'$\mu_x$')
    pt.plot(mx_1_bs_bin, mx_1_bs_hist, 'r-', label=r'${\rm cluster\ 1}$')
    pt.plot(mx_2_bs_bin, mx_2_bs_hist, 'g-', label=r'${\rm cluster\ 2}$')
    pt.plot(mx_3_bs_bin, mx_3_bs_hist, 'c-', label=r'${\rm cluster\ 3}$')
    pt.plot(mx_1_bs_bin, multivariate_normal.pdf(x=mx_1_bs_bin, mean=mu_bs_mean[0, 0], cov=mu_bs_var[0, 0]), 'k--')
    pt.plot(mx_2_bs_bin, multivariate_normal.pdf(x=mx_2_bs_bin, mean=mu_bs_mean[1, 0], cov=mu_bs_var[1, 0]), 'k--')
    pt.plot(mx_3_bs_bin, multivariate_normal.pdf(x=mx_3_bs_bin, mean=mu_bs_mean[2, 0], cov=mu_bs_var[2, 0]), 'k--')
    pt.subplot(2, 3, 3)
    pt.xlabel(r'$\mu_y$')
    pt.plot(my_1_bs_bin, my_1_bs_hist, 'r-', label=r'${\rm cluster\ 1}$')
    pt.plot(my_2_bs_bin, my_2_bs_hist, 'g-', label=r'${\rm cluster\ 2}$')
    pt.plot(my_3_bs_bin, my_3_bs_hist, 'c-', label=r'${\rm cluster\ 3}$')
    pt.plot(my_1_bs_bin, multivariate_normal.pdf(x=my_1_bs_bin, mean=mu_bs_mean[0, 1], cov=mu_bs_var[0, 1]), 'k--')
    pt.plot(my_2_bs_bin, multivariate_normal.pdf(x=my_2_bs_bin, mean=mu_bs_mean[1, 1], cov=mu_bs_var[1, 1]), 'k--')
    pt.plot(my_3_bs_bin, multivariate_normal.pdf(x=my_3_bs_bin, mean=mu_bs_mean[2, 1], cov=mu_bs_var[2, 1]), 'k--')
    pt.subplot(2, 3, 4)
    pt.xlabel(r'$\Sigma_{xx}$')
    pt.plot(Sxx_1_bs_bin, Sxx_1_bs_hist, 'r-', label=r'${\rm cluster\ 1}$')
    pt.plot(Sxx_2_bs_bin, Sxx_2_bs_hist, 'g-', label=r'${\rm cluster\ 2}$')
    pt.plot(Sxx_3_bs_bin, Sxx_3_bs_hist, 'c-', label=r'${\rm cluster\ 3}$')
    pt.plot(Sxx_1_bs_bin, multivariate_normal.pdf(x=Sxx_1_bs_bin, mean=Sigma_bs_mean[0, 0, 0], cov=Sigma_bs_var[0, 0, 0]), 'k--')
    pt.plot(Sxx_2_bs_bin, multivariate_normal.pdf(x=Sxx_2_bs_bin, mean=Sigma_bs_mean[1, 0, 0], cov=Sigma_bs_var[1, 0, 0]), 'k--')
    pt.plot(Sxx_3_bs_bin, multivariate_normal.pdf(x=Sxx_3_bs_bin, mean=Sigma_bs_mean[2, 0, 0], cov=Sigma_bs_var[2, 0, 0]), 'k--')
    pt.subplot(2, 3, 5)
    pt.xlabel(r'$\Sigma_{yy}$')
    pt.plot(Syy_1_bs_bin, Syy_1_bs_hist, 'r-', label=r'${\rm cluster\ 1}$')
    pt.plot(Syy_2_bs_bin, Syy_2_bs_hist, 'g-', label=r'${\rm cluster\ 2}$')
    pt.plot(Syy_3_bs_bin, Syy_3_bs_hist, 'c-', label=r'${\rm cluster\ 3}$')
    pt.plot(Syy_1_bs_bin, multivariate_normal.pdf(x=Syy_1_bs_bin, mean=Sigma_bs_mean[0, 1, 1], cov=Sigma_bs_var[0, 1, 1]), 'k--')
    pt.plot(Syy_2_bs_bin, multivariate_normal.pdf(x=Syy_2_bs_bin, mean=Sigma_bs_mean[1, 1, 1], cov=Sigma_bs_var[1, 1, 1]), 'k--')
    pt.plot(Syy_3_bs_bin, multivariate_normal.pdf(x=Syy_3_bs_bin, mean=Sigma_bs_mean[2, 1, 1], cov=Sigma_bs_var[2, 1, 1]), 'k--')
    pt.subplot(2, 3, 6)
    pt.xlabel(r'$\Sigma_{xy}$')
    pt.plot(Sxy_1_bs_bin, Sxy_1_bs_hist, 'r-', label=r'${\rm cluster\ 1}$')
    pt.plot(Sxy_2_bs_bin, Sxy_2_bs_hist, 'g-', label=r'${\rm cluster\ 2}$')
    pt.plot(Sxy_3_bs_bin, Sxy_3_bs_hist, 'c-', label=r'${\rm cluster\ 3}$')
    pt.plot(Sxy_1_bs_bin, multivariate_normal.pdf(x=Sxy_1_bs_bin, mean=Sigma_bs_mean[0, 0, 1], cov=Sigma_bs_var[0, 0, 1]), 'k--')
    pt.plot(Sxy_2_bs_bin, multivariate_normal.pdf(x=Sxy_2_bs_bin, mean=Sigma_bs_mean[1, 0, 1], cov=Sigma_bs_var[1, 0, 1]), 'k--')
    pt.plot(Sxy_3_bs_bin, multivariate_normal.pdf(x=Sxy_3_bs_bin, mean=Sigma_bs_mean[2, 0, 1], cov=Sigma_bs_var[2, 0, 1]), 'k--')
    pt.tight_layout()
    #pt.show()
    pt.savefig('param_histogram_bootstrap.png')

