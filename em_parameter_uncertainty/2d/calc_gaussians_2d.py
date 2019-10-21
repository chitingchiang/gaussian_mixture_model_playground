import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import det

def calc_gaussians_2d(x, mu, Sigma):
    G = multivariate_normal.pdf(x, mean=mu, cov=Sigma)

    diff = x-mu
    D = det(Sigma)
    dGdm1_numerator = diff[:, 0]*Sigma[1, 1]-diff[:, 1]*Sigma[0, 1]
    dGdm2_numerator = diff[:, 1]*Sigma[0, 0]-diff[:, 0]*Sigma[0, 1]
    dGdS1_term = dGdm1_numerator**2/D-Sigma[1, 1]
    dGdS2_term = dGdm2_numerator**2/D-Sigma[0, 0]
    dGdS12_numerator = diff[:, 0]*diff[:, 1]*(Sigma[0, 0]*Sigma[1, 1]+Sigma[0, 1]**2) \
                      -Sigma[0, 1]*(diff[:, 0]**2*Sigma[1, 1]+diff[:, 1]**2*Sigma[0, 0])
    dGdS12_term = dGdS12_numerator/D+Sigma[0, 1]

    dGdm1 = G*dGdm1_numerator/D
    dGdm2 = G*dGdm2_numerator/D
    dGdS1 = 0.5*G/D*dGdS1_term
    dGdS2 = 0.5*G/D*dGdS2_term
    dGdS12 = G/D*dGdS12_term

    dG = np.array([dGdm1, dGdm2, dGdS1, dGdS2, dGdS12]).T

    ddGdm1dm1 = dGdS1*2
    #ddGdm1dm1 = (dGdm1*dGdm1_numerator-G*Sigma[1, 1])/D
    ddGdm1dm2 = (dGdm2*dGdm1_numerator+G*Sigma[0, 1])/D
    ddGdm1dS1 = (dGdS1-G*Sigma[1, 1]/D)*dGdm1_numerator/D
    ddGdm1dS2 = (dGdS2*dGdm1_numerator+G*(diff[:, 0]*D-Sigma[0, 0]*dGdm1_numerator)/D)/D
    ddGdm1dS12 = (dGdS12*dGdm1_numerator+G*(2*Sigma[0, 1]*dGdm1_numerator-diff[:, 1]*D)/D)/D

    ddGdm2dm2 = dGdS2*2
    #ddGdm2dm2 = (dGdm2*dGdm2_numerator-G*Sigma[0, 0])/D
    ddGdm2dS1 = (dGdS1*dGdm2_numerator+G*(diff[:, 1]*D-Sigma[1, 1]*dGdm2_numerator)/D)/D
    ddGdm2dS2 = (dGdS2-G*Sigma[0, 0]/D)*dGdm2_numerator/D
    ddGdm2dS12 = (dGdS12*dGdm2_numerator+G*(2*Sigma[0, 1]*dGdm2_numerator-diff[:, 0]*D)/D)/D

    ddGdS1dS1 = 0.5*(dGdS1*dGdS1_term-G/D*Sigma[1, 1]*(dGdS1_term+dGdm1_numerator**2/D))/D
    ddGdS1dS2 = 0.5*(dGdS2*dGdS1_term-G*((dGdS1_term*Sigma[0, 0]-dGdm1_numerator*(2*diff[:, 0]*D-Sigma[0, 0]*dGdm1_numerator)/D)/D+1))/D
    ddGdS1dS12 = (0.5*dGdS12*dGdS1_term+G*(dGdS1_term*Sigma[0, 1]+dGdm1_numerator*(Sigma[0, 1]*dGdm1_numerator-diff[:, 1]*D)/D)/D)/D

    ddGdS2dS2 = 0.5*(dGdS2*dGdS2_term-G/D*Sigma[0, 0]*(dGdS2_term+dGdm2_numerator**2/D))/D
    ddGdS2dS12 = (0.5*dGdS12*dGdS2_term+G*(dGdS2_term*Sigma[0, 1]+dGdm2_numerator*(Sigma[0, 1]*dGdm2_numerator-diff[:, 0]*D)/D)/D)/D

    ddGdS12dS12 = (dGdS12*dGdS12_term+2*G*dGdS12_term*Sigma[0, 1]/D
                  +G*(1+(2*diff[:, 0]*diff[:, 1]*Sigma[0, 1]-diff[:, 0]**2*Sigma[1, 1]-diff[:, 1]**2*Sigma[0, 0])/D
                     +2*Sigma[0, 1]*dGdS12_numerator/D**2))/D

    ddG = np.array([[  ddGdm1dm1,   ddGdm1dm2,   ddGdm1dS1,   ddGdm1dS2,  ddGdm1dS12],
                    [  ddGdm1dm2,   ddGdm2dm2,   ddGdm2dS1,   ddGdm2dS2,  ddGdm2dS12],
                    [  ddGdm1dS1,   ddGdm2dS1,   ddGdS1dS1,   ddGdS1dS2,  ddGdS1dS12],
                    [  ddGdm1dS2,   ddGdm2dS2,   ddGdS1dS2,   ddGdS2dS2,  ddGdS2dS12],
                    [ ddGdm1dS12,  ddGdm2dS12,  ddGdS1dS12,  ddGdS2dS12, ddGdS12dS12]]).T

    return G, dG.T, ddG.T

if __name__=='__main__':
    mu = np.array([1, 1])
    Sigma = np.array([[0.5, 0.2], [0.2, 0.4]])

    x = np.array([[0.5, 1.5], [1.5, 0.5], [2, 3]])

    delta = 1e-5

    m1_p = [mu[0]+delta, mu[1]]
    m1_n = [mu[0]-delta, mu[1]]
    m2_p = [mu[0], mu[1]+delta]
    m2_n = [mu[0], mu[1]-delta]
    S1_p = [[Sigma[0, 0]+delta, Sigma[0, 1]], [Sigma[0, 1], Sigma[1, 1]]]
    S1_n = [[Sigma[0, 0]-delta, Sigma[0, 1]], [Sigma[0, 1], Sigma[1, 1]]]
    S2_p = [[Sigma[0, 0], Sigma[0, 1]], [Sigma[0, 1], Sigma[1, 1]+delta]]
    S2_n = [[Sigma[0, 0], Sigma[0, 1]], [Sigma[0, 1], Sigma[1, 1]-delta]]
    S12_p = [[Sigma[0, 0], Sigma[0, 1]+delta], [Sigma[0, 1]+delta, Sigma[1, 1]]]
    S12_n = [[Sigma[0, 0], Sigma[0, 1]-delta], [Sigma[0, 1]-delta, Sigma[1, 1]]]

    '''
    #check first derivative
    print(dG[:, 0])
    print((multivariate_normal.pdf(x, mean=m1_p, cov=Sigma)-\
           multivariate_normal.pdf(x, mean=m1_n, cov=Sigma))/2/delta)

    print(dG[:, 1])
    print((multivariate_normal.pdf(x, mean=m2_p, cov=Sigma)-\
           multivariate_normal.pdf(x, mean=m2_n, cov=Sigma))/2/delta)

    print(dG[:, 2])
    print((multivariate_normal.pdf(x, mean=mu, cov=S1_p)-\
           multivariate_normal.pdf(x, mean=mu, cov=S1_n))/2/delta)

    print(dG[:, 3])
    print((multivariate_normal.pdf(x, mean=mu, cov=S2_p)-\
           multivariate_normal.pdf(x, mean=mu, cov=S2_n))/2/delta)

    print(dG[:, 4])
    print((multivariate_normal.pdf(x, mean=mu, cov=S12_p)-\
           multivariate_normal.pdf(x, mean=mu, cov=S12_n))/2/delta)
    '''

    G, dG, ddG = calc_gaussians_2d(x, mu, Sigma)
    print(G.shape, dG.shape, ddG.shape)

    #print((multivariate_normal.pdf(x, mean=m1_p, cov=Sigma)
    #      +multivariate_normal.pdf(x, mean=m1_n, cov=Sigma)
    #      -2*multivariate_normal.pdf(x, mean=mu, cov=Sigma))/delta**2)

    #print((multivariate_normal.pdf(x, mean=[mu[0]+delta, mu[1]+delta], cov=Sigma)
    #      -multivariate_normal.pdf(x, mean=[mu[0]+delta, mu[1]-delta], cov=Sigma)
    #      -multivariate_normal.pdf(x, mean=[mu[0]-delta, mu[1]+delta], cov=Sigma)
    #      +multivariate_normal.pdf(x, mean=[mu[0]-delta, mu[1]-delta], cov=Sigma))/4/delta**2)

    #print((multivariate_normal.pdf(x, mean=m1_p, cov=S1_p)
    #      -multivariate_normal.pdf(x, mean=m1_p, cov=S1_n)
    #      -multivariate_normal.pdf(x, mean=m1_n, cov=S1_p)
    #      +multivariate_normal.pdf(x, mean=m1_n, cov=S1_n))/4/delta**2)

    #print((multivariate_normal.pdf(x, mean=m1_p, cov=S2_p)
    #      -multivariate_normal.pdf(x, mean=m1_p, cov=S2_n)
    #      -multivariate_normal.pdf(x, mean=m1_n, cov=S2_p)
    #      +multivariate_normal.pdf(x, mean=m1_n, cov=S2_n))/4/delta**2)

    #print((multivariate_normal.pdf(x, mean=m1_p, cov=S12_p)
    #      -multivariate_normal.pdf(x, mean=m1_p, cov=S12_n)
    #      -multivariate_normal.pdf(x, mean=m1_n, cov=S12_p)
    #      +multivariate_normal.pdf(x, mean=m1_n, cov=S12_n))/4/delta**2)

    #print((multivariate_normal.pdf(x, mean=m2_p, cov=Sigma)
    #      +multivariate_normal.pdf(x, mean=m2_n, cov=Sigma)
    #      -2*multivariate_normal.pdf(x, mean=mu, cov=Sigma))/delta**2)

    #print((multivariate_normal.pdf(x, mean=m2_p, cov=S1_p)
    #      -multivariate_normal.pdf(x, mean=m2_p, cov=S1_n)
    #      -multivariate_normal.pdf(x, mean=m2_n, cov=S1_p)
    #      +multivariate_normal.pdf(x, mean=m2_n, cov=S1_n))/4/delta**2)

    #print((multivariate_normal.pdf(x, mean=m2_p, cov=S2_p)
    #      -multivariate_normal.pdf(x, mean=m2_p, cov=S2_n)
    #      -multivariate_normal.pdf(x, mean=m2_n, cov=S2_p)
    #      +multivariate_normal.pdf(x, mean=m2_n, cov=S2_n))/4/delta**2)

    #print((multivariate_normal.pdf(x, mean=m2_p, cov=S12_p)
    #      -multivariate_normal.pdf(x, mean=m2_p, cov=S12_n)
    #      -multivariate_normal.pdf(x, mean=m2_n, cov=S12_p)
    #      +multivariate_normal.pdf(x, mean=m2_n, cov=S12_n))/4/delta**2)

    #print((multivariate_normal.pdf(x, mean=mu, cov=S1_p)
    #      +multivariate_normal.pdf(x, mean=mu, cov=S1_n)
    #      -2*multivariate_normal.pdf(x, mean=mu, cov=Sigma))/delta**2)

    #print((multivariate_normal.pdf(x, mean=mu, cov=[[Sigma[0, 0]+delta, Sigma[0, 1]], [Sigma[1, 0], Sigma[1, 1]+delta]])
    #      -multivariate_normal.pdf(x, mean=mu, cov=[[Sigma[0, 0]+delta, Sigma[0, 1]], [Sigma[1, 0], Sigma[1, 1]-delta]])
    #      -multivariate_normal.pdf(x, mean=mu, cov=[[Sigma[0, 0]-delta, Sigma[0, 1]], [Sigma[1, 0], Sigma[1, 1]+delta]])
    #      +multivariate_normal.pdf(x, mean=mu, cov=[[Sigma[0, 0]-delta, Sigma[0, 1]], [Sigma[1, 0], Sigma[1, 1]-delta]]))/4/delta**2)

    #print((multivariate_normal.pdf(x, mean=mu, cov=[[Sigma[0, 0]+delta, Sigma[0, 1]+delta], [Sigma[1, 0]+delta, Sigma[1, 1]]])
    #      -multivariate_normal.pdf(x, mean=mu, cov=[[Sigma[0, 0]+delta, Sigma[0, 1]-delta], [Sigma[1, 0]-delta, Sigma[1, 1]]])
    #      -multivariate_normal.pdf(x, mean=mu, cov=[[Sigma[0, 0]-delta, Sigma[0, 1]+delta], [Sigma[1, 0]+delta, Sigma[1, 1]]])
    #      +multivariate_normal.pdf(x, mean=mu, cov=[[Sigma[0, 0]-delta, Sigma[0, 1]-delta], [Sigma[1, 0]-delta, Sigma[1, 1]]]))/4/delta**2)

    #print((multivariate_normal.pdf(x, mean=mu, cov=S2_p)
    #      +multivariate_normal.pdf(x, mean=mu, cov=S2_n)
    #      -2*multivariate_normal.pdf(x, mean=mu, cov=Sigma))/delta**2)

    #print((multivariate_normal.pdf(x, mean=mu, cov=[[Sigma[0, 0], Sigma[0, 1]+delta], [Sigma[1, 0]+delta, Sigma[1, 1]+delta]])
    #      -multivariate_normal.pdf(x, mean=mu, cov=[[Sigma[0, 0], Sigma[0, 1]-delta], [Sigma[1, 0]-delta, Sigma[1, 1]+delta]])
    #      -multivariate_normal.pdf(x, mean=mu, cov=[[Sigma[0, 0], Sigma[0, 1]+delta], [Sigma[1, 0]+delta, Sigma[1, 1]-delta]])
    #      +multivariate_normal.pdf(x, mean=mu, cov=[[Sigma[0, 0], Sigma[0, 1]-delta], [Sigma[1, 0]-delta, Sigma[1, 1]-delta]]))/4/delta**2)

    #print((multivariate_normal.pdf(x, mean=mu, cov=S12_p)
    #      +multivariate_normal.pdf(x, mean=mu, cov=S12_n)
    #      -2*multivariate_normal.pdf(x, mean=mu, cov=Sigma))/delta**2)
