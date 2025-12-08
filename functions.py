import numpy as np
import matplotlib.pyplot as plt
from math import comb
from scipy import stats


def log_lik(theta, x, y, n):
    alpha, beta = theta

    p = 1 / (1 + np.exp(-(alpha + beta * x)))

    ll = np.sum(stats.binom.logpmf(y, n, p))
    
    return ll

def lik(theta, x, y, n):
    return np.exp(log_lik(theta, x, y, n))

def neg_log_lik(theta, x, y, n):
    return -log_lik(theta, x, y, n)

def neg_lik(theta, x, y, n):
    return -lik(theta, x, y, n)

# def log_prior(theta, sigma_prior=10):
#     alpha, beta = theta
#     return - (alpha**2 + beta**2) / (2 * sigma_prior**2)

def log_prior(theta):
    alpha, beta = theta
        
    lp_alpha = - (alpha**2) / (2 * 2**2)  # sigma=2
    lp_beta =  - (beta**2)  / (2 * 1**2)  # sigma=1
    
    return lp_alpha + lp_beta

def log_posterior(theta, x, y, n):
    return log_lik(theta, x,y,n) + log_prior(theta)

def neg_log_posterior(theta,x,y,n):
    return -log_posterior(theta, x,y,n)