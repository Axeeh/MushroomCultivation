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