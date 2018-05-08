import numpy as np

def s_hat(s, sigma_z):
    """estimate singular value as function of true singular value"""
    def _s_hat(s_i):
        return (sigma_z**2 +s_i**2)/s_i if s_i/sigma_z > 1 else 2 * sigma_z  # ??

    return [_s_hat(s_i) for s_i in s]


def mp(s_hat, sigma_z):
    """Marchenko-pastur distribution"""

    indices = np.logical_or(s_hat <= 0, s_hat >= 2 * sigma_z)
    res = np.sqrt(4- ((s_hat/sigma_z)**2 - 2)**2)/(np.pi * s_hat)
    res[indices] = 0.
   
    return res 

def s_of_t(s_hat, t, epsilon, tau):
    """Exact solutions to mode growth from Saxe et al. 2013"""
    def _s_of_t(s_hat_i):
        return (s_hat_i )/(1 + np.exp(-2* s_hat_i * t/tau) *(s_hat_i/epsilon - 1))

    return [_s_of_t(s_hat_i) if s_hat_i > 0 else 0 for s_hat_i in s_hat]
