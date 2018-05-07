import numpy as np

def s_hat(s, sigma_z, N_2):
    """estimate singular value as function of true singular value"""
    def _s_hat(s_i):
        return (sigma_z**2 +s_i**2)/s_i if s_i > 1 else sigma_z * (1+N_2) 

    return [_s_hat(s_i) for s_i in s]


def mp(s_hat):
    """Marchenko-pastur distribution"""
    s_hat[s_hat <= 0] = 0
    s_hat[s_hat >= 2] = 0
    return np.sqrt(4- (s_hat**2 - 2)**2)

def s_of_t(s_hat, t, epsilon, tau):
    """Exact solutions to mode growth from Saxe et al. 2013"""
    def _s_of_t(s_hat_i):
        return (s_hat_i * np.exp(2 *s_hat_i * t/tau))/(np.exp(2 *s_hat_i * t/tau) + s_hat_i/epsilon - 1)

    return [_s_of_t(s_hat_i) if s_hat_i > 0 else 0 for s_hat_i in s_hat]
