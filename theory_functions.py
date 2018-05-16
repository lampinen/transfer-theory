import numpy as np

def get_noise_multiplier(s_bar, sigma_z, A=1):
    if A == 1:
        res = (1 - (s_bar/sigma_z)**-2)
        res[s_bar/sigma_z <= 1] = 0.
    else:
        s_b_sc = s_bar/sigma_z
        res = np.sqrt(1 - A * (1+s_b_sc**2)/(s_b_sc**2 * (A+ s_b_sc**2 ))) * np.sqrt(1 -  (A+s_b_sc**2)/(s_b_sc**2 * (1+ s_b_sc**2 ))) 
        res[s_b_sc/np.power(A, 1./4) <= 1] = 0.
    return res


def s_hat(s, sigma_z):
    """estimate singular value as function of true singular value"""
    def _s_hat(s_i):
        return (sigma_z**2 +s_i**2)/s_i if s_i/sigma_z > 1 else 2 * sigma_z  # ??

    return [_s_hat(s_i) for s_i in s]

def s_hat_by_A(s, A=1):
    """estimate singular value as function of true singular value at other aspect ratios, with sigma_z = 1"""
    def _s_hat(s_i):
        return np.sqrt((1+s_i**2)*(A+s_i**2))/s_i if s_i/np.power(A, 1./4) > 1 else 1 + np.sqrt(A) 

    return [_s_hat(s_i) for s_i in s]

def mp(s_hat, sigma_z, A=1):
    """Marchenko-pastur distribution"""

    indices = np.logical_or(s_hat <= sigma_z* (1-np.sqrt(A)), s_hat  >= (1+np.sqrt(A)) * sigma_z)
    res = np.sqrt(4*A- ((s_hat/sigma_z)**2 - (1+A))**2)/(np.pi *A* s_hat)
    res[indices] = 0.
   
    return res 

def s_of_t(s_hat, t, epsilon, tau):
    """Exact solutions to mode growth from Saxe et al. 2013"""
    def _s_of_t(s_hat_i):
        return (s_hat_i )/(1 + np.exp(-2* s_hat_i * t/tau) *(s_hat_i/epsilon - 1))

    return [_s_of_t(s_hat_i) if s_hat_i > 0 else 0 for s_hat_i in s_hat]
