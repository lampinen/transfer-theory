from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from theory_functions import *
### Parameters
num_examples = 100
output_sizes = [100] 
sigma_zs = [1]
#num_runs = 10 
learning_rate = 0.001
num_epochs = 5000
#batch_size = num_examples
filename_prefix = "single_generalization_comparison_results_rank4/"
#input_type = "one_hot" # one_hot, orthogonal, gaussian
#track_SVD = True
save_every = 5
singular_value_multiplier = 10
epsilon = 1e-5
delta_x = 0.001 # for the numerical integration of M-P dist
N_2_bar = 4 # number of teacher modes
num_hidden = num_examples 
singular_value_multipliers = [float(i) for i in range(1,11)]
### 

base_singular_values = [float(i) for i in range(N_2_bar, 0, -1)] 


def numeric_integral_mp(delta_x, t, x_min, x_max):
    x = np.arange(x_min, x_max, delta_x)
    return np.sum(mp(x, sigma_z)* (np.array(s_of_t(x, t, epsilon, tau))**2 )* delta_x)


def get_noise_multiplier(s_bar, sigma_z):
    res = (1 - (s_bar/sigma_z)**-2)
    res[s_bar/sigma_z <= 1] = 0.
    return res


for sigma_z in sigma_zs:
    for singular_value_multiplier in singular_value_multipliers:
	noise_var = sigma_z**2

	singular_values = [s * singular_value_multiplier for s in base_singular_values]

	y_frob_norm_sq = np.sum([s**2 for s in singular_values])

	N_2 = num_hidden
	N_3 = N_1 = num_examples
	tau = 1./learning_rate
	sigma_z = np.sqrt(noise_var)

	s_hats = s_hat(singular_values, sigma_z)
	s_bar = np.array(singular_values)
	noise_multiplier = get_noise_multiplier(s_bar, sigma_z) 
	with open(filename_prefix + "noise_var_%.2f_svm_%f_theory_track.csv" % (noise_var, singular_value_multiplier), "w") as fout:
	    fout.write("epoch, generalization_error, s0\n")
	    for epoch_i in xrange(1, num_epochs + 1, save_every):

		sot = np.array(s_of_t(s_hats, epoch_i, epsilon, tau))
		
		generr = (N_2-len(singular_values))*numeric_integral_mp(delta_x*sigma_z, epoch_i, 0, 2*sigma_z) # number of points in integral estimate is constant in sigma_z 
		generr += np.sum(sot**2) 
		generr += y_frob_norm_sq
		generr -= 2 * np.sum(sot * s_bar * noise_multiplier) 
		generr /= y_frob_norm_sq
		print("%i, %f, %f" % (epoch_i, generr, sot[0]))
		fout.write("%i, %f, %f\n" % (epoch_i, generr, sot[0]))

    #    print()
    #    print()
    #    print(s_hats)
    #    print(noise_multiplier)
    #    generr = 0 #numeric_integral_mp(0.001, epoch_i, 0, 2)
    #    generr += np.sum(np.array(s_hats)**2) 
    #    print(generr)
    #    generr += y_frob_norm_sq
    #    print(generr)
    #    generr -= 2 * np.sum(s_hats * s_bar * noise_multiplier) 
    #    print(generr)
    #    generr /= y_frob_norm_sq
    #    print(generr) 
    #        
