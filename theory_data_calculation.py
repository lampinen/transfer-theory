from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from theory_functions import *
### Parameters
num_examples = 1000
output_sizes = [1000] 
noise_var = 10 
singular_values = [4., 3., 2., 1.] 
#num_runs = 10 
learning_rate = 0.001
num_epochs = 5000
num_hidden = 10
#batch_size = num_examples
filename_prefix = "single_generalization_full_results/"
#input_type = "one_hot" # one_hot, orthogonal, gaussian
#track_SVD = True
save_every = 5
singular_value_multiplier = 10
epsilon = 0.001
delta_x = 0.001 # for the numerical integration of M-P dist
### 

singular_values = [s * singular_value_multiplier for s in singular_values]

y_frob_norm_sq = np.sum([s**2 for s in singular_values])

N_2 = num_hidden
N_3 = N_1 = num_examples
tau = 1/learning_rate
sigma_z = np.sqrt(noise_var)

s_bar = np.array(singular_values)
noise_multiplier = 1 - (s_bar/sigma_z)**-2


def numeric_integral_mp(delta_x, t, x_min, x_max):
    x = np.arange(x_min, x_max, delta_x)
    return np.sum(mp(x)* s_of_t(x, t, epsilon, tau)* delta_x)


s_hats = s_hat(singular_values, sigma_z, N_2)


with open(filename_prefix + "noise_var_%.2f_theory_track.csv" % (noise_var), "w") as fout:
    fout.write("epoch, generalization_error\n")
    for epoch_i in xrange(1, num_epochs + 1, save_every):

        sot = np.array(s_of_t(s_hats, epoch_i, epsilon, tau))
        
        generr = numeric_integral_mp(0.001, epoch_i, 0, 2)
        generr += np.sum(sot**2) 
        generr += y_frob_norm_sq
        generr -= 2 * np.sum(sot * s_bar * noise_multiplier) 
        generr /= y_frob_norm_sq
        print("%i, %f" % (epoch_i, generr))
        fout.write("%i, %f\n" % (epoch_i, generr))
        

print("Final:")
print(generr)





