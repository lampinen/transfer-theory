from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from theory_functions import *
### Parameters
num_examples = 100
output_sizes = [50] 
sigma_zs = [1] 
ps = [100]#, 800, 400, 200, 100, 90, 80, 70, 60, 40, 30, 20]
#num_runs = 10 
learning_rate = 0.001
num_epochs = 150000
#batch_size = num_examples
filename_prefix = "deep_single_generalization_paper_results/"
#input_type = "one_hot" # one_hot, orthogonal, gaussian
#track_SVD = True
save_every = 10
singular_value_multiplier = 10
epsilon = 1e-5
delta_x = 0.001 # for the numerical integration of M-P dist
N_2_bar = 1 # number of teacher modes
num_hidden = 50
num_hidden_layers = 3 # deeper is only supported for num_hidden_layers = 3, rank 1 and sigma z = 1 right now, not because of theoretical limitations, just to save me time
inverse_theory_num_points = 1000
singular_value_multipliers = [10., 0.84, 2., 4., 6., 8.] #[1., 2., 4., 8.] #np.arange(0., 10., 0.05) #

min_gen_approx = False # if true, only approximate min gen by assuming 1 or 0 learning of modes
### 

tau = 1./learning_rate
base_singular_values = [float(i) for i in range(N_2_bar, 0, -1)] 

def numeric_integral_mp(delta_x, t, x_min, x_max, A=1):
    x = np.arange(x_min, x_max, delta_x)
    return np.sum(mp(x, sigma_z, A=A)* (np.array(s_of_t(x, t, epsilon, tau))**2 )* delta_x)

def train_numeric_integral_mp(delta_x, t, x_min, x_max, A=1):
    x = np.arange(x_min, x_max, delta_x)
    return np.sum(mp(x, sigma_z, A=A)* ((np.array(s_of_t(x, t, epsilon, tau)) - x)**2 )* delta_x)

def prob_check_mp(delta_x, x_min, x_max, A=1):
    x = np.arange(x_min, x_max, delta_x)
    return np.sum(mp(x, 1, A=A)* delta_x)

for p in ps:
    for output_size in output_sizes:
        A = float(output_size)/num_examples
#    print(prob_check_mp(delta_x*1, 1-np.sqrt(A), 1+np.sqrt(A), A=A ))
        if min_gen_approx:
            with open(filename_prefix + "A_%f_min_gen_approx.csv" % A, "w") as fout:
                fout.write("sigma_z, p, singular_value_multiplier, approx_opt_test_error\n")

        for sigma_z in sigma_zs:
            assert(sigma_z == 1) # for now code doesn't work with other sigma_z
            if num_hidden_layers > 1:
                #prepare inverse theory solutions for MP dist integral
                x_min, x_max, this_delta_x = 1-np.sqrt(A), 1+np.sqrt(A), delta_x*sigma_z
                mp_x = np.arange(x_min, x_max, this_delta_x)
                deep_mp_sots = {}

                for x in mp_x:
                    points = np.arange(epsilon, x, (x - epsilon)/inverse_theory_num_points)
                    starting_point =   np.arctanh(np.sqrt(epsilon/x))/(2*x**1.5) - 0.5/(x*np.sqrt(epsilon))
                    times = tau* (np.arctanh(np.sqrt(points/x))/(2*x**1.5) - 0.5/(x*np.sqrt(points)) - starting_point)

                    sot = []
                    for epoch_i in xrange(1, num_epochs + 1, save_every):
                        this_index = np.argmin(np.abs(times-epoch_i)) # find closest time point to this
                        sot.append(points[this_index]) # use s value from that time point
                    deep_mp_sots[x] = sot

                def numeric_integral_mp_deep(delta_x, t, x_min, x_max, A=1):
                    x = np.arange(x_min, x_max, delta_x)
                    these_s_of_t = [deep_mp_sots[this_x][(t-1)//save_every] for this_x in x]
                    return np.sum(mp(x, sigma_z, A=A)* (np.array(these_s_of_t)**2 )* delta_x)

                def train_numeric_integral_mp_deep(delta_x, t, x_min, x_max, A=1):
                    x = np.arange(x_min, x_max, delta_x)
                    these_s_of_t = [deep_mp_sots[this_x][(t-1)//save_every] for this_x in x]
                    return np.sum(mp(x, sigma_z, A=A)* ((np.array(these_s_of_t) - x)**2 )* delta_x)

            for singular_value_multiplier in singular_value_multipliers:

                N_2 = num_hidden
                N_1 = num_examples
                N_3 = output_size

                noise_var = sigma_z**2

                singular_values = [s * np.sqrt(float(p)/N_1) * singular_value_multiplier for s in base_singular_values]

                y_frob_norm_sq = np.sum([s**2 for s in singular_values])

                net_rank = min(N_1, N_2, N_3)
                sigma_z = np.sqrt(noise_var)

#	s_hats = s_hat(singular_values, sigma_z)
                s_hats = s_hat_by_A(singular_values, A=A)
                s_bar = np.array(singular_values)
                noise_multiplier = get_noise_multiplier(s_bar, sigma_z, A=A) 

                if min_gen_approx:
                    with open(filename_prefix + "A_%f_min_gen_approx.csv" % (A), "a") as fout:
#                    sot = np.array([s_i if s_i/sigma_z > 1 else 0 for s_i in s_hats])
#
#                    generr = 0 #(N_2-len(singular_values))*numeric_integral_mp(delta_x*sigma_z, epoch_i, 0, 2*sigma_z) # number of points in integral estimate is constant in sigma_z 
#                    generr += np.sum(sot**2) 
#                    generr += y_frob_norm_sq
#                    generr -= 2 * np.sum(sot * s_bar * noise_multiplier) 
#                    generr /= y_frob_norm_sq
                        generr = 1-noise_multiplier**2
                        print("%f, %i, %f, %f" % (sigma_z, p, singular_value_multiplier, generr))
                        fout.write("%f, %i, %f, %f\n" % (sigma_z, p, singular_value_multiplier, generr))
                    continue

                with open(filename_prefix + "noise_var_%.2f_p_%i_svm_%f_theory_track.csv" % (noise_var, p, singular_value_multiplier), "w") as fout:
                    fout.write("epoch, generalization_error, s0, train_error\n")
                    noisy_y_frob_norm_sq = np.sum(np.array(s_hats)**2) +  (min(N_2, N_3)-len(singular_values)) * numeric_integral_mp(delta_x*sigma_z, 1e6, 1-np.sqrt(A), 1+np.sqrt(A), A=A) # hacky

                    if num_hidden_layers == 1:
                        for epoch_i in xrange(1, num_epochs + 1, save_every):

                            sot = np.array(s_of_t(s_hats, epoch_i, epsilon, tau))
                            
                            generr = (min(p, net_rank)-len(singular_values))*numeric_integral_mp(delta_x*sigma_z, epoch_i, 1-np.sqrt(A), 1+np.sqrt(A), A=A) # number of points in integral estimate is constant in sigma_z 
                            generr += np.sum(sot**2) 
                            generr += y_frob_norm_sq
                            generr -= 2 * np.sum(sot * s_bar * noise_multiplier) 
                            generr /= y_frob_norm_sq
                            trainerr = np.sum((s_hats[:net_rank]-sot[:net_rank])**2)
                            trainerr += (min(N_2, N_3)-len(singular_values)) * train_numeric_integral_mp(delta_x*sigma_z, epoch_i, 1-np.sqrt(A), 1+np.sqrt(A), A=A)
                            trainerr /= noisy_y_frob_norm_sq
                            print("%i, %f, %f, %f" % (epoch_i, generr, sot[0], trainerr))
                            fout.write("%i, %f, %f, %f\n" % (epoch_i, generr, sot[0], trainerr))

                    elif num_hidden_layers == 3:  # num hidden layers = 3, need inverse theory
                        this_s = s_hats[0] # we assume rank 1 so I don't have to write this code vectorized
                        points = np.arange(epsilon, this_s, (this_s - epsilon)/inverse_theory_num_points)

                        starting_point =   np.arctanh(np.sqrt(epsilon/this_s))/(2*this_s**1.5) - 0.5/(this_s*np.sqrt(epsilon))

                        est_times =  tau* (np.arctanh(np.sqrt(points/this_s))/(2*this_s**1.5) - 0.5/(this_s*np.sqrt(points)) - starting_point)

                        for epoch_i in xrange(1, num_epochs + 1, save_every):

                            this_index = np.argmin(np.abs(est_times-epoch_i)) # find closest time point to this
                            sot = np.array([points[this_index]]) # use s value from that time point
                            
                            generr = (min(p, net_rank)-1)*numeric_integral_mp_deep(delta_x*sigma_z, epoch_i, 1-np.sqrt(A), 1+np.sqrt(A), A=A) # number of points in integral estimate is constant in sigma_z 
                            generr += np.sum(sot**2) 
                            generr += y_frob_norm_sq
                            generr -= 2 * np.sum(sot * s_bar * noise_multiplier) 
                            generr /= y_frob_norm_sq
                            trainerr = np.sum((s_hats[:net_rank]-sot[:net_rank])**2)
                            trainerr += (min(N_2, N_3)-len(singular_values)) * train_numeric_integral_mp_deep(delta_x*sigma_z, epoch_i, 1-np.sqrt(A), 1+np.sqrt(A), A=A)
                            trainerr /= noisy_y_frob_norm_sq
                            print("%i, %f, %f, %f" % (epoch_i, generr, sot[0], trainerr))
                            fout.write("%i, %f, %f, %f\n" % (epoch_i, generr, sot[0], trainerr))

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
