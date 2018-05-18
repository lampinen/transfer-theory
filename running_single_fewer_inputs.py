from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import datasets
from orthogonal_matrices import random_orthogonal
### Parameters
num_examples = 100
output_sizes = [50] #[10, 50, 100, 200, 400]
ps = [100, 50, 20, 4]
sigma_zs = [1] 
num_runs = 10
learning_rate = 0.001
num_epochs = 5000
batch_size = num_examples
filename_prefix = "fewer_inputs_results/"
input_type = "orthogonal" # one_hot, orthogonal, gaussian
track_SVD = False
save_every = 5
epsilon = 1e-4
singular_value_multiplier = 10 
N_2_bar = 1 # rank of teacher
singular_value_multipliers = [float(x) for x in [0.84, 2, 6, 10, 100]]
num_hidden = 50#num_examples

###
#var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=2*np.sqrt(epsilon), mode='FAN_AVG')

for run_i in xrange(num_runs):
    for sigma_z in sigma_zs:
        noise_var = sigma_z**2
	for singular_value_multiplier in singular_value_multipliers:
	    for aligned in [True, False]:
		for output_size in output_sizes:
                    for num_inputs in ps:
		    
                        scaled_noise_var = noise_var/output_size
                        np.random.seed(run_i)
                        print("Now running noise_var: %.2f, output_size: %i, num_inputs: %i, svm: %f, alignment: %s, run: %i" % (noise_var, output_size, num_inputs, singular_value_multiplier, aligned, run_i))
              
                        num_input = num_examples
                        num_output = output_size
                        if input_type == "one_hot":
                            x_data, y_data, noisy_y_data, input_modes = datasets.noisy_SVD_dataset(num_examples, output_size, noise_var=scaled_noise_var, singular_value_multiplier=singular_value_multiplier, num_nonempty=N_2_bar) 
                        else:
                            x_data, y_data, noisy_y_data, input_modes = datasets.noisy_SVD_dataset_different_inputs(num_examples, output_size, noise_var=scaled_noise_var, input_type=input_type, singular_value_multiplier=singular_value_multiplier, num_nonempty=N_2_bar) 

                        noisy_y_data = noisy_y_data.transpose()

                        y_data = y_data.transpose()

                        y_data_frob_squared = np.sum(y_data**2)
              
                        if track_SVD:
                            U_bar, S_bar, V_bar = np.linalg.svd(y_data, full_matrices=False)
                            U_bar = U_bar[:, :N_2_bar]; S_bar = S_bar[:N_2_bar]; V_bar = V_bar[:N_2_bar, :]; # save some computation later -- other singular values are zero
                            U_hat, S_hat, V_hat = np.linalg.svd(noisy_y_data, full_matrices=False)
                            U_hat_base = U_hat; V_hat_base=V_hat;
                            U_hat = U_hat[:, :N_2_bar]; S_hat = S_hat[:N_2_bar]; V_hat = V_hat[:N_2_bar, :]; 

                        np.savetxt(filename_prefix + "noise_var_%.2f_output_size_%i_num_inputs_%i_svm_%f_aligned_%s_run_%i_y_data.csv"% (noise_var, output_size, num_inputs, singular_value_multiplier, aligned, run_i), y_data, delimiter=",")
                        np.savetxt(filename_prefix + "noise_var_%.2f_output_size_%i_num_inputs_%i_svm_%f_aligned_%s_run_%i_noisy_y_data.csv"% (noise_var, output_size, num_inputs, singular_value_multiplier, aligned, run_i), noisy_y_data, delimiter=",")
                        if input_type != "one_hot":
                          np.savetxt(filename_prefix + "noise_var_%.2f_output_size_%i_num_inputs_%i_svm_%f_aligned_%s_run_%i_x_data.csv"% (noise_var, output_size, num_inputs, singular_value_multiplier, aligned, run_i), x_data, delimiter=",")
             
                         
                        if aligned:
                            # initialize weights aligned with noisy data moddes and scale so
                            # product has singular values of epsilon
                            if not track_SVD:
                                U_hat, S_hat, V_hat = np.linalg.svd(noisy_y_data, full_matrices=False)
                                U_hat_base = U_hat; V_hat_base=V_hat;
                            W21 = np.sqrt(epsilon) * V_hat_base[:num_hidden, :]
                            W32 = np.sqrt(epsilon) * U_hat_base[:, :num_hidden]
                        else:
                            # initialize weights as random orthogonal matrices and scale so
                            # product has singular values of epsilon
                            W21 = np.sqrt(epsilon) * random_orthogonal(num_input)[:num_hidden, :] 
                            W32 = np.sqrt(epsilon) * random_orthogonal(num_output)[:, :num_hidden] 
                    
                        sigma_31 = np.matmul(noisy_y_data[:, :num_inputs], x_data[:, :num_inputs].transpose())
                        sigma_11 = np.matmul(x_data[:, :num_inputs], x_data[:, :num_inputs].transpose())

                        
                        def train_epoch():
                            global W21, W32
                            l = sigma_31 - np.matmul(np.matmul(W32, W21), sigma_11)
                            W21 += learning_rate * np.matmul(W32.transpose(), l) 
                            W32 += learning_rate * np.matmul(l, W21.transpose()) 
                            

                        def evaluate():
                            global W21, W32
                            curr_loss = np.sum(np.square(y_data - np.matmul(np.matmul(W32, W21), x_data)))
                            return curr_loss/y_data_frob_squared # appropriate normalization
                        
                            
#		    if track_SVD:
#			fsvd = open(filename_prefix + "noise_var_%.2f_output_size_%i_svm_%f_aligned_%s_run_%i_SVD_track.csv" % (noise_var, output_size, singular_value_multiplier, aligned, run_i), "w")
#			fsvd.write("epoch, " + ", ".join(["s%i"%i for i in range(1)]) + ", " + ", ".join(["U%iUhat%i" %(i,j) for i in range(1) for j in range(1)]) + "\n")
                        with open(filename_prefix + "noise_var_%.2f_output_size_%i_num_inputs_%i_svm_%f_aligned_%s_run_%i.csv" % (noise_var, output_size, num_inputs, singular_value_multiplier, aligned, run_i), "w") as fout:

                            fout.write("epoch, loss\n")
                            fout.write("%i, %f\n" % (0, evaluate()))
                            for epoch_i in xrange(1, num_epochs + 1):
                                train_epoch()	
                                curr_loss = evaluate()
                                if epoch_i % save_every == 0:
                                    fout.write("%i, %f\n" % (epoch_i, curr_loss))
                                    print("%i, %f" % (epoch_i, curr_loss))
                                    if track_SVD:
                                        curr_output = np.matmul(np.matmul(W32, W21), x_data)
                                        U, S, V = np.linalg.svd(curr_output, full_matrices=False)
                                        U = U[:, :1]; S = S[:1]; V = V[:1, :]; 
                                        norm_S = S/S_hat
                                        U_dots = np.dot(U.transpose(), U_hat).flatten()
                                        fsvd.write("%i, " % epoch_i + ", ".join(["%f" for i in range(1)]) % tuple(norm_S) + ", " + ", ".join(["%f" for i in range(1) for j in range(1)]) % tuple(U_dots) + "\n")

#		    if track_SVD:
#			fsvd.close()
        
