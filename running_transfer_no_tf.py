from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import datasets
from orthogonal_matrices import random_orthogonal
### Parameters
num_examples = 100
output_sizes = [num_examples] #[10, 50, 100, 200, 400]
sigma_zs = [1] 

num_runs = 5
learning_rate = 0.001
num_epochs = 5000
batch_size = num_examples
filename_prefix = "transfer_results/"
#input_type = "one_hot" # one_hot, orthogonal, gaussian
#track_SVD = False
save_every = 5
epsilon = 1e-5
singular_value_multiplier = 10 
N_2_bar = 1 # rank of teacher
qs = [1, 0.5,  0]
singular_value_1_multipliers = [float(x) for x in [1, 2, 4, 10]]
singular_value_2_multipliers = [float(x) for x in [2, 1, 10]]
alignments = [True] # if false, run with random inits
num_hidden = num_examples

###
#var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=2*np.sqrt(epsilon), mode='FAN_AVG')

for run_i in xrange(num_runs):
    for sigma_z in sigma_zs:
        noise_var = sigma_z**2
        for q in qs: 
            for svm1 in singular_value_1_multipliers:
                for svm2 in singular_value_2_multipliers:
                    for aligned in alignments:
                        for output_size in output_sizes:
                            
                            scaled_noise_var = noise_var/output_size
                            np.random.seed(run_i)
                            print("Now running noise_var: %.2f, output_size: %i, svm1: %f, alignment: %s, run: %i" % (noise_var, output_size, svm1, aligned, run_i))

                            curr_filename = "noise_var_%.2f_output_size_%i_q_%f_svm1_%f_svm2_%f_aligned_%s_run_%i_"% (noise_var, output_size, q, svm1, svm2, aligned, run_i)
                  
                            num_input = num_examples
                            num_output = output_size
                            x_data, y_data, noisy_y_data = datasets.noisy_rank_one_correlated_dataset(num_examples, output_size, q, svm1=svm1, svm2=svm2, noise_var=scaled_noise_var) 


                            noisy_y_data = noisy_y_data.transpose()
                            y_data = y_data.transpose()

                            t1_y_data = y_data[:100, :]
                            t2_y_data = y_data[100:, :]
                            t1_noisy_y_data = noisy_y_data[:100, :]
                            t2_noisy_y_data = noisy_y_data[100::, :]

                            y_data_frob_squared = np.sum(y_data**2)
                            t1_y_data_frob_squared = np.sum(t1_y_data**2)
                            t2_y_data_frob_squared = np.sum(t2_y_data**2)
                  
#		    if track_SVD:
#			U_bar, S_bar, V_bar = np.linalg.svd(y_data, full_matrices=False)
#			U_bar = U_bar[:, :N_2_bar]; S_bar = S_bar[:N_2_bar]; V_bar = V_bar[:N_2_bar, :]; # save some computation later -- other singular values are zero
#			U_hat, S_hat, V_hat = np.linalg.svd(noisy_y_data, full_matrices=False)
#			U_hat_base = U_hat; V_hat_base=V_hat;
#			U_hat = U_hat[:, :N_2_bar]; S_hat = S_hat[:N_2_bar]; V_hat = V_hat[:N_2_bar, :]; 

                            np.savetxt(filename_prefix + curr_filename + "y_data.csv", y_data, delimiter=",")
                            np.savetxt(filename_prefix + curr_filename + "noisy_y_data.csv", noisy_y_data, delimiter=",")
#		    if input_type != "one_hot":
#		      np.savetxt(filename_prefix + "noise_var_%.2f_output_size_%i_svm_%f_aligned_%s_run_%i_x_data.csv"% (noise_var, output_size, singular_value_multiplier, aligned, run_i), x_data, delimiter=",")
                 
                             
                            if aligned:
                                # initialize weights aligned with noisy data moddes and scale so
                                # product has singular values of epsilon
                                U_hat, S_hat, V_hat = np.linalg.svd(noisy_y_data, full_matrices=False)
                                U_hat_base = U_hat; V_hat_base=V_hat;
                                W21 = np.sqrt(epsilon) * V_hat_base[:num_hidden, :]
                                W32 = np.sqrt(epsilon) * U_hat_base[:, :num_hidden]

                                U_hat, S_hat, V_hat = np.linalg.svd(t1_noisy_y_data, full_matrices=False)
                                U_hat_base = U_hat; V_hat_base=V_hat;
                                t1_W21 = np.sqrt(epsilon) * V_hat_base[:num_hidden, :]
                                t1_W32 = np.sqrt(epsilon) * U_hat_base[:, :num_hidden]
                                U_hat, S_hat, V_hat = np.linalg.svd(t2_noisy_y_data, full_matrices=False)
                                U_hat_base = U_hat; V_hat_base=V_hat;
                                t2_W21 = np.sqrt(epsilon) * V_hat_base[:num_hidden, :]
                                t2_W32 = np.sqrt(epsilon) * U_hat_base[:, :num_hidden]
                            else:
                                # initialize weights as random orthogonal matrices and scale so
                                # product has singular values of epsilon
                                W21 = np.sqrt(epsilon) * random_orthogonal(num_input)[:num_hidden, :] 
                                W32 = np.sqrt(epsilon) * random_orthogonal(2*num_output)[:, :num_hidden] 

                                t1_W21 = W21[:, :] 
                                t1_W32 = W32[:num_output, :] 
                                t2_W21 = W21[:, :] 
                                t2_W32 = W32[num_output:, :]
                        
                            sigma_31 = np.matmul(noisy_y_data, x_data.transpose())
                            sigma_11 = np.matmul(x_data, x_data.transpose())

                            t1_sigma_31 = np.matmul(t1_noisy_y_data, x_data.transpose())
                            t2_sigma_31 = np.matmul(t2_noisy_y_data, x_data.transpose())

                            def train_epoch():
                                global W21, W32, t1_W21, t1_W32, t2_W21, t2_W32
                                l = sigma_31 - np.matmul(np.matmul(W32, W21), sigma_11)
                                W21 += learning_rate * np.matmul(W32.transpose(), l) 
                                W32 += learning_rate * np.matmul(l, W21.transpose()) 
                                
                                t1_l = t1_sigma_31 - np.matmul(np.matmul(t1_W32, t1_W21), sigma_11)
                                t1_W21 += learning_rate * np.matmul(t1_W32.transpose(), t1_l) 
                                t1_W32 += learning_rate * np.matmul(t1_l, t1_W21.transpose()) 
                                t2_l = t2_sigma_31 - np.matmul(np.matmul(t2_W32, t2_W21), sigma_11)
                                t2_W21 += learning_rate * np.matmul(t2_W32.transpose(), t2_l) 
                                t2_W32 += learning_rate * np.matmul(t2_l, t2_W21.transpose()) 

                            def evaluate():
                                curr_loss = np.sum(np.square(y_data - np.matmul(np.matmul(W32, W21), x_data)))
                                t1_curr_loss = np.sum(np.square(t1_y_data - np.matmul(np.matmul(t1_W32, t1_W21), x_data)))
                                t2_curr_loss = np.sum(np.square(t2_y_data - np.matmul(np.matmul(t2_W32, t2_W21), x_data)))
                                t1_joint_curr_loss = np.sum(np.square(y_data[:100, :] - np.matmul(np.matmul(W32, W21), x_data)[:100, :]))
                                t2_joint_curr_loss = np.sum(np.square(y_data[100:, :] - np.matmul(np.matmul(W32, W21), x_data)[100:, :]))
                                print(curr_loss, t1_curr_loss, t2_curr_loss, t1_joint_curr_loss, t2_joint_curr_loss)
                                exit()
                                symm_ben = (t1_curr_loss + t2_curr_loss - curr_loss)/y_data_frob_squared
                                t1_ben = (t1_curr_loss - t1_joint_curr_loss)/t1_y_data_frob_squared
                                t2_ben = (t2_curr_loss - t2_joint_curr_loss)/t2_y_data_frob_squared
                                curr_norm_loss = curr_loss/y_data_frob_squared
                                
                                return curr_norm_loss, symm_ben, t1_ben, t2_ben  # appropriate normalization
                            
                                
#                        if track_SVD:
#                            fsvd = open(filename_prefix + "noise_var_%.2f_output_size_%i_svm_%f_aligned_%s_run_%i_SVD_track.csv" % (noise_var, output_size, singular_value_multiplier, aligned, run_i), "w")
#                            fsvd.write("epoch, " + ", ".join(["s%i"%i for i in range(1)]) + ", " + ", ".join(["U%iUhat%i" %(i,j) for i in range(1) for j in range(1)]) + "\n")
                            with open(filename_prefix + curr_filename + ".csv", "w") as fout:

                                fout.write("epoch, norm_loss, symm_ben, t1_ben, t2_ben\n")
                                cnl, sb, t1b, t2b = evaluate()
                                fout.write("%i, %f, %f, %f, %f\n" % (0, cnl, sb, t1b, t2b))
                                for epoch_i in xrange(1, num_epochs + 1):
                                    train_epoch()	
                                    curr_loss = evaluate()
                                    if epoch_i % save_every == 0:
                                        cnl, sb, t1b, t2b = evaluate()
                                        fout.write("%i, %f, %f, %f, %f\n" % (epoch_i, cnl, sb, t1b, t2b))
                                        print("%i, %f, %f, %f, %f\n" % (epoch_i, cnl, sb, t1b, t2b))
#                                    if track_SVD:
#                                        curr_output = np.matmul(np.matmul(W32, W21), x_data)
#                                        U, S, V = np.linalg.svd(curr_output, full_matrices=False)
#                                        U = U[:, :1]; S = S[:1]; V = V[:1, :]; 
#                                        norm_S = S/S_hat
#                                        U_dots = np.dot(U.transpose(), U_hat).flatten()
#                                        fsvd.write("%i, " % epoch_i + ", ".join(["%f" for i in range(1)]) % tuple(norm_S) + ", " + ", ".join(["%f" for i in range(1) for j in range(1)]) % tuple(U_dots) + "\n")

#                        if track_SVD:
#                            fsvd.close()
            
