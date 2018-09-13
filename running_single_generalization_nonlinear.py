from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import datasets
from orthogonal_matrices import random_orthogonal
### Parameters
num_examples = 100
output_sizes = [50] #[10, 50, 100, 200, 400]
#num_layers = 4 # not actually a changeable parameter
num_layerss = [2, 4]
sigma_zs = [1.]
num_runs = 1
learning_rate = 0.001
num_hidden = 50
N_2_bars = [1, 3]
#N_2_bar = 3
batch_size = num_examples
filename_prefix = "sg_nl_results_orth_leaky/"
input_type = "one_hot" # one_hot, orthogonal, gaussian
track_SVD = True
save_every = 50
epsilon = 1e-5 
singular_value_multipliers = [30.]#[1., 2., 3., 5., 10., 20.]  
leak_alpha = 0.1

###
#var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=2*np.sqrt(epsilon), mode='FAN_AVG')

for num_layers in num_layerss:
    num_epochs = 150000 if num_layers != 2 else 10000
    eps_per = np.power(epsilon, 1./num_layers)
    for N_2_bar in N_2_bars:
        for run_i in xrange(num_runs):
            for sigma_z in sigma_zs:
                noise_var = sigma_z**2
                for output_size in output_sizes:
                    for singular_value_multiplier in singular_value_multipliers:
                        
                        scaled_noise_var = noise_var/output_size
                        np.random.seed(run_i)
                        tf.set_random_seed(run_i)
                        curr_filename = "depth_%i_n2b_%i_noise_var_%.2f_output_size_%i_svm_%f_run_%i_" %(num_layers, N_2_bar, noise_var, output_size, singular_value_multiplier, run_i)
                        print("Now running " + curr_filename)
              
                        num_input = num_examples
                        num_output = output_size
                        if input_type == "one_hot":
                            x_data, y_data, noisy_y_data, input_modes = datasets.noisy_SVD_dataset(num_examples, output_size, noise_var=scaled_noise_var, singular_value_multiplier=singular_value_multiplier, num_nonempty=N_2_bar) 
                        else:
                            x_data, y_data, noisy_y_data, input_modes = datasets.noisy_SVD_dataset_different_inputs(num_examples, output_size, noise_var=scaled_noise_var, input_type="orthogonal", singular_value_multiplier=singular_value_multiplier, num_nonempty=N_2_bar) 

                        y_data_frob_squared = np.sum(y_data**2)
                        noisy_y_data_frob_squared = np.sum(noisy_y_data**2)
              
                        if track_SVD:
                            U_bar, S_bar, V_bar = np.linalg.svd(y_data, full_matrices=False)
                            U_bar = U_bar[:, :1]; S_bar = S_bar[:1]; V_bar = V_bar[:1, :]; # save some computation later -- other singular values are zero
                            U_hat, S_hat, V_hat = np.linalg.svd(noisy_y_data, full_matrices=False)
                            U_hat_base = U_hat; V_hat_base=V_hat;
                            U_hat = U_hat[:, :1]; S_hat = S_hat[:1]; V_hat = V_hat[:1, :]; 

                        np.savetxt(filename_prefix + curr_filename+ "y_data.csv", y_data, delimiter=",")
                        np.savetxt(filename_prefix + curr_filename+ "noisy_y_data.csv", noisy_y_data, delimiter=",")
                        if input_type != "one_hot":
                            np.savetxt(filename_prefix + curr_filename + "x_data.csv", x_data, delimiter=",")
             
                        
                        input_ph = tf.placeholder(tf.float32, shape=[num_input, None])
                        target_ph = tf.placeholder(tf.float32, shape=[num_output, None])
              
                        # initialize weights as random orthogonal matrices and scale so
                        # product has singular values of epsilon
                        O1 = eps_per * random_orthogonal(num_input)[:num_hidden, :] 
                        O4 = eps_per * random_orthogonal(num_output)[:, :num_hidden] 
                        W1 = tf.get_variable('W1', shape=[num_hidden, num_input], initializer=tf.constant_initializer(O1))
                        W4 = tf.get_variable('W4', shape=[num_output, num_hidden], initializer=tf.constant_initializer(O4))
                        hidden = tf.nn.leaky_relu(tf.matmul(W1, input_ph), alpha=leak_alpha)

                        if num_layers == 4:
                            O2 = eps_per * random_orthogonal(num_hidden)
                            O3 = eps_per * random_orthogonal(num_hidden)
                            W2 = tf.get_variable('W2', shape=[num_hidden, num_hidden], initializer=tf.constant_initializer(O2))
                            W3 = tf.get_variable('W3', shape=[num_hidden, num_hidden], initializer=tf.constant_initializer(O3))
                        
                            hidden = tf.nn.leaky_relu(tf.matmul(W2, hidden), alpha=leak_alpha)
                            hidden = tf.nn.leaky_relu(tf.matmul(W3, hidden), alpha=leak_alpha)

                        output = (tf.matmul(W4, hidden))
                        
                        loss = tf.reduce_sum(tf.square(output-target_ph))#2*tf.nn.l2_loss(output - target_ph)
                        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                        train = optimizer.minimize(loss)	
                        
                        with tf.Session() as sess:
                            def train_epoch():
                                for batch_i in xrange(num_examples//batch_size):
                                    sess.run(train, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :].transpose(), target_ph: noisy_y_data[batch_i:batch_i+batch_size, :].transpose()})
              
                            def evaluate():
                                curr_loss = 0.
                                curr_tr_loss = 0.
                                for batch_i in xrange(num_examples//batch_size):
                                    curr_loss += sess.run(loss, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :].transpose(), target_ph: y_data[batch_i:batch_i+batch_size, :].transpose()})
                                    curr_tr_loss += sess.run(loss, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :].transpose(), target_ph: noisy_y_data[batch_i:batch_i+batch_size, :].transpose()})
                                return curr_loss/y_data_frob_squared, curr_tr_loss/noisy_y_data_frob_squared # appropriate normalization
                            
                            sess.run(tf.global_variables_initializer())
                                
#                    if track_SVD:
#                        fsvd = open(filename_prefix + "noise_var_%.2f_output_size_%i_run_%i_SVD_track.csv" % (noise_var, output_size, run_i), "w")
#                        fsvd.write("epoch, " + ", ".join(["s%i"%i for i in range(1)]) + ", " + ", ".join(["U%iUhat%i" %(i,j) for i in range(1) for j in range(1)]) + "\n")
                            with open(filename_prefix + curr_filename + ".csv", "w") as fout:

                                fout.write("epoch, loss, train_loss\n")
                                curr_loss, curr_tr_loss = evaluate()
                                fout.write("%i, %f, %f\n" % (0, curr_loss, curr_tr_loss))
                                for epoch_i in xrange(1, num_epochs + 1):
                                    train_epoch()	
                                    if epoch_i % save_every == 0:
                                        curr_loss, curr_tr_loss = evaluate()
                                        fout.write("%i, %f, %f\n" % (epoch_i, curr_loss, curr_tr_loss))
                                        print("%i, %f, %f" % (epoch_i, curr_loss, curr_tr_loss))
#                                if track_SVD:
#                                    curr_output = sess.run(output, feed_dict={input_ph: x_data.transpose()})
#                                    U, S, V = np.linalg.svd(curr_output, full_matrices=False)
#                                    U = U[:, :1]; S = S[:1]; V = V[:1, :]; 
#                                    norm_S = S/S_hat
#                                    U_dots = np.dot(U.transpose(), U_hat).flatten()
#                                    fsvd.write("%i, " % epoch_i + ", ".join(["%f" for i in range(1)]) % tuple(norm_S) + ", " + ", ".join(["%f" for i in range(1) for j in range(1)]) % tuple(U_dots) + "\n")

#                    if track_SVD:
#                        fsvd.close()
              
                        tf.reset_default_graph()
