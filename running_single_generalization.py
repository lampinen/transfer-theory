from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import datasets
### Parameters
num_examples = 30
output_sizes = [100] #[10, 50, 100, 200, 400]
noise_vars = [0.1] #[0., 0.025, 0.05, 0.1, 0.2] # independent gaussian per output
num_runs = 10 
learning_rate = 0.001
num_epochs = 5000
batch_size = 30
filename_prefix = "single_generalization_SVD_results/"
input_type = "one_hot" # one_hot, orthogonal, gaussian
track_SVD = True
save_every = 5

###
var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=0.001, mode='FAN_AVG')

for run_i in xrange(num_runs):
    for noise_var in noise_vars:
        for output_size in output_sizes:
            np.random.seed(run_i)
            tf.set_random_seed(run_i)
            print("Now running noise_var: %.2f, output_size: %i, run: %i" % (noise_var, output_size, run_i))
  
            num_input = num_examples
            num_output = output_size
            num_hidden = num_input 
            if input_type == "one_hot":
                x_data, y_data, noisy_y_data, input_modes = datasets.noisy_SVD_dataset(num_examples, output_size, noise_var=noise_var) 
            else:
                x_data, y_data, noisy_y_data, input_modes = datasets.noisy_SVD_dataset_different_inputs(num_examples, output_size, noise_var=noise_var, input_type="orthogonal") 
  
            if track_SVD:
                U_bar, S_bar, V_bar = np.linalg.svd(y_data, full_matrices=False)
                U_bar = U_bar[:, :4]; S_bar = S_bar[:4]; V_bar = V_bar[:4, :]; # save some computation later -- other singular values are zero
                U_hat, S_hat, V_hat = np.linalg.svd(noisy_y_data, full_matrices=False)
                U_hat = U_hat[:, :4]; S_hat = S_hat[:4]; V_hat = V_hat[:4, :]; 
  
            
            input_ph = tf.placeholder(tf.float32, shape=[num_input, None])
            target_ph = tf.placeholder(tf.float32, shape=[num_output, None])
  
            W1 = tf.get_variable('W1', shape=[num_hidden, num_input], initializer=var_scale_init)
            W2 = tf.get_variable('W2', shape=[num_output, num_hidden], initializer=var_scale_init)
        
            hidden = tf.matmul(W1, input_ph)
            output = tf.matmul(W2, hidden)
            
            loss = tf.nn.l2_loss(output - target_ph)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train = optimizer.minimize(loss)	
            
            with tf.Session() as sess:
                def train_epoch():
                    for batch_i in xrange(num_examples//batch_size):
                        sess.run(train, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :].transpose(), target_ph: noisy_y_data[batch_i:batch_i+batch_size, :].transpose()})
  
                def evaluate():
                    curr_loss = 0.
                    for batch_i in xrange(num_examples//batch_size):
                        curr_loss += sess.run(loss, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :].transpose(), target_ph: y_data[batch_i:batch_i+batch_size, :].transpose()})
                    return curr_loss
                
                sess.run(tf.global_variables_initializer())
                    
                if track_SVD:
                    fsvd = open(filename_prefix + "noise_var_%.2f_output_size_%i_run_%i_SVD_track.csv" % (noise_var, output_size, run_i), "w")
                    fsvd.write("epoch, " + ", ".join(["s%i"%i for i in range(4)]) + ", " + ", ".join(["U%iUhat%i" %(i,j) for i in range(4) for j in range(4)]) + "\n")
                with open(filename_prefix + "noise_var_%.2f_output_size_%i_run_%i.csv" % (noise_var, output_size, run_i), "w") as fout:

                    fout.write("epoch, loss\n")
                    fout.write("%i, %f\n" % (0, evaluate()))
                    for epoch_i in xrange(1, num_epochs + 1):
                        train_epoch()	
                        curr_loss = evaluate()
                        if epoch_i % save_every == 0:
                            fout.write("%i, %f\n" % (epoch_i, curr_loss))
                            print("%i, %f" % (epoch_i, curr_loss))
                            if track_SVD:
                                curr_output = sess.run(output, feed_dict={input_ph: x_data.transpose()})
                                U, S, V = np.linalg.svd(curr_output.transpose(), full_matrices=False)
                                U = U[:, :4]; S = S[:4]; V = V[:4, :]; 
                                norm_S = S/S_hat
                                U_dots = np.dot(U.transpose(), U_hat).flatten()
                                fsvd.write("%i, " % epoch_i + ", ".join(["%f" for i in range(4)]) % tuple(norm_S) + ", " + ", ".join(["%f" for i in range(4) for j in range(4)]) % tuple(U_dots) + "\n")

                if track_SVD:
                    fsvd.close()
  
            tf.reset_default_graph()
