from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import datasets
PI = np.pi
### Parameters
num_examples = 100
output_size = 50
num_domains = [2]
qs = [0.5, 1.]#np.arange(0, 1.05, 0.1) 
svm1s = [0.84, 100, 3]
svm2s = [0.84, 100, 3]
sigma_z = 1.
num_runs = 1 
learning_rate = 0.001
num_epochs = 5000
batch_size = num_examples
filename_prefix = "paper_nonlinear_transfer_results/"
input_type = "one_hot" # one_hot, orthogonal, gaussian
rank = 4 # teacher rank
save_every = 5

###
#TODO: replace
var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=np.sqrt(0.0001), mode='FAN_AVG')



for run_i in xrange(num_runs):
    for num_dom in num_domains:
        for q in qs:
            for svm1 in svm1s:
                for svm2 in svm2s:
                    np.random.seed(run_i)
                    tf.set_random_seed(run_i)
                    print("Now running q: %.2f, num_dom: %i, run: %i" % (q, num_dom, run_i))
                    num_input = num_examples
                    noise_var = sigma_z**2 / num_input
                    num_output = output_size * num_dom
                    num_hidden = num_input 
                    if input_type == "one_hot":
                        x_data, y_data, noisy_y_data = datasets.noisy_rank_k_correlated_dataset(num_examples, output_size, q=q, svm1=svm1, svm2=svm2, noise_var=noise_var, k=rank) 
                    else: 
                        raise ValueError("Not implemented")


                    
                    input_ph = tf.placeholder(tf.float32, shape=[num_input, None])
                    target_ph = tf.placeholder(tf.float32, shape=[num_output, None])

                    W1 = tf.get_variable('W1', shape=[num_hidden, num_input], initializer=var_scale_init)
                    W2 = tf.get_variable('W2', shape=[num_output, num_hidden], initializer=var_scale_init)
                
                    hidden = tf.nn.relu(tf.matmul(W1, input_ph))
                    output = tf.nn.relu(tf.matmul(W2, hidden))
                    
                    loss = tf.nn.l2_loss(output - target_ph)
                    first_domain_loss = tf.nn.l2_loss(output[:output_size, :] - target_ph[:output_size, :])
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                    train = optimizer.minimize(loss)	

                    W11 = tf.get_variable('W11', shape=[num_hidden, num_input], initializer=var_scale_init)
                    W12 = tf.get_variable('W12', shape=[output_size, num_hidden], initializer=var_scale_init)
                
                    hidden1 = tf.nn.relu(tf.matmul(W11, input_ph))
                    output1 = tf.nn.relu(tf.matmul(W12, hidden1))
                    
                    loss1 = tf.nn.l2_loss(output1 - target_ph[:output_size, :])
                    train1 = optimizer.minimize(loss1)	

                    curr_filename = "noise_var_%.2f_output_size_%i_q_%f_svm1_%f_svm2_%f_aligned_%s_run_%i_"% (noise_var, output_size, q, svm1, svm2, False, run_i)

                    t1_y_data = y_data[:, :output_size]
		    y_data_frob_squared = np.sum(y_data**2)
		    t1_y_data_frob_squared = np.sum(t1_y_data**2)

                    with tf.Session() as sess:
                        def train_epoch():
                            for batch_i in xrange(num_examples//batch_size):
                                sess.run(train, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :].transpose(), target_ph: noisy_y_data[batch_i:batch_i+batch_size, :].transpose()})
                                sess.run(train1, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :].transpose(), target_ph: noisy_y_data[batch_i:batch_i+batch_size, :].transpose()})

                        def evaluate():
                            curr_loss = 0.
                            curr_loss1 = 0.
                            for batch_i in xrange(num_examples//batch_size):
                                curr_loss += sess.run(first_domain_loss, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :].transpose(), target_ph: y_data[batch_i:batch_i+batch_size, :].transpose()})
                                curr_loss1 += sess.run(loss1, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :].transpose(), target_ph: y_data[batch_i:batch_i+batch_size, :].transpose()})
                            return curr_loss/t1_y_data_frob_squared, curr_loss1/t1_y_data_frob_squared
                        
                        sess.run(tf.global_variables_initializer())
                            
                        with open(filename_prefix + curr_filename + ".csv", "w") as fout:
                            fout.write("epoch, loss_joint, loss_sep\n")
                            loss_joint, loss_sep = evaluate()
                            fout.write("%i, %f, %f\n" % (0, loss_joint, loss_sep))
                            for epoch_i in xrange(1, num_epochs + 1):
                                train_epoch()	
                                curr_loss = evaluate()
                                if epoch_i % save_every == 0:
                                    loss_joint, loss_sep = evaluate()
                                    fout.write("%i, %f, %f\n" % (epoch_i, loss_joint, loss_sep))
                                    print("%i, %f, %f\n" % (epoch_i, loss_joint, loss_sep))

                    tf.reset_default_graph()
