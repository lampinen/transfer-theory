from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import datasets
### Parameters
num_examples = 30
output_sizes = [10, 50, 100, 200, 400]
noise_vars = [0., 0.025, 0.05, 0.1, 0.2] # independent gaussian per output
num_runs = 10 
learning_rate = 0.001
num_epochs = 5000
batch_size = 30
filename_prefix = "single_generalization_ortho_inputs_results/"
input_type = "orthogonal" # one_hot, orthogonal, gaussian
save_every = 5

###
var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=0.1, mode='FAN_AVG')

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
                  
              with open(filename_prefix + "noise_var_%.2f_output_size_%i_run_%i.csv" % (noise_var, output_size, run_i), "w") as fout:
                  fout.write("epoch, loss\n")
                  fout.write("%i, %f\n" % (0, evaluate()))
                  for epoch_i in xrange(1, num_epochs + 1):
                      train_epoch()	
                      curr_loss = evaluate()
                      if epoch_i % save_every == 0:
                          fout.write("%i, %f\n" % (epoch_i, curr_loss))
                          print("%i, %f" % (epoch_i, curr_loss))

          tf.reset_default_graph()
