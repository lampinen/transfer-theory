from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import datasets
PI = np.pi
### Parameters
num_examples = 20
output_size = 100
num_domains = [2]
qs = [0, PI/16, PI/8, PI/4, PI/2, PI] 
noise_var = 0.025 # independent gaussian per output
num_runs = 10 
learning_rate = 0.01
num_epochs = 1000
batch_size = 1
filename_prefix = "shared_input_mode_results/"
save_every = 10

###

for run_i in xrange(num_runs):
  for num_dom in num_domains:
      for q in qs:
          print("Now running q: %.2f, num_dom: %i, run: %i" % (q, num_dom, run_i))
          num_input = num_examples
          num_output = output_size * num_dom
          num_hidden = num_input 
          x_data, y_data, noisy_y_data, input_modes = datasets.noisy_shared_input_modes_dataset(num_examples, output_size, num_dom, q, noise_var=noise_var) 
          
          input_ph = tf.placeholder(tf.float32, shape=[num_input, None])
          target_ph = tf.placeholder(tf.float32, shape=[num_output, None])

          W1 = tf.get_variable('W1', shape=[num_hidden, num_input], initializer=tf.contrib.layers.xavier_initializer())	
          W2 = tf.get_variable('W2', shape=[num_output, num_hidden], initializer=tf.contrib.layers.xavier_initializer())	
      
          hidden = tf.matmul(W1, input_ph)
          output = tf.matmul(W2, hidden)
          
          loss = tf.nn.l2_loss(output - target_ph)
          optimizer = tf.train.GradientDescentOptimizer(learning_rate)
          train = optimizer.minimize(loss)	
          
          with tf.Session() as sess:
              def train_epoch():
                  for batch_i in xrange(num_examples//batch_size):
                      sess.run(train, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :].reshape([num_input, 1]), target_ph: noisy_y_data[batch_i:batch_i+batch_size, :].reshape([num_output, 1])})

              def evaluate():
                  curr_loss = 0.
                  for batch_i in xrange(num_examples//batch_size):
                      curr_loss += sess.run(loss, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :].reshape([num_input, 1]), target_ph: y_data[batch_i:batch_i+batch_size, :].reshape([num_output, 1])})
                  return curr_loss
              
              sess.run(tf.global_variables_initializer())
                  
              with open(filename_prefix + "loss_ndom%i_q%.2f_run%i.csv" % (num_dom, q, run_i), "w") as fout:
                  fout.write("epoch, loss\n")
                  fout.write("%i, %f\n" % (0, evaluate()))
                  for epoch_i in xrange(1, num_epochs + 1):
                      train_epoch()	
                      curr_loss = evaluate()
                      if epoch_i % save_every == 0:
                          fout.write("%i, %f\n" % (epoch_i, curr_loss))
                          print("%i, %f" % (epoch_i, curr_loss))

          tf.reset_default_graph()
