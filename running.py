from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import datasets

### Parameters
num_examples_per = 20
output_size = 100
correlations = [1, 0.5, 0, -0.5, -1]
num_domains = [2]
num_runs = 100 
learning_rate = 0.01
num_epochs = 2000
batch_size = 1
filename_prefix = "correlation_results/"

###

for run_i in xrange(num_runs):
  for num_dom in num_domains:
      num_input = num_examples_per
      num_output = output_size * num_dom
      num_hidden = num_input 
      for correlation in correlations:
	  x_data, y_data = datasets.correlated_dataset(num_examples_per, output_size, num_dom, correlation) 
	  
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
		  for batch_i in xrange(num_examples_per//batch_size):
		      sess.run(train, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :].reshape([num_input, 1]), target_ph: y_data[batch_i:batch_i+batch_size, :].reshape([num_output, 1])})

	      def evaluate():
		  curr_loss = 0.
		  for batch_i in xrange(num_examples_per//batch_size):
		      curr_loss += sess.run(loss, feed_dict={input_ph: x_data[batch_i:batch_i+batch_size, :].reshape([num_input, 1]), target_ph: y_data[batch_i:batch_i+batch_size, :].reshape([num_output, 1])})
		  return curr_loss
	      
	      sess.run(tf.global_variables_initializer())
		  
	      with open(filename_prefix + "loss_ndom%i_correlation%f_run%i.csv" % (num_dom, correlation, run_i), "w") as fout:
		  fout.write("epoch, loss\n")
		  fout.write("%i, %f\n" % (0, evaluate()))
		  for epoch_i in xrange(1, num_epochs + 1):
		      train_epoch()	
		      curr_loss = evaluate()
		      fout.write("%i, %f\n" % (epoch_i, curr_loss))
		      print("%i, %f" % (epoch_i, curr_loss))

	  tf.reset_default_graph()
