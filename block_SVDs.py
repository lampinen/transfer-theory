from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import matplotlib.pyplot as plot
from datasets import shared_input_modes_dataset
from orthogonal_matrices import random_orthogonal

num_inputs = 5
q = np.pi/4 
_, M, im = shared_input_modes_dataset(num_inputs, num_inputs, 2, q, num_inputs)

A = M[:,:num_inputs]
B =  M[:,num_inputs:]

Y = np.concatenate([A, B], 1) # Y == M
blah = np.concatenate([A, -B], 1)
Y2 = np.concatenate([Y, blah], 0)

UY, SY, VY = np.linalg.svd(Y.transpose())
Ublah, Sblah, Vblah = np.linalg.svd(blah.transpose())
UY2, SY2, VY2 = np.linalg.svd(Y2.transpose())
print(SY)
print(Sblah)
print(SY2)

# VY modes from first half of VY2 modes
print("VY from VY2 first half")
sets = [np.concatenate([VY[np.newaxis, i, :], VY2[np.newaxis, 2*i, :num_inputs], VY2[np.newaxis, (2*i) + 1, :num_inputs]],0) for i in xrange(num_inputs)]
print([np.linalg.matrix_rank(X) for X in sets]) # what is the rank of these sets?

# input modes from first half of VY2 modes
print("Input from VY2 first half")
sets = [np.concatenate([im[np.newaxis, i, :], VY2[np.newaxis, 2*i, :num_inputs], VY2[np.newaxis, (2*i) + 1, :num_inputs]],0) for i in xrange(num_inputs)]
print([np.linalg.matrix_rank(X) for X in sets]) # what is the rank of these sets?
print([np.linalg.svd(X)[1] for X in sets]) # (sometimes SVD vals exceed numerical cutoff but are still 10^-15)

# VY2 modes first half from second half
print("VY2 halves")
sets = [np.concatenate([VY2[np.newaxis, 2*i, :num_inputs], VY2[np.newaxis, (2*i) + 1, :num_inputs], VY2[np.newaxis, 2*i, num_inputs:], VY2[np.newaxis, (2*i) + 1, num_inputs:]],0) for i in xrange(num_inputs)]
print([np.linalg.matrix_rank(X) for X in sets]) # what is the rank of these sets?

#plot.figure()
#plot.imshow(im)
plot.figure()
plot.imshow(VY)
plot.figure()
#plot.imshow(Vblah)
#plot.figure()
plot.imshow(VY2)
plot.show()

