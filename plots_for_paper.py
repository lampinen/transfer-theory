import numpy as np
import datasets

from orthogonal_matrices import random_orthogonal
from theory_functions import *



# Figure 1
##  figure 1a
ts = np.arange(0, 10, 0.05)
s10 = np.array([s_of_t([10], t, 0.001, 1) for t in ts])
s7 = np.array([s_of_t([7], t, 0.001, 1) for t in ts])
s4 = np.array([s_of_t([4], t, 0.001, 1) for t in ts])
s1 = np.array([s_of_t([1], t, 0.001, 1) for t in ts])


with open("figure_stuff/s_of_t_a.csv", "w") as fout:
    fout.write("t, s_hat, s_of_t\n")
    for i in range(len(ts)):
        fout.write("%f, %f, %f\n" % (ts[i], 10, s10[i]))
        fout.write("%f, %f, %f\n" % (ts[i], 7, s7[i]))
        fout.write("%f, %f, %f\n" % (ts[i], 4, s4[i]))
        fout.write("%f, %f, %f\n" % (ts[i], 1, s1[i]))


## figure 1b
ss = np.arange(0, 10, 0.05)
t5 = np.array(s_of_t(ss, 5, 0.001, 1))
t4 = np.array(s_of_t(ss, 4, 0.001, 1))
t3 = np.array(s_of_t(ss, 3, 0.001, 1))
t2 = np.array(s_of_t(ss, 2, 0.001, 1))
t1 = np.array(s_of_t(ss, 1, 0.001, 1))
t05 = np.array(s_of_t(ss, 0.5, 0.001, 1))

with open("figure_stuff/s_of_t_b.csv", "w") as fout:
    fout.write("t, s_hat, s_of_t\n")
    for i in range(len(ss)):
        fout.write("%f, %f, %f\n" % (5, ss[i],  t5[i]))
        fout.write("%f, %f, %f\n" % (4, ss[i],  t4[i]))
        fout.write("%f, %f, %f\n" % (3, ss[i],  t3[i]))
        fout.write("%f, %f, %f\n" % (2, ss[i],  t2[i]))
        fout.write("%f, %f, %f\n" % (1, ss[i],  t1[i]))
        fout.write("%f, %f, %f\n" % (0.5, ss[i],  t05[i]))
