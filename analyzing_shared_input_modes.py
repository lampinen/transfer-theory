from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import datasets

PI = np.pi
#
qs = [0, PI/16, PI/8, PI/4, PI/2, PI, 3*PI/2]
num_datasets_per = 100
num_domains = [2, 10]
output_size = 100
num_examples = 30
noise_var = 0.1
#

with open("shared_mode_analysis/mode_strengths.csv", "w") as fout:
    fout.write("q, dataset, num_domains, mode, dom1_value, shared_value, dom1_dot_shared\n")
    for q in qs:
        for n_dom in num_domains:
            for dataset_i in range(num_datasets_per):
                x_data, y_data, noisy_y_data, input_modes = datasets.noisy_shared_input_modes_dataset(num_examples, output_size, n_dom, q, noise_var=noise_var)
                U_dom1, S_dom1, V_dom1 = np.linalg.svd(y_data[:, :output_size])
                U_shared, S_shared, V_shared = np.linalg.svd(y_data)
                overlaps = np.abs(np.diagonal(np.matmul(U_dom1.transpose(), U_shared)))
                for i in xrange(len(S_dom1)):
                    fout.write("%.2f, %i, %i, %i, %f, %f, %f\n" % (q, dataset_i, n_dom, i, S_dom1[i], S_shared[i], overlaps[i]))

