import numpy as np
import datasets

from orthogonal_matrices import random_orthogonal
from theory_functions import *



# Figure 2
## figure 2cd

step=0.05
sbars = np.arange(0, 5, step)
C_A1 = get_noise_multiplier(sbars, 1., 1.) 
C_A075 = get_noise_multiplier(sbars, 1., 0.75) 
C_A05 = get_noise_multiplier(sbars, 1., 0.5) 
C_A025 = get_noise_multiplier(sbars, 1., 0.25) 
sqrt_10 = np.sqrt(10)
shat_A1 = s_hat_by_A(sbars, 1) 
shat_A075 = s_hat_by_A(sbars, 0.75) 
shat_A05 = s_hat_by_A(sbars, 0.5) 
shat_A025 = s_hat_by_A(sbars, 0.25) 


with open("figure_stuff/figure_2d.csv", "w") as fout:
    with open("figure_stuff/figure_2c.csv", "w") as fout2:
        fout2.write("sbar, A, S_hat, S_emp_mean, S_emp_se\n")
        fout.write("sbar, A, C, C_emp_mean, C_emp_se\n")
        for i in range(len(sbars)):
            if i % (len(sbars)//10) == 0:
                Os = []
                Ss = []
                for _ in range(10): 
                    _, y_data, noisy_y_data, _ = datasets.noisy_SVD_dataset(1000, int(1000*1), noise_var=1./(1000), singular_value_multiplier=sbars[i], num_nonempty=1)
                    U, S, V = np.linalg.svd(y_data, full_matrices=False)
                    U_hat, S_hat, V_hat = np.linalg.svd(noisy_y_data, full_matrices=False)
                    O = np.dot(U[:, :1].transpose(), U_hat[:,:1])
                    O *= np.dot(V[:1, :], V_hat[:1,:].transpose())
                    Os.append(O)
                    Ss.append(S_hat[0])
                fout.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 1, C_A1[i], np.mean(Os), np.std(Os)/sqrt_10))
                fout2.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 1, shat_A1[i], np.mean(Ss), np.std(Ss)/sqrt_10))
                Os = []
                Ss = []
                for _ in range(10): 
                    _, y_data, noisy_y_data, _ = datasets.noisy_SVD_dataset(1000, int(1000*0.75), noise_var=1./(1000), singular_value_multiplier=sbars[i], num_nonempty=1)
                    U, S, V = np.linalg.svd(y_data, full_matrices=False)
                    U_hat, S_hat, V_hat = np.linalg.svd(noisy_y_data, full_matrices=False)
                    O = np.dot(U[:, :1].transpose(), U_hat[:,:1])
                    O *= np.dot(V[:1, :], V_hat[:1,:].transpose())
                    Os.append(O)
                    Ss.append(S_hat[0])
                fout.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 0.75, C_A075[i], np.mean(Os), np.std(Os)/sqrt_10))
                fout2.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 0.75, shat_A075[i], np.mean(Ss), np.std(Ss)/sqrt_10))
                Os = []
                Ss = []
                for _ in range(10): 
                    _, y_data, noisy_y_data, _ = datasets.noisy_SVD_dataset(1000, int(1000*0.5), noise_var=1./(1000), singular_value_multiplier=sbars[i], num_nonempty=1)
                    U, S, V = np.linalg.svd(y_data, full_matrices=False)
                    U_hat, S_hat, V_hat = np.linalg.svd(noisy_y_data, full_matrices=False)
                    O = np.dot(U[:, :1].transpose(), U_hat[:,:1])
                    O *= np.dot(V[:1, :], V_hat[:1,:].transpose())
                    Os.append(O)
                    Ss.append(S_hat[0])
                fout.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 0.5, C_A05[i], np.mean(Os), np.std(Os)/sqrt_10))
                fout2.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 0.5, shat_A05[i], np.mean(Ss), np.std(Ss)/sqrt_10))
                Os = []
                Ss = []
                for _ in range(10): 
                    _, y_data, noisy_y_data, _ = datasets.noisy_SVD_dataset(1000, int(1000*0.25), noise_var=1./(1000), singular_value_multiplier=sbars[i], num_nonempty=1)
                    U, S, V = np.linalg.svd(y_data, full_matrices=False)
                    U_hat, S_hat, V_hat = np.linalg.svd(noisy_y_data, full_matrices=False)
                    O = np.dot(U[:, :1].transpose(), U_hat[:,:1])
                    O *= np.dot(V[:1, :], V_hat[:1,:].transpose())
                    Os.append(O)
                    Ss.append(S_hat[0])
                fout.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 0.25, C_A025[i], np.mean(Os), np.std(Os)/sqrt_10))
                fout2.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 0.25, shat_A025[i], np.mean(Ss), np.std(Ss)/sqrt_10))
            else:
                fout.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 1, C_A1[i], 0, -1))
                fout.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 0.75, C_A075[i], 0, -1))
                fout.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 0.5, C_A05[i], 0, -1))
                fout.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 0.25, C_A025[i], 0, -1))

                fout2.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 1, shat_A1[i], 0, -1))
                fout2.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 0.75, shat_A075[i], 0, -1))
                fout2.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 0.5, shat_A05[i], 0, -1))
                fout2.write("%f, %.2f, %f, %f, %f\n" % (sbars[i], 0.25, shat_A025[i], 0, -1))


exit()
## figure 2b  -- A = 1/2

A = 0.5
_, y_data, noisy_y_data, _ = datasets.noisy_SVD_dataset(1000, int(1000*A), noise_var=1./(1000), singular_value_multiplier=3, num_nonempty=3)
U, S, V = np.linalg.svd(y_data, full_matrices=False)
U_hat, S_hat, V_hat = np.linalg.svd(noisy_y_data, full_matrices=False)
pred_sig_S_hat = s_hat_by_A(S[:3], A)
step = 0.2
sbars = np.arange(0, 10, step)
obsv_S_hat = np.histogram(S_hat, bins=sbars)
pred_S_hat = (1000*A-3)*step*mp(sbars, 1, A)
pred_sig_hat = np.histogram(np.array(pred_sig_S_hat), bins=sbars)


with open("figure_stuff/figure_2b_half.csv", "w") as fout:
    fout.write("sbar, obsv_S_hat, pred_MP_S_hat, pred_sig_S_hat\n")
    for i in range(len(sbars)-2):
        fout.write("%f, %f, %f, %f\n" % (sbars[i], obsv_S_hat[0][i], pred_S_hat[i], pred_sig_hat[0][i]))


exit()
## figure 2b  -- A = 1

A = 1
_, y_data, noisy_y_data, _ = datasets.noisy_SVD_dataset(1000, int(1000*A), noise_var=1./(1000), singular_value_multiplier=3, num_nonempty=3)
U, S, V = np.linalg.svd(y_data, full_matrices=False)
U_hat, S_hat, V_hat = np.linalg.svd(noisy_y_data, full_matrices=False)
pred_sig_S_hat = s_hat_by_A(S[:3], A)
#TODO: for other A
step = 0.2
sbars = np.arange(0, 10, step)
obsv_S_hat = np.histogram(S_hat, bins=sbars)
pred_S_hat = (1000*A-3)*step*mp(sbars, 1)
pred_sig_hat = np.histogram(np.array(pred_sig_S_hat), bins=sbars)


with open("figure_stuff/figure_2b.csv", "w") as fout:
    fout.write("sbar, obsv_S_hat, pred_MP_S_hat, pred_sig_S_hat\n")
    for i in range(len(sbars)-2):
        fout.write("%f, %f, %f, %f\n" % (sbars[i], obsv_S_hat[0][i+1], pred_S_hat[i+1], pred_sig_hat[0][i+1]))


exit()


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
