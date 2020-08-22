import torch

import goptimization as gop

import matplotlib.pyplot as plt


to_optimize = gop.paraboloid_fun

dim = 20
x0 = torch.zeros(dim)

trajectory_mgd = []

num_time = 1000
num_runs = 100

torch.manual_seed(0)
for _ in range(num_runs):
    _, _, _, y_list = gop.modified_gradient_descent(to_optimize, x0, num_time, 0.1, 1, 9.9)
    trajectory_mgd.append(y_list)


trajectory_mgd_t = list(map(list, zip(*trajectory_mgd)))
mean_mgd = [sum(x) / num_runs for x in trajectory_mgd_t]

trajectory_gd = []

torch.manual_seed(0)
for _ in range(num_runs):
    _, _, _, y_list = gop.modified_gradient_descent(to_optimize, x0, num_time, 0.1, 1, 0)
    trajectory_gd.append(y_list)


trajectory_gd_t = list(map(list, zip(*trajectory_gd)))
mean_gd = [sum(x) / num_runs for x in trajectory_gd_t]

# Histogram
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.hist(trajectory_mgd_t[100])
ax2.hist(trajectory_gd_t[100])

plt.show()


# Mean
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(mean_mgd)
ax2.plot(mean_gd)

plt.show()
