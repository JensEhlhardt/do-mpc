import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc

import matplotlib.pyplot as plt
import pickle
import time

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

#model2 = template_model_dis()
model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
# estimator = do_mpc.estimator.StateFeedback(model)
nD_E = int(model.n_x/4)
x0 = vertcat(np.ones([nD_E, 1]), np.ones([3*nD_E, 1]) * 50.0)
u0 = vertcat(36.7, 2.55, 0.9)

simulator.x0 = x0
for i in range(50):
 #   mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    print(i)


mpc.u0 = u0
mpc.x0 = x0
mpc.set_initial_guess()
#u0 = mpc.make_step(x0)
#for i in range(10):
for i in range(100):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    print(i)
do_mpc.graphics.default_plot(simulator.data)
plt.show()
#mpc.set_initial_guess()
#mpc.make_step(x0)

input('Press any key to exit.')

# estimator.x0 = x0
#
# mpc.set_initial_guess()
#
# fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
# plt.ion()
#
# for k in range(60):
#     u0 = mpc.make_step(x0)
#     y_next = simulator.make_step(u0)
#     x0 = estimator.make_step(y_next)
#
#     graphics.plot_results(t_ind=k)
#     graphics.plot_predictions(t_ind=k)
#     graphics.reset_axes()
#     plt.show()
#     plt.pause(0.1)
# input('Press any key to exit.')
