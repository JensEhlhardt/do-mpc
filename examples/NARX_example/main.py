

import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import time
sys.path.append('../../')
import do_mpc

from template_simulator import template_simulator
from narx_model import template_narx, template_model

from template_mpc import template_mpc


"""
Get configured do-mpc modules:
"""
narx = template_narx()
model = template_model()
simulator = template_simulator(model)
mpc = template_mpc(narx)
#simulator_narx = template_simulator(narx)
"""
Set initial state
"""


u0 = vertcat(.5)
x0 = 1.

simulator.x0 = x0
simulator.set_initial_guess()

x0 = np.ones([4,1])
mpc.x0 = x0
mpc.u0 = u0
mpc.set_initial_guess()
x0_i = x0
#simulator_narx.x0 = x0
#simulator_narx.set_initial_guess()


fig, ax = plt.subplots(3,1)
mpc_plot = do_mpc.graphics.Graphics(mpc.data)
sim_plot = do_mpc.graphics.Graphics(simulator.data)
mpc_plot.add_line('_x', 'x', ax[0])
mpc_plot.add_line('_u', 'u', ax[1])
sim_plot.add_line('_aux', 'cost', ax[2])
sim_plot.add_line('_x', 'h1', ax[0])
plt.ion()

for i in range(250):
    #simulator_narx.make_step(u0)
    x0 = simulator.make_step(u0)
    #x0_i = simulator_narx.make_step(u0)
    if i > 1:
        x0_i = np.vstack((simulator.data._x[-2:], simulator.data._u[-2:]))
        u0 = mpc.make_step(x0_i)
    #    x0_i[0]  = x0
        mpc.x0 = x0_i
        mpc.set_initial_guess()

        mpc_plot.plot_results()
        mpc_plot.plot_predictions()

        sim_plot.plot_results()

        sim_plot.reset_axes()
        mpc_plot.reset_axes()
        
        #plt.show()
        #plt.pause(0.01)

do_mpc.graphics.default_plot(simulator.data)
# do_mpc.graphics.default_plot(mpc.data)
# do_mpc.graphics.default_plot(simulator_narx.data)
# plt.show()
"""
Setup graphic:
"""


"""
Run MPC main loop:
"""



input('Press any key to exit.')


