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

from upstream_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

#model2 = template_model_dis()
model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model, 60)
# estimator = do_mpc.estimator.StateFeedback(model)
nD_E = int(model.n_x/6)
x0 = vertcat(np.ones([nD_E, 1])*(2.55/3.45),np.ones([nD_E, 1])*(0.9/3.45), np.zeros([nD_E, 1]), np.ones([3*nD_E, 1]) * 50.0)
#x0 = vertcat(np.ones([nD_E, 1])*(2.55/3.45),np.ones([nD_E, 1])*(0.9/3.45), np.ones([3*nD_E, 1]) * 50.0)
u0 = vertcat(36.7, 2.55, 0.9)
if True:
    simulator.x0 = x0
    for i in range(50):
     #   mpc.make_step(x0)
        x0 = simulator.make_step(u0)
        print(i)

    simulator = template_simulator(model, 60)
    simulator.x0 = x0

    mpc.u0 = u0
    mpc.x0 = x0
    mpc.set_initial_guess()
    #u0 = mpc.make_step(x0)
    #for i in range(10):
    for i in range(50):
        u0 = mpc.make_step(x0)
        for j in range(5):
            x0 = simulator.make_step(u0)
        print(i)
    do_mpc.graphics.default_plot(mpc.data)
    plt.show()
    do_mpc.data.save_results([simulator, mpc], result_name="result_nom")
else:
    results = do_mpc.data.load_results("results/073_result_nom.pkl")
    mpc.data = results["mpc"]
    simulator.data = results["simulator"]
#mpc.set_initial_guess()
#mpc.make_step(x0)

#graphics = do_mpc.graphics.Graphics(simulator.data)
mpc.data._time = mpc.data._time/60
simulator.data._time = simulator.data._time/60

fig, ax = plt.subplots(3, sharex=True)
TI04 = 3*nD_E+2
TC01 = 5*nD_E-2
TI21 = 4*nD_E-1
# T_FE
ax[0].plot(simulator.data._time, simulator.data._x[:, TI21])
#ax[0].step(mpc.data._time, mpc.data._tvp[:, 0], '--', where='post')
ax[0].plot(simulator.data._time[[0, -1]], [350, 350], '--')
# T_W
ax[0].plot(simulator.data._time, simulator.data._x[:, TC01])
ax[0].legend(["TI21.PV", "TI21.SV", "TC01.PV"])
ax[0].grid(True)
ax[0].set_xlabel('t [min]')
ax[0].set_ylabel('T [°C]')
# S_C
ax[1].step(simulator.data._time, simulator.data._u[:, 1], where='post')
# ax[1].plot(simulator.data._time, simulator.data._aux[:, 2])
# ax[1].legend(["S_C"])
# ax[1].grid(True)
ax[1].set_xlabel('t [min]')
ax[1].set_ylabel('TC01.MV [%]')

# inputs
#ax[2].step(mpc.data._time, mpc.data._u[:, 0]/100, where='post') # heating
ax[2].step(mpc.data._time, mpc.data._u[:, 1]/1, where='post') # water
ax[2].step(mpc.data._time, mpc.data._u[:, 2]/1, where='post') # naphtha
ax[2].legend(["WC01.PV", "FC14.PV"])
ax[2].grid(True)
ax[2].set_xlabel('t [min]')
ax[2].set_ylabel('mdot [kg/h]')
#ax[2].set_ylim([0, 100])
plt.show()


input('Press any key to exit.')

