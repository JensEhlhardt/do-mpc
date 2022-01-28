import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')


import matplotlib.pyplot as plt
import pickle as pkl
import time
import pandas as pd
from upstream_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model, 60.0)

# load data set
#with open("data/preprocessed.xlsx", "rb") as file:
dataset = pd.read_excel("data/preprocessed.xlsx")
dataset = dataset[dataset["time_min"] > 180]
firstIndex = dataset.index[0]



## simulate the model
nD_E = int(model.n_x/6)
TI21 = 4*nD_E-1
x0 = vertcat(np.ones([nD_E, 1])*(2.55/3.45),np.ones([nD_E, 1])*(0.9/3.45), np.zeros([nD_E, 1]), np.ones([3*nD_E, 1]) * 50.0)
u0 = vertcat(dataset["TC01.MV"][firstIndex], dataset["WI01.PV"][firstIndex], dataset["FIC14.PV"][firstIndex])
u0 = vertcat(37.6, 2.55, 0.9)
# get initial values 
simulator.x0 = x0
for i in range(100):
    simulator.make_step(u0)
x0 = simulator.x0
simulator = template_simulator(model, 60.0)
simulator.x0 = x0
IntPart = 0
# simulate the data set
nSteps = 300
Qdot = 0
for i in range(nSteps):
    if i > 1:
        T_Set = 300
        T_ist = simulator.data._x[-1, TI21]
        Diff  = T_Set - T_ist
        if Qdot >0 and Qdot<100:
            IntPart += Diff
        K_P = 0.192
        K_I = 0.0101
        Qdot = K_P * Diff + K_I * IntPart
        Qdot = np.min( [np.max([Qdot, 0]), 100])
        u0 = vertcat(Qdot, 2.55, 0.9)

    simulator.make_step(u0)
## plot everything
fig, ax = plt.subplots(3, sharex=True)


# plot simulation
TI04 = 3*nD_E+2
TC01 = 5*nD_E-2
values = simulator.data._x
ax[0].plot(range(nSteps), (values[:, TI04]+values[:, TI04+1])/2)
ax[1].plot(range(nSteps), values[:, TC01])
ax[1].plot(range(nSteps), values[:, TI21])
ax[2].plot(range(nSteps), simulator.data._u[:, 0])
# show plots
plt.show()
input("Say Hi!")