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
# simulate the data set
for i in dataset.index:
    u0 = vertcat(dataset["TC01.MV"][i], dataset["WI01.PV"][i], dataset["FIC14.PV"][i])
    simulator.make_step(u0)
## plot everything
fig, ax = plt.subplots(3, sharex=True)

# plot data
dataset.plot(x="time_min", y="TI04.PV", ax=ax[0])

dataset.plot(x="time_min", y="TI21.PV", ax=ax[1])
dataset.plot(x="time_min", y="TC01.PV", ax=ax[1])

dataset.plot(x="time_min", y="TC01.MV", ax=ax[2])


# plot simulation
TI04 = 3*nD_E+2
TC01 = 5*nD_E-2
TI21 = 4*nD_E-1
values = simulator.data._x
#
ax[0].plot(dataset["time_min"], (values[:, TI04]+values[:, TI04+1])/2, '--')
ax[0].set_ylim([50, 65])
ax[0].set_ylabel("T [°C]")
ax[0].grid(True)
ax[0].set_title("Fit to data, (solid:data, dashed:simulation)")

#
#ax[1].set_ylim([])
ax[1].plot(dataset["time_min"], values[:, TC01], '--')
ax[1].plot(dataset["time_min"], (values[:, TI21]+values[:, TI21-1])/2, "--")
ax[1].set_ylim([250, 400])
ax[1].set_ylabel("T [°C]")
ax[1].grid(True)
#ax[1].plot(dataset["time_min"], values[:, TI21-1])
ax[2].set_ylim([35, 40])
ax[2].set_ylabel("MV [%]")
ax[2].grid(True)
ax[2].set_xlabel("t [min]")


# show plots
plt.show()
a = 1
input("Say Hi!")