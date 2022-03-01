import do_mpc
from casadi import *
from casadi.tools import *
from upstreamModel2Phase import template_model, getRawInitialState
from upstreamModel3Phase import template_model_3

from upstreamModel3Phase import getRawInitialState as getNewRaw
from simulator import template_simulator, convergeToSteadyState
import matplotlib.pyplot as plt
import pandas as pd
import CoolProp.CoolProp as CP



def readAndCutData():
    #dataset = pd.read_excel("data/validationData_.xlsx")
    dataset = pd.read_excel("data/evapRedo.xlsx")
    firstIndex = 5
    dataset = dataset[dataset.index >= firstIndex]

    u0 = vertcat(dataset["TC01.MV"][firstIndex], dataset["WI01.SV"][firstIndex], dataset["FIC14.SV"][firstIndex],
                 dataset['TC03.MV'][firstIndex], dataset["TC08.MV"][firstIndex])
    return dataset, u0

########################################################################################################################

# read the data
validationData, u0 = readAndCutData()

nD_Evap =40
nD_Pipe =10
nD_Super =20
nD_Pre = 30

# initialize model and simulator instances
model = template_model()
simulator = template_simulator(model)

model3 = template_model_3()
simulator3 = template_simulator(model3)

#get raw initial states
#x03 = getNewRaw(model3)
#simulator3.x0 = x03
#simulator3.set_initial_guess()
# get initial states for the simulation
x0 = getRawInitialState(model)
simulator.x0 = x0
simulator = convergeToSteadyState(simulator, u0, tSteps=500)
#simulator3 = convergeToSteadyState(simulator3, u0, tSteps=300)
#do_mpc.graphics.default_plot(simulator.data)
x0 = simulator.x0
simulator.reset_history()
simulator.x0 = x0
simulator.set_initial_guess()

#x0 = simulator3.x0
#simulator3.reset_history()
#simulator3.x0 = x0
#simulator3.set_initial_guess()

# get some indices


TI04 = nD_Evap + 2
TI21 = 2*nD_Evap-1
TI01 = 3*nD_Evap-3
TI02 = 4*nD_Evap + 3*nD_Pipe + nD_Super + 2
TI14 = 4*nD_Evap + 3*nD_Pipe + 2*nD_Super - 1
TI22 = 4*nD_Evap + 3*nD_Pipe + 4*nD_Super + 2*nD_Pre - 1
#  simulate
for i in validationData.index:
    if i > 1:
        u0 = vertcat(validationData['TC01.MV'][i], validationData['WI01.SV'][i], validationData['FIC14.PV'][i],
                     validationData['TC03.MV'][i], validationData['TC08.MV'][i])
        simulator.make_step(u0)
 #       simulator3.make_step(u0)
        print(i)
        if i == 68:
            print(i)

do_mpc.graphics.default_plot(simulator.data)
## Plot comparison
# Plot evap
fig, ax = plt.subplots(5, 1)

validationData.plot(x="time_min", y="TI21.PV", ax=ax[0])
validationData.plot(x="time_min", y="TC01.PV", ax=ax[0])
validationData.plot(x="time_min", y="WI01.SV", ax=ax[1])
validationData.plot(x="time_min", y="FIC14.PV", ax=ax[1])
#validationData.plot(x='time_min', y="FIC13.MV", ax=ax[1])
validationData.plot(x="time_min", y="TC01.MV", ax=ax[2])
validationData.plot(x="time_min", y="TC03.MV", ax=ax[2])
validationData.plot(x="time_min", y="TC08.MV", ax=ax[2])
ax[0].plot(validationData["time_min"], simulator.data._x[:, TI21])
ax[0].plot(validationData["time_min"], (simulator.data._x[:, TI01]+simulator.data._x[:, TI01-1])/2)
# Super heater
ax[3].plot(validationData["time_min"], simulator.data._x[:, TI02])
ax[3].plot(validationData["time_min"], simulator.data._x[:, TI14-1])
ax[3].plot(validationData["time_min"], simulator.data._x[:, TI14])
ax[3].plot(validationData["time_min"], simulator.data._x[:, TI14-2])
validationData.plot(x="time_min", y="TI02.PV", ax=ax[3])
validationData.plot(x="time_min", y="TI14.PV", ax=ax[3])
# preheating zone
ax[4].plot(validationData["time_min"], simulator.data._x[:, TI22])
validationData.plot(x="time_min", y="TI22.PV", ax=ax[4])
plt.show()




input('Press any key to exit.')



