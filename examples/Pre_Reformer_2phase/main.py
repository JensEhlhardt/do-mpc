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

def readAndCutData(name, cut=0):
    dataset = pd.read_excel("data/" + name + ".xlsx")
    dataset = dataset[dataset["time_min"] > cut]
    firstIndex = dataset.index[0]
    u0 = vertcat(dataset["TC01.MV"][firstIndex], dataset["WI01.SV"][firstIndex], dataset["FIC14.SV"][firstIndex])
    return dataset, u0


def createIntegratorForFittin(model):

    _x = model.sv.sym_struct([entry('_x', struct=model._x)])
    _p = model.sv.sym_struct([entry('_u', struct=model._u),
                              entry('_p', struct=model._p)])
    #_p = model.sv.sym_struct([entry('_p', struct=model._p)])

    xdot = model._rhs_fun(_x['_x'], _p['_u'], [], [], _p['_p'], [])
    dae = {
        'x':_x,
        'p':_p,
        'ode': xdot
         }
    F = integrator('F', 'idas', dae)
    return F
datasetNames = ["evapHeater", "naphthaPump", "waterPump"]
cutOffs = [180, 10, 120]


def getInitialCps(mdot_W=2.55, mdot_N=0.9, P=30):

    P = P * 1e5
    # Water properties
    rhoL = CP.PropsSI("D", "Q", 0, "P", P, "Water") * 1000  # g/m³
    rhoV = CP.PropsSI("D", "Q", 1, "P", P, "Water") * 1000  # g/m³
    cp_L = CP.PropsSI("C", "Q", 0, "P", P, "Water") / 1000  # j/g/K
    cp_V = 1.0* CP.PropsSI("C", "Q", 1, "P", P, "Water") / 1000  # j/g/K
    T_sat = CP.PropsSI("T", "Q", 0, "P", P, "Water")-273.15 # °C
    hL = CP.PropsSI("H", "Q", 0, "P", P, "Water") / 1000  # j/g
    hV = CP.PropsSI("H", "Q", 1, "P", P, "Water") / 1000  # j/g
    hLV = hV - hL

    # Naphtha properties
    nStr = "SRK::CycloHexane[0.5]&Benzene[0.5]"
    nStr = "Water"
    rhoLN = CP.PropsSI("D", "Q", 0, "P", P, nStr) * 1000  # g/m³
    rhoVN = CP.PropsSI("D", "Q", 1, "P", P, nStr) * 1000  # g/m³
    cp_LN = CP.PropsSI("C", "Q", 0, "P", P, nStr) / 1000  # j/g/K
    cp_VN = CP.PropsSI("C", "Q", 1, "P", P, nStr) / 1000  # j/g/K
    T_satN = CP.PropsSI("T", "Q", 0, "P", P, nStr) - 273.15  # °C
    # T_satN = T_sat
    hL = CP.PropsSI("H", "Q", 0, "P", P, nStr) / 1000  # j/g
    hV = CP.PropsSI("H", "Q", 1, "P", P, nStr) / 1000  # j/g
    hLVN = (hV - hL)

    omegaL = mdot_W / (mdot_W + mdot_N)
    omegaLN = mdot_N / (mdot_W + mdot_N)

    # cpL_M = omegaL * cp_L + omegaLN * cp_LN
    # cpV_M = omegaL * cp_V + omegaLN * cp_VN
    # hLV_M = omegaL * hLV  + omegaLN * hLVNa
    cpL_M = cp_L
    cpV_M = cp_V
    hLV_M = hLV

    return cpL_M, cpV_M, hLV_M

#optimizer = do_mpc.optimizer.Optimizer()


########################################################################################################################
########################################################################################################################
########################################################################################################################
dataInd = 0

cut = cutOffs[dataInd]
name = datasetNames[dataInd]

dataset, u0 = readAndCutData(name, cut)

model = template_model()
simulator = template_simulator(model)

x0 = getRawInitialState(model)

simulator.x0 = x0
simulator.set_initial_guess()
simulator = convergeToSteadyState(simulator, u0, tSteps=500)
x0 = simulator.x0
simulator.reset_history()
simulator.x0 = x0
simulator.set_initial_guess()

nD = int(model.n_x / len(model._x.keys()))

TI04 = nD + 2
TI21 = 2*nD-1
TI01 = 3*nD-5


cpL = MX.sym('cpL')
cpV = MX.sym('cpV')
kEvap = MX.sym('kEvap')
U_L = MX.sym('U_L')
U_V = MX.sym('U_V')
hLV = MX.sym('hLV')
Qdot_E = MX.sym('Qdot_e')
mdot_W = MX.sym('mdot_W')
mdot_N = MX.sym('mdot_N')
param = vertcat(cpL, cpV, kEvap, U_L, U_V, hLV)
J=0
xi = x0
#p_k = vertcat(Qdot_E, mdot_W, mdot_N, cpL, cpV, kEvap)
# for i in dataset.index:
#     u0 = vertcat(dataset['TC01.MV'][i], dataset['WI01.PV'][i], dataset['FIC14.SV'][i])
#     p_k = vertcat(dataset['TC01.MV'][i], dataset['WI01.PV'][i], dataset['FIC14.SV'][i], param)
#     if i != dataset.index[0]:
#         xi = xi['xf']
#     xi = F(x0=xi, p=p_k)
#
#     J += 1.0 * (xi['xf'][TI21]-dataset['TI21.PV'][i])**2
#     J += 1.0 * (xi['xf'][TI04]-dataset['TI04.PV'][i])**2
#     J += 0.5 * (xi['xf'][TI01]-dataset['TC01.PV'][i])**2

#do_mpc.graphics.default_plot(simulator.data)

J = J / len(dataset.index)

g = vertcat(cpV - cpL, U_V - U_L)
prob = {'f':J, 'x':param, 'g': g}

cpL_M, cpV_M, hLV_M = getInitialCps()

param0 = [cpL_M, cpV_M, 0., 1.2, 1.4, hLV_M/1000]
paramL = [0., 0., 0., 0., 0., 0]
paramU = [5., 5., 5., 5., 5., 2.]
options = {"ipopt": {"max_iter":100}}
solver = nlpsol('solver', 'ipopt', prob, options)
#paramOpt = solver(x0 = param0, lbx=paramL, ubx=paramU, ubg=[-1., 0.])
param0 = [cpL_M, cpV_M, 1, hLV_M/1000]
paramOpt = {'x': param0}
## simulate the opt result
#print(paramOpt['x'])

# simulator = template_simulator(model2, cpL=paramOpt['x'][0], cpV=paramOpt['x'][1], kEvap=paramOpt['x'][2],
#                             hLV=paramOpt['x'][3])
#simulator = template_simulator(model)
simulator.x0 = x0
simulator.set_initial_guess()

for i in dataset.index:
    u0 = vertcat(dataset['TC01.MV'][i], dataset['WI01.PV'][i], dataset['FIC14.PV'][i])
    simulator.make_step(u0)



########################################################################################################################
########################################################################################################################
## plots along the z-coordinate at some time points ####################################################################
tPoints = np.linspace(0, 3*nD, 5)

fig, ax = plt.subplots(3, 1)


for i in range(3):
    ax[i].plot(np.transpose(simulator.data._x[tPoints.astype(int), i * nD: (i+1) * nD]))

fig, ax = plt.subplots(3, 1)

#dataset.plot(x='time_min', y='TI04.MV', ax=ax[0])
dataset.plot(x='time_min', y='TI21.PV', ax=ax[0])
#dataset.plot(x='time_min', y='TI21.MV')

nameDic = {"evapHeater":"TC01.MV", "naphthaPump":"FIC14.PV", "waterPump":'WI01.SV'}

dataset.plot(x='time_min', y=nameDic[name], ax=ax[1])

ax[0].set_title("Change in " + name)
ax[0].plot(dataset.time_min, simulator.data._x[:, TI21])
ax[0].grid(True)
ax[0].set_ylabel('T [°C]')
ax[1].grid(True)


ax[2].plot(dataset.time_min, dataset['TC01.PV'])
ax[2].plot(dataset.time_min, simulator.data._x[:, TI01+4])
#ax[2].plot(dataset.time_min, simulator.data._x[:, nD-1])
ax[2].grid(True)
ax[2].set_ylabel('alpha_out [-]')
plt.show()