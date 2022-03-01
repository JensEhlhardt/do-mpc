
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
from neuralTools import createMultiLayerPerceptron, assembleNarxInput, assembleNarxOutput
import pandas as pd
import matplotlib.pyplot as plt

def template_narx(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)


    narxNet = createMultiLayerPerceptron(nInputs=155,layer=[100, 100], nOutputs=1)
    narxNet["nOutputs"] = 1
    narxNet["nInputs"] = 155
    narxNet["nOutputDelay"] = 25
    narxNet["nInputDelay"] = 25
    nWeights = narxNet["weights"].shape
    #params = model.set_variable(var_type='_p', var_name='p', shape=nWeights)
    params = pd.read_excel("data/evapNN_Patrick_weights.xlsx")
    nInputs = 5
    TI_21_delayed = model.set_variable(var_type="_x", var_name="TI_21_delayed", shape=(25, 1))
    u = model.set_variable(var_type="_u", var_name="u", shape=(1, 1))
    u_delayed = model.set_variable(var_type="_x", var_name="u_delayed", shape=(25*5, 1))

    u_fixed = model.set_variable(var_type='_tvp', var_name="f_u_P", shape=(4,1))
    u_k = vertcat(u_fixed[0:3], u, u_fixed[3])
    y_k = vertcat(u_k, u_delayed, TI_21_delayed)

    TI_21_hat = narxNet["fun"](y_k, params["weights"])

    model.set_expression('TI_21', TI_21_hat)
    TI_21_rescaled = TI_21_hat * 17.3572 + 313.1758
    model.set_expression('TI_21_rs', TI_21_rescaled)


    TI_21_next = vertcat(TI_21_hat, TI_21_delayed[0:-1])
    u_delayed_next = vertcat(u_k, u_delayed[0:-nInputs])
    model.set_rhs('TI_21_delayed', TI_21_next)
    model.set_rhs('u_delayed', u_delayed_next)
    model.setup()
    return model

def template_narxsim(model):
    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step=1)

    tvp_0 = simulator.get_tvp_template()
    tvp_0['f_u_P'] = np.array([0.23, 0.28, -0.17, 0.04])

    def tvp_fun(t_now):
        return tvp_0
    simulator.set_tvp_fun(tvp_fun)
    simulator.setup()

    return simulator

def template_mpc(model):
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 30,
        't_step': 1,
        'store_full_solution':True,
    }
    mpc.set_param(**setup_mpc)
    tvp_0 = mpc.get_tvp_template()
    tvp_0['_tvp'] = np.array([0.23, 0.28, -0.17, 0.04])


    def tvp_fun(t_now):
        return tvp_0

    mpc.set_tvp_fun(tvp_fun)


    TI_21 = model._x["TI_21_delayed"][0]

    TI_21_rescaled = TI_21 * 17.3572 + 313.1758

    #FIC13_lower = (0-)



    sigma_U = np.reshape(np.array([5.443404262147878, 5.949014814542105, 2.010324835814683, 7.716671791992713, 1.076585440242248]), [5,1])
    mue_U = np.reshape(np.array([52.612421041456190,46.527576539872270,46.435674333704710,36.708290868626200,29.930147101645126]), [5,1])

    u_lower = np.ones([5, 1])* 30
    #u_lower[4] = 10
    u_upper = np.ones([5,1]) * 40
    #u_lower[4] = 70

    u_l_scaled = (u_lower - mue_U) / sigma_U
    u_u_scaled = (u_upper - mue_U) / sigma_U

   # mpc.bounds['lower', '_u', 'u'] = u_l_scaled[3]
    #mpc.bounds['upper', '_u', 'u'] = u_u_scaled[3]

    mterm =(TI_21-0.6)**2
    lterm =(TI_21-0.6)**2
    mpc.set_rterm(u=1e-1)
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.setup()
    return mpc

model = template_narx()
simulator = template_narxsim(model)
mpc = template_mpc(model)
u0 = np.ones([1, 1])*0.1
x0 = np.zeros([150, 1])
mpc.x0 = x0
mpc.u0 = u0
mpc.set_initial_guess()
for i in range(100):
    x0 = simulator.make_step(u0)
    if i == 40:
        mpc.x0 = x0
        mpc.u0 = u0
        mpc.set_initial_guess()
    if i >= 40:
        u0 = mpc.make_step(x0)

do_mpc.graphics.default_plot(simulator.data)
plt.show()