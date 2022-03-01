
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
from neuralTools import createMultiLayerPerceptron, assembleNarxInput, assembleNarxOutput
import pandas as pd

def template_narx(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    pTable = pd.read_excel("params.xlsx")
    narxNet = createMultiLayerPerceptron(nInputs=4,layer=[10], nOutputs=1)
    narxNet["nOutputs"] = 1
    narxNet["nInputs"] = 1
    narxNet["nOutputDelay"] = 1
    narxNet["nInputDelay"] = 1
    params = model.set_variable(var_type='_p', var_name='p', shape=(61,1))

    #params = pTable['a'].to_numpy()
    # params = np.random.rand(narxNet['weights'].shape[0])
    # Simple oscillating masses example with two masses and two inputs.
    # States are the position and velocitiy of the two masses.
    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(4,1))

    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    h_set = model.set_variable(var_type='_tvp', var_name='h_set', shape=(1,1))

    # xk+1 = [yk yk-1 uk uk-1]
    xx = assembleNarxInput(_x, _u, narxNet)

    #xk1 = 0.43 + 2.02 * np.tanh(0.5 + p1_xk) + 2.11 * np.tanh(-0.77 + p2_xk)
    xk1 = narxNet['fun'](xx, params)
    x_next = assembleNarxOutput(xx, xk1, narxNet)
    #x_next = vertcat(xk1, _x[0], _u, _x[2])

    model.set_rhs('x', x_next)
    model.set_expression('cost', (_x[1]-h_set)**2)
    #model.set_expression('p2', dot(p2, xk))


    model.setup()
    return model

def template_model(symvar_type = "SX"):
    model_type = 'continuous'  # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    h1 = model.set_variable(var_type='_x', var_name='h1', shape=(1,1))

    u  = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

    F_in = 180/1000/3600
    A = 0.04**2 * np.pi

    c_13 = 3.4375e7
    c_23 = 0.9128e7

    dh1_dt = F_in / A - 1/A * (u * np.sqrt(h1) / np.sqrt(c_13 * u**2 + c_23))

    model.set_rhs('h1', dh1_dt)
    model.set_expression('cost', (h1 - 0.25) ** 2)
    model.setup()
    return model