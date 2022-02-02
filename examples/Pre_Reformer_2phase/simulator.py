import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc


def template_simulator(model, cpL=None, cpV=None, kEvap=None, U_L=None, U_V=None, hLV=None):

    # create simulator object
    simulator = do_mpc.simulator.Simulator(model)

    # define the simulation parameters
    params_simulator = {
        'integration_tool': 'idas',
        't_step': 30,
        'abstol': 1e-10,
        'reltol': 1e-10
    }

    # set the parameters
    simulator.set_param(**params_simulator)


    # set the parameter functions if necessary
    if model.modelUsage:
        p_template = simulator.get_p_template()

        def p_fun(t_now):
            p_template['cpL'] = cpL
            p_template['cpV'] = cpV
            p_template['kEvap'] = kEvap
            p_template['hLV'] = hLV
            return p_template

        simulator.set_p_fun(p_fun)
    # setup and return the object
    simulator.setup()
    return simulator

def convergeToSteadyState(simulator, u0, tSteps=200):
    for i in range(tSteps):
        #print(i)
        try:
            simulator.make_step(u0)
        except:
            break
    return simulator
