
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
import CoolProp.CoolProp as CP


def template_simulator(model, tDelta):
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        'integration_tool': 'idas',
        't_step': tDelta,
        'abstol': 10**-10,
        'reltol': 10**-10
    }

    simulator.set_param(**params_simulator)

    nStr = "SRK::CycloHexane"
    #nStr = "Water"
    P = 30e5
    hL = CP.PropsSI("H", "Q", 0, "P", 30e5, nStr) / 1000  # j/g
    hV = CP.PropsSI("H", "Q", 1, "P", 30e5, nStr) / 1000  # j/g
    hLVN = (hV - hL) * 2

    cpV = CP.PropsSI("C", "Q", 1, "P", 30e5, nStr)/1000  # j/g/K
    cpL = CP.PropsSI("C", "Q", 0, "P", 30e5, nStr)/1000  # j/g/K



    p_num = simulator.get_p_template()
    # p_num['a_In'] = 1
    # p_num['T_In'] = 50
    #p_num['hLVN'] = hLVN
    p_num['cpLN'] = cpL
    p_num['cpVN'] = cpV
    def p_now(t_now):
        return p_num



    simulator.set_p_fun(p_now)

    tvp_temp = simulator.get_tvp_template()
    tvp_temp2 = simulator.get_tvp_template()
    tvp_temp2['mdotN_set'] = 1.2
    tvp_temp['mdotN_set'] = 0.9

    def tvp_fun(t_now):
        if t_now < 7200.0:
            return tvp_temp
        else:
            return tvp_temp2

    simulator.set_tvp_fun(tvp_fun)


    simulator.setup()
    return simulator
