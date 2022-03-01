
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
import pandas as pd

def template_mpc(model):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_robust': 1,
        'n_horizon': 100,
        't_step': 1,
        'open_loop': True,
        'store_full_solution':True,
    }

    mpc.set_param(**setup_mpc)

    mterm = model.aux['cost']
    lterm = model.aux['cost'] # terminal cost

    tvp1 = mpc.get_tvp_template()
    tvp2 = mpc.get_tvp_template()

    tvp1['_tvp',:] = 0.5
    tvp2['_tvp',:] = 0.2
    def tvp_fun(t_now):
        if t_now < 100:
            return tvp1
        else:
            return tvp2

    mpc.set_tvp_fun(tvp_fun)
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1e0)

    mpc.bounds['lower', '_u', 'u'] = 0.0
    mpc.bounds['upper', '_u', 'u'] = 1.0

    pTable = pd.read_excel("params.xlsx")

    n_tree = 5
    p_template = mpc.get_p_template(n_tree)
    p_template['_p', 0] = pTable['a']
    p_template['_p', 1] = pTable['b']
    p_template['_p', 2] = pTable['c']
    p_template['_p', 3] = pTable['d']
    p_template['_p', 4] = pTable['e']

    def p_fun(t_now):
        return p_template

    mpc.set_p_fun(p_fun)

    mpc.setup()

    return mpc
