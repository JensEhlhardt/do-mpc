import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
import CoolProp.CoolProp as CP


def template_mpc(model):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 6,
        'n_robust' : 1,
        'open_loop': 0,
        't_step': 300,
        'state_discretization': 'collocation',
        'collocation_deg': 2,
        'collocation_ni': 1,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }
    mpc.set_param(**setup_mpc)
    suppress_ipopt = {'ipopt.print_level':0, 'ipopt.sb':'yes', 'print_time':0}
    #mpc.set_param(nlpsol_opts = suppress_ipopt)

    # uncertain parameters
    nStr = "SRK::CycloHexane"
    hL = CP.PropsSI("H", "Q", 0, "P", 30e5, nStr) / 1000  # j/g
    hV = CP.PropsSI("H", "Q", 1, "P", 30e5, nStr) / 1000  # j/g
    cpV = CP.PropsSI("C", "Q", 1, "P", 30e5, nStr)/1000  # j/g/K
    cpL = CP.PropsSI("C", "Q", 0, "P", 30e5, nStr)/1000  # j/g/K
    hLVN = hV - hL
    #hlv_var = np.array([2.0, 2.1, 1.9])*hLVN
    #hlv_var = np.array([2.1])*hLVN #, 1.9, 2.05
    cpL_var = np.array([0.9]) * cpL
    cpV_var = np.array([0.9]) * cpV

#    mpc.set_uncertainty_values(hLVN=hlv_var)
    mpc.set_uncertainty_values(cpVN = cpV_var)
    mpc.set_uncertainty_values(cpLN = cpL_var)
    # TVP for set point
    tvp_temp = mpc.get_tvp_template()
    tvp_temp2 = mpc.get_tvp_template()
    tvp_temp2['_tvp', :] = 1.0
    tvp_temp['_tvp', :] = 0.9
    def tvp_fun(t_now):
        if t_now <7200.0:
            return tvp_temp
        else:
            return tvp_temp2

    mpc.set_tvp_fun(tvp_fun)

    _u = model.u
    _x = model.x
    aux= model.aux
    mdotN_diff = aux['mdotN_diff']
    #T_diff = aux['T_diff']
    T_diff = (_x['T_FE'][-1] - 350)
    S_C = aux['S_C']
    a_E  = _x['a_E']
    T_FE = _x['T_FE']
    lterm = T_diff**2 + 1e4*(mdotN_diff)**2 + 1e3*(_u['mdot_W'] - 2.55)**2
    mterm =  vertcat(0.0)#T_diff**2
    mpc.set_rterm(Qdot_E = 1)
    mpc.set_rterm(mdot_N = 2)
    mpc.set_rterm(mdot_W = 2)

    
    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.bounds['lower', '_u', 'Qdot_E'] = 0
    mpc.bounds['upper', '_u', 'Qdot_E'] = 100
    mpc.bounds['lower', '_u', 'mdot_W'] = 0
    mpc.bounds['upper', '_u', 'mdot_W'] = 5
    mpc.bounds['lower', '_u', 'mdot_N'] = 0
    mpc.bounds['upper', '_u', 'mdot_N'] = 3

    #mpc.bounds['upper', '_x', 'T_FE'] = 450
    #mpc.bounds['upper', '_x', 'T_WE'] = 450

    #mpc.set_nl_cons('test', _x["T_FE"], ub = 350)
    #mpc.set_nl_cons('S_C_lower', -aux['S_C'], ub=-0.15, soft_constraint=True, penalty_term_cons=1)
    #mpc.set_nl_cons('S_C_upper', aux['S_C'], ub=0.2, soft_constraint=True, penalty_term_cons=1)

    #mpc.bounds['lower', '_x', 'a_E'] = 0.0
    
    mpc.scaling['_x', 'a_E'] = 1e-2
    mpc.scaling['_u', 'mdot_N'] = 1e-1
    mpc.scaling['_u', 'mdot_W'] = 1e-1
    mpc.scaling['_u', 'Qdot_E'] = 1e0

    mpc.setup()
    return mpc
