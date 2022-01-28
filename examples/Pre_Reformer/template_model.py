import numpy as np
from casadi import *
#from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
from casadi import *
import CoolProp.CoolProp as CP
def template_model(symvar_type='SX'):

    model_type = 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    ## fixed parameters
    # Discretisation
    nD_Evap = 30
    # Scales
    lengthEvap = 2.3 # [m]
    dz_E = lengthEvap/nD_Evap # [m]
    gammaP = 0.89
    dr_ES = 0.0036
    r_ES = 0.0212 - dr_ES


    A_z_E = r_ES**2 * np.pi * gammaP
    A_z_EP= r_ES**2 * np.pi * (1-gammaP)
    A_r_E = r_ES * 2 * np.pi * dz_E
    A_z_W = ((r_ES + dr_ES)**2 - r_ES**2) * np.pi
    # fluid properties
    P = 30*1e5 # Pa 30bar
    #P = 1e5# Pa 1bar

    rhoW = 7.87e6
    rhoP = rhoW

    # Water properties
    rhoL = CP.PropsSI("D", "Q", 0, "P", P, "Water") * 1000  # g/m³
    rhoV = CP.PropsSI("D", "Q", 1, "P", P, "Water") * 1000  # g/m³
    cp_L = CP.PropsSI("C", "Q", 0, "P", P, "Water") / 1000  # j/g/K
    cp_V = CP.PropsSI("C", "Q", 1, "P", P, "Water") / 1000  # j/g/K
    T_sat = CP.PropsSI("T", "Q", 0, "P", P, "Water")-273.15 # °C

    hL = CP.PropsSI("H", "Q", 0, "P", P, "Water") / 1000  # j/g
    hV = CP.PropsSI("H", "Q", 1, "P", P, "Water") / 1000  # j/g
    hLV = hV - hL
    S_L = (rhoL / rhoV)
    # Naphtha
    nStr = "SRK::CycloHexane"
    rhoLN = CP.PropsSI("D", "Q", 0, "P", P, nStr) * 1000  # g/m³
    rhoVN = CP.PropsSI("D", "Q", 1, "P", P, nStr) * 1000  # g/m³
    cp_LN = CP.PropsSI("C", "Q", 0, "P", P, nStr) / 1000  # j/g/K
    cp_VN = CP.PropsSI("C", "Q", 1, "P", P, nStr) / 1000  # j/g/K
    T_satN = CP.PropsSI("T", "Q", 0, "P", P, nStr) - 273.15  # °C

    hL = CP.PropsSI("H", "Q", 0, "P", P, nStr) / 1000  # j/g
    hV = CP.PropsSI("H", "Q", 1, "P", P, nStr) / 1000  # j/g
    hLVN = hV - hL

    U_L  = 3000
    U_V  = 1000


    cp_W = 0.5
    cp_P = cp_W
    S_V  = 1000
    D_W  = 15
    T_env = 20
    kEvap = 2
    # create state variables
    a_E  = model.set_variable(var_type='_x', var_name='a_E', shape=(nD_Evap, 1))
    T_FE = model.set_variable(var_type='_x', var_name='T_FE', shape=(nD_Evap, 1))
    T_WE = model.set_variable(var_type='_x', var_name='T_WE', shape=(nD_Evap, 1))
    T_PE = model.set_variable(var_type='_x', var_name='T_PE', shape=(nD_Evap, 1))

    # create input variables
    Qdot_E = model.set_variable(var_type='_u', var_name='Qdot_E', shape=(1, 1))
    mdot_W = model.set_variable(var_type='_u', var_name='mdot_W', shape=(1, 1))
    mdot_N = model.set_variable(var_type='_u', var_name='mdot_N', shape=(1, 1))
    #mdot_W = 2.55
    #mdot_N = 0.9
    # creat model parameters
    # a_In = model.set_variable(var_type = '_tvp', var_name='a_In', shape = (1,1))
    #     # T_In = model.set_variable(var_type = '_tvp', var_name='T_In', shape = (1,1))
    a_In = 1
    T_In = 20
    # some calculations
    v_L = (mdot_N + mdot_W) * 1000 / 3600 / rhoL / A_z_E

    # create RHS
    #some helper functions
    x  = SX.sym('x')
    aE = SX.sym('aE')
    TF = SX.sym('TF')
    r_s = 0.1

    softplus = Function('soft', [x], [(x + sqrt(x**2 + 0.01))/2])
    softplus = Function('relu', [x], [fmax(x,0)])
    #softplus = Function('soft', [x], [casadi.log(1+casadi.exp(x))])
    #softplus = Function('soft2', [x], [x * (1/(1+casadi.exp(-x)))])
    #mDotE = Function('mDotE', [aE, TF], [fmax(r_s * rhoL * aE * (TF-T_sat)/T_sat, 0)])
    mDotE = Function('mDotE', [aE, TF], [softplus(r_s * rhoL * aE * (TF-T_sat)/T_sat)])
    #mDotC = Function('mDotC', [aE, TF], [fmax(r_s * rhoV * (1-aE) * (T_sat-TF)/T_sat, 0)])
    mDotC = Function('mDotC', [aE, TF], [softplus(r_s * rhoV * (1-aE) * (T_sat-TF)/T_sat)])
    #mDotC = Function('mDotC', [aE, TF], [0])

    v_F    = Function('v_F', [aE], [v_L * (aE*(1-S_L)+S_L)])

    U_F   = Function('U_F', [aE], [aE * (U_L - U_V) + U_V])
    #cp_F  = Function('cp_F', [aE], [aE * (cp_L - cp_V) + cp_V])
    cp_F = Function('cp_F', [aE], [(aE * rhoL * cp_L + (1-aE) * rhoV * cp_V)/(aE * rhoL + (1-aE) * rhoV)])
    rho_F = Function('U_F', [aE], [aE * (rhoL - rhoV) + rhoV])



    # Evap
    da_E  = SX.sym('da_E',  nD_Evap, 1)
    dT_FE = SX.sym('dT_FE', nD_Evap, 1)
    dT_PE = SX.sym('dT_PE', nD_Evap, 1)
    dT_WE = SX.sym('dT_WE', nD_Evap, 1)

    # Prevent some numerics
    #a_E = casadi.if_else(a_E<1e-5, 0, a_E)

    # get correct heating power
    Qdot_E = -0.318 * Qdot_E**2 + 64.163 * Qdot_E + 275.54
    #Qdot_E = Qdot_E/100 * 3600
    # get the heated zone in the evaporator
    stoptHeating = np.float(nD_Evap/3)
    startHeating  = np.float(2*nD_Evap/3)

    # define pdes
    for i in range(0, nD_Evap):
        if i != 0:
            # get actual heating
            Qdot_E_i = casadi.if_else(i <= stoptHeating or i >= startHeating, 3*Qdot_E/lengthEvap*dz_E, 0)
            #Qdot_E_i = Qdot_E/lengthEvap * dz_E
            # volume fraction
            da_E[i] = -v_L * (a_E[i]-a_E[i-1])/dz_E + (mDotC(a_E[i], T_FE[i]) - mDotE(a_E[i], T_FE[i]))/rhoL
            # fluid temperature
            cpF = cp_F(a_E[i])
            vF  = v_F(a_E[i])
            rhoF = rho_F(a_E[i])
            dT_FE[i] = -vF * (T_FE[i] - T_FE[i-1])/dz_E + A_r_E * U_F(a_E[i]) * (T_WE[i]-T_FE[i]) / (cpF * rhoF * A_z_E * dz_E) \
                - (mDotE(a_E[i], T_FE[i]) - mDotC(a_E[i], T_FE[i])) * hLV / (cpF * rhoF) \
                - S_V * U_F(a_E[i]) * (T_FE[i] - T_PE[i])/ (cpF * rhoF * (gammaP))
              #  - A_z_E * T_FE[i] * (a_E[i]-a_E[i-1])/dz_E * ((cp_L-cp_V)*rhoF*vF + (rhoL-rhoV)*cpF*vF + v_L*(1-S_L)*cpF*rhoF)
        # packing temperature
        dT_PE[i] = S_V * U_F(a_E[i]) * (T_FE[i] - T_PE[i]) / (cp_P * rhoP * (1-gammaP))
        # wall temperature
        if i != nD_Evap-1 and i != 0:
            dT_WE[i] = Qdot_E_i / (cp_W * rhoW * A_z_W * dz_E) - kEvap * (T_WE[i] - T_env) / (rhoW * cp_W * A_z_W)\
                - A_r_E * U_F(a_E[i]) * (T_WE[i]-T_FE[i]) / (cp_W * rhoW * A_z_W * dz_E) \
                + D_W / (rhoW * cp_W) * (T_WE[i+1] - 2*T_WE[i] + T_WE[i-1])/dz_E**2
    # define boundaries
    da_E[0] = -v_L * (a_E[0] - a_In)/dz_E + (mDotC(a_E[0], T_FE[0]) - mDotE(a_E[0], T_FE[0]))/rhoL
    #da_E[0] = a_In - a_E[0]
    dT_FE[0] = -v_F(a_E[0]) * (T_FE[0] - T_In)/dz_E + A_r_E * U_F(a_E[0]) * (T_WE[0]-T_FE[0]) / (cp_F(a_E[0]) * rho_F(a_E[0]) * A_z_E * dz_E) \
                - (mDotE(a_E[0], T_FE[0]) - mDotC(a_E[0], T_FE[0])) * hLV / (cp_F(a_E[0]) * rho_F(a_E[0])) \
                - S_V * U_F(a_E[0]) * (T_FE[0] - T_PE[0])/ (cp_F(a_E[0]) * rho_F(a_E[0]))
    #dT_FE[0] = T_In - T_FE[0]

    dT_WE[0] = 2 * D_W / (rhoW * cp_W) * (T_WE[1]-T_WE[0])/dz_E**2 + Qdot_E_i/(cp_W * rhoW * A_z_W * dz_E) \
           - A_r_E * U_F(a_E[0]) * (T_WE[0] - T_FE[0]) / (cp_W * rhoW * A_z_W * dz_E)
    dT_WE[-1] = 2 * D_W / (rhoW * cp_W) * (T_WE[-2] - T_WE[-1]) / dz_E**2 + Qdot_E_i/(cp_W * rhoW * A_z_W * dz_E) \
           - A_r_E * U_F(a_E[-1]) * (T_WE[-1] - T_FE[-1]) / (cp_W * rhoW * A_z_W * dz_E)

    model.set_rhs("a_E",  da_E)
    model.set_rhs("T_FE", dT_FE)
    model.set_rhs("T_WE", dT_WE)
    model.set_rhs("T_PE", dT_PE)

    T_set = model.set_variable(var_type='_tvp', var_name= 'T_set', shape=(1,1))

    model.set_expression('T_diff', (T_FE[-1] - T_set))
    #model.set_expression('v_L', v_L)
    #model.set_expression('U_F', Qd)
    #model.set_expression('mdotC', mDotC(a_E, T_FE))
    #model.set_expression('mdotE', mDotE(a_E, T_FE))
    # return the complete model
    model.setup()
    return model


