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
    nD_Evap = 40
    # Scales
    lengthEvap = 2.228  # [m]
    dz_E = lengthEvap/nD_Evap  # [m]
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
    S_L = (rhoL / rhoV)**(1/3)
    # Naphtha
    #P = 10e5
    nStr = "SRK::CycloHexane"
    rhoLN = CP.PropsSI("D", "Q", 0, "P", P, nStr) * 1000  # g/m³
    rhoVN = CP.PropsSI("D", "Q", 1, "P", P, nStr) * 1000  # g/m³
    cp_LN = CP.PropsSI("C", "Q", 0, "P", P, nStr) / 1000  # j/g/K
    cp_VN = CP.PropsSI("C", "Q", 1, "P", P, nStr) / 1000  # j/g/K
    T_satN = CP.PropsSI("T", "Q", 0, "P", P, nStr) - 273.15  # °C

    hL = CP.PropsSI("H", "Q", 0, "P", P, nStr) / 1000  # j/g
    hV = CP.PropsSI("H", "Q", 1, "P", P, nStr) / 1000  # j/g
    hLVN = hV - hL
    #hLVN = hLV

    U_L  = 800
    U_V  = 400


    cp_W = 0.5
    cp_P = cp_W
    S_V  = 1000
    D_W  = 15
    T_env = 20
    kEvap = 1
    # create state variables
    a_E  = model.set_variable(var_type='_x', var_name='a_E',  shape=(nD_Evap, 1))
    a_EN = model.set_variable(var_type='_x', var_name='a_EN', shape=(nD_Evap, 1))
    a_VN = model.set_variable(var_type='_x', var_name='a_VN', shape=(nD_Evap, 1))
    T_FE = model.set_variable(var_type='_x', var_name='T_FE', shape=(nD_Evap, 1))
    T_WE = model.set_variable(var_type='_x', var_name='T_WE', shape=(nD_Evap, 1))
    T_PE = model.set_variable(var_type='_x', var_name='T_PE', shape=(nD_Evap, 1))
    #V_F  = model.set_variable(var_type='_x', var_name='V_F', shape=(1, 1))

    # create input variables
    Qdot_E = model.set_variable(var_type='_u', var_name='Qdot_E', shape=(1, 1))
    mdot_W = model.set_variable(var_type='_u', var_name='mdot_W', shape=(1, 1))
    mdot_N = model.set_variable(var_type='_u', var_name='mdot_N', shape=(1, 1))
    #mdot_W = 2.55
    #mdot_N = 0.9
    ## create model parameters
    #hLVN = model.set_variable(var_type='_p', var_name='hLVN', shape=(1, 1))
    cp_LN = model.set_variable(var_type='_p', var_name='cpLN', shape=(1, 1))
    cp_VN = model.set_variable(var_type='_p', var_name='cpVN', shape=(1, 1))
    # a_In = model.set_variable(var_type = '_tvp', var_name='a_In', shape = (1,1))
    # T_In = model.set_variable(var_type = '_tvp', var_name='T_In', shape = (1,1))
    a_In = mdot_W/rhoL / (mdot_N/rhoLN + mdot_W/rhoL)
    a_InN = mdot_N/rhoLN / (mdot_N/rhoLN + mdot_W/rhoL)
    T_In = 20
    # some calculations
    v_L = (mdot_N/rhoLN + mdot_W/rhoL) * 1000 / 3600 / A_z_E

    # a = (rhoL * mdot_W + rhoLN * mdot_N) / (mdot_N + mdot_W)
    # b = (rhoV * mdot_W + rhoVN * mdot_N) / (mdot_N + mdot_W)
    # S_L = a/b

    # create RHS
    #some helper functions
    x  = SX.sym('x')
    aE = SX.sym('aE')
    aE2 = SX.sym('aE2')
    aEN = SX.sym('aEN')
    aVN = SX.sym('aVN')
    TF = SX.sym('TF')
    TS = SX.sym('TS')
    r_s = 0.1

    softplus = Function('soft', [x], [(x + sqrt(x**2 + 1e-5))/2])
    softplus = Function('relu', [x], [fmax(x,0)])
    #softplus = Function('warum', [x], [0.0])
    mDotE = Function('mDotE', [aE, TF, TS], [softplus(r_s * aE * (TF-TS)/TS)])
    #mDotE = Function('mDotE', [aE, TF, TS], [0])
    #mDotC = Function('mDotC', [aE, aE2, TF], [softplus(r_s * rhoV * (1-aE-aE2) * (T_sat-TF)/T_sat)])
    mDotC = Function('mDotC', [aE, aE2, TF], [0])

   # v_F    = Function('v_F', [aE, aEN], [v_L * ((aE+aEN)*(1-S_L)+S_L)])
    v_F    = Function('v_F', [aE, aEN], [((aE+aEN)*rhoL*v_L + (1-aE-aEN)*rhoV*S_L*v_L)/((aE+aEN)*rhoL + (1-aE-aEN)*rhoV)])
    U_F   = Function('U_F', [aE, aEN], [(aE+aEN) * U_L + (1-aE-aEN) * U_V])
    cp_F = Function('cp_F', [aE, aEN, aVN], [(aE * rhoL * cp_L + aEN * rhoLN * cp_LN + aVN * cp_VN * rhoVN + (1-aE-aEN-aVN) * rhoV * cp_V) \
                                        /(aE * rhoL + aEN * rhoLN + aVN * rhoVN + (1-aE-aEN-aVN) * rhoV)])
    cp_F  = Function('cp_F', [aE, aEN, aVN], [aE * cp_L + aEN * cp_LN + aVN * cp_VN + (1-aE-aEN-aVN) * cp_V])
    rho_F = Function('rho_F', [aE, aEN, aVN], [aE * rhoL + aEN * rhoLN + aVN * rhoVN + (1-aE-aEN-aVN) * rhoV])



    # Evap
    da_E  = SX.sym('da_E',  nD_Evap, 1)
    da_EN = SX.sym('da_EN', nD_Evap, 1)
    da_VN = SX.sym('da_VN', nD_Evap, 1)
    dT_FE = SX.sym('dT_FE', nD_Evap, 1)
    dT_PE = SX.sym('dT_PE', nD_Evap, 1)
    dT_WE = SX.sym('dT_WE', nD_Evap, 1)

    # get correct heating power
    #Qdot_E = -0.318 * Qdot_E**2 + 64.163 * Qdot_E + 275.54
    Qdot_E = Qdot_E/100 * 3600
    # get the heated zone in the evaporator

    heatZone1Start   = 110.5
    heatZone1End     = 764
    heatZone2Start   = 1464
    heatZone2End     = 2164
    # define pdes

    #dV_F = (v_L - V_F) / 1000
    for i in range(0, nD_Evap):
        if i != 0:
            # averaged properties
            cpF = cp_F(a_E[i], a_EN[i], a_VN[i])
            vF  = v_F(a_E[i], a_EN[i])
            rhoF = rho_F(a_E[i], a_EN[i], a_VN[i])
            mDotE_i = mDotE(a_E[i], T_FE[i], T_sat) * rhoL
            mDotE_iN = mDotE(a_EN[i], T_FE[i], T_satN) * rhoLN
            # get actual heating
            iZ = i * dz_E
            Qdot_E_i = casadi.if_else((iZ > 0.1105 and iZ< 0.764) or (1.464 < iZ and iZ < 2.164), Qdot_E * dz_E / 0.636, 0)
            #Qdot_E_i = Qdot_E/lengthEvap * dz_E
            # volume fractions
            da_EN[i] = -v_L * (a_EN[i] - a_EN[i-1])/dz_E + (mDotC(a_EN[i], a_E[i], T_FE[i]) - mDotE_iN)/rhoLN
            da_E[i] = -v_L * (a_E[i]-a_E[i-1])/dz_E + (mDotC(a_E[i], a_EN[i], T_FE[i]) - mDotE_i)/rhoL
            da_VN[i] = -v_L * S_L * (a_VN[i]-a_VN[i-1])/dz_E + mDotE_iN/rhoVN
            # fluid temperature
            dT_FE[i] = -vF * (T_FE[i] - T_FE[i-1])/dz_E \
                + A_r_E * U_F(a_E[i], a_EN[i]) * (T_WE[i]-T_FE[i]) / (cpF * rhoF * A_z_E * dz_E) \
                - (mDotE_i - mDotC(a_E[i], a_EN[i], T_FE[i])) * hLV / (cpF * rhoF) \
                - (mDotE_iN - mDotC(a_EN[i], a_E[i], T_FE[i])) * hLVN / (cpF * rhoF) \
                - S_V * U_F(a_E[i], a_EN[i]) * (T_FE[i] - T_PE[i])/ (cpF * rhoF * gammaP)
        # packing temperature
        dT_PE[i] = S_V * U_F(a_E[i], a_EN[i]) * (T_FE[i] - T_PE[i]) / (cp_P * rhoP * (1-gammaP))
        # wall temperature
        if i != nD_Evap-1 and i != 0:
            dT_WE[i] = Qdot_E_i / (cp_W * rhoW * A_z_W * dz_E) - kEvap * (T_WE[i] - T_env) / (rhoW * cp_W * A_z_W)\
                - A_r_E * U_F(a_E[i], a_EN[i]) * (T_WE[i]-T_FE[i]) / (cp_W * rhoW * A_z_W * dz_E) \
                + D_W / (rhoW * cp_W) * (T_WE[i+1] - 2*T_WE[i] + T_WE[i-1])/dz_E**2
    ## define boundaries
    # volume fractions
    da_VN[0] = -v_L * S_L * (a_VN[0] - 0)/dz_E - (mDotC(a_EN[0], 0, T_FE[0]) + mDotE(a_EN[0], T_FE[0], T_satN))/rhoVN
    da_EN[0] = -v_L * (a_EN[0] - a_InN)/dz_E + (mDotC(a_EN[0], a_E[0], T_FE[0]) - mDotE(a_EN[0], T_FE[0], T_satN))/rhoLN
    da_E[0] = -v_L * (a_E[0] - a_In)/dz_E + (mDotC(a_E[0], a_EN[0], T_FE[0]) - mDotE(a_E[0], T_FE[0], T_sat))/rhoL
    # Fluid temperature
    # averaged properties
    cpF = cp_F(a_E[0], a_EN[0], a_VN[0])
    vF = v_F(a_E[0], a_EN[0])
    rhoF = rho_F(a_E[0], a_EN[0], a_VN[0])

    dT_FE[0] = -vF * (T_FE[0] - T_In)/dz_E + A_r_E * U_F(a_E[0], a_EN[0]) * (T_WE[0]-T_FE[0]) / (cpF * rhoF * A_z_E * dz_E) \
                - (mDotE(a_E[0], T_FE[0], T_sat)*hLV + mDotE(a_EN[0], T_FE[0], T_satN)*hLVN) / (cpF * rhoF) \
                - S_V * U_F(a_E[0], a_EN[0]) * (T_FE[0] - T_PE[0]) / (cpF * rhoF)
    # Wall temperature
    dT_WE[0] = 2 * D_W / (rhoW * cp_W) * (T_WE[1]-T_WE[0])/dz_E**2 + Qdot_E_i/(cp_W * rhoW * A_z_W * dz_E) \
           - A_r_E * U_F(a_E[0], a_EN[-1]) * (T_WE[0] - T_FE[0]) / (cp_W * rhoW * A_z_W * dz_E)
    dT_WE[-1] = 2 * D_W / (rhoW * cp_W) * (T_WE[-2] - T_WE[-1]) / dz_E**2 + Qdot_E_i/(cp_W * rhoW * A_z_W * dz_E) \
           - A_r_E * U_F(a_E[-1], a_EN[-1]) * (T_WE[-1] - T_FE[-1]) / (cp_W * rhoW * A_z_W * dz_E)

    model.set_rhs("a_E",  da_E)
    model.set_rhs("a_EN", da_EN)
    model.set_rhs("a_VN", da_VN)
    model.set_rhs("T_FE", dT_FE)
    model.set_rhs("T_WE", dT_WE)
    model.set_rhs("T_PE", dT_PE)
#    model.set_rhs("V_F", dV_F)
    mdot_N_set = model.set_variable(var_type='_tvp', var_name= 'mdotN_set', shape=(1,1))

    model.set_expression('mdotN_diff', (mdot_N - mdot_N_set))
    #model.set_expression('alpha_tot', (a_E + a_EN + a_VN))
    a_V = 1 - a_E[-1] - a_EN[-1] - a_VN[-1]

    s_CC =  (rhoV * 12)/(rhoVN * 0.85 * 18)
    model.set_expression('S_C', fmin((a_V / a_VN[-1] + 0.00001  ) * s_CC, 5) )
    #model.set_expression('v_L', v_L)
    model.set_expression('a_V', a_V)
    #model.set_expression('U_F', Qd)
    #model.set_expression('mdotC', mDotC(a_E, T_FE))
    #model.set_expression('mdotE', mDotE(a_E, T_FE))
    # return the complete model
    model.setup()
    return model


