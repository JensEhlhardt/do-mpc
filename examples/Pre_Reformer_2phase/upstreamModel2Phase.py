


import sys
sys.path.append('../../')
import do_mpc
from casadi import *
import numpy as np
import CoolProp.CoolProp as CP


def template_model(system_type='SX', nD_Evap=40, modelUsage=0):

        model_type = 'continuous'
        model = do_mpc.model.Model(model_type, system_type)

        model.modelUsage = modelUsage
        gammaP = 0.89 # free volume due to packing

        lengthEvap = 2.228  # [m]
        dz_E = lengthEvap/nD_Evap  # [m]
        dr_E = 0.0036  # [m]
        r_E = 0.0212 - dr_E  # [m]

        A_z_E = r_E**2 * np.pi * gammaP  # [m²] pseudo cross sectional area of the free volume inside the pipe
        A_r_E = 2 * r_E * np.pi * dz_E  # [m²] heat transfer area between wall and fluid
        A_z_W = ((r_E + dr_E)**2 - r_E**2) * np.pi  # [m²] cross sectional area of the wall
        A_r_W = (r_E + dr_E) * 2 * np.pi * dz_E  # [m²] transfer area between wall and isolation

        P = 40e5  # [Pa] pressure

        D_W = 15  # [] heat
        rhoW = 7.87e6  # [g/m³] density of the wall
        cpW = 0.5  # [J/g/k] heat capacity of the wall
        T_Env = 20  # [°C] environmental temperature
        kEvap = 1.4  # [W/m²] heat loss coefficient

        rhoL = CP.PropsSI("D", "Q", 0, "P", P, "Water") * 1000  # [g/m³] # density of the liquid phase
        rhoV = CP.PropsSI("D", "Q", 1, "P", P, "Water") * 1000  # [g/m³] # density of the vapor phase
        cpL = CP.PropsSI("C", "Q", 0, "P", P, "Water") / 1000  # [j/g/K] heat capacity of the liquid phase
        cpV = CP.PropsSI("C", "Q", 1, "P", P, "Water") / 1000  # [j/g/K] heat capacity of the vapor phase
        beta = 0.1  # [1/s] Lee-model parameter
        S_L = (rhoL / rhoV) ** (1/1)  # [-] slip ratio
        hL = CP.PropsSI("H", "Q", 0, "P", P, "Water") / 1000  # j/g
        hV = CP.PropsSI("H", "Q", 1, "P", P, "Water") / 1000  # j/g
        hLV = hV - hL  # [j/g] heat of evaporation
        T_sat = CP.PropsSI("T", "Q", 0, "P", P, "Water")-273.15 - 20 # [°C] saturation temperature
        U_L = 2000  # [] heat transfer coefficient liquid phase
        U_V = 1000   # [] heat transfer coefficient vapor phase

        ## set the variables
        a_E = model.set_variable(var_type='_x', var_name='a_E', shape=(nD_Evap, 1))  # volume fraction of the liquid
        T_FE = model.set_variable(var_type='_x', var_name='T_FE', shape=(nD_Evap, 1))  # pseudo fluid temperature
        T_WE = model.set_variable(var_type='_x', var_name='T_WE', shape=(nD_Evap, 1))  # wall temperature


        # set the inputs
        Qdot_E = model.set_variable(var_type='_u', var_name='Qdot_E', shape=(1, 1))  # [%] heating power percentage evaporator
        mdot_W = model.set_variable(var_type='_u', var_name='mdot_W', shape=(1, 1))  # [g/s] inlet flow of water
        mdot_N = model.set_variable(var_type='_u', var_name='mdot_N', shape=(1, 1))  # [g/s] inlet flow of naphtha

        # set params
        if modelUsage != 0:
            cpL = model.set_variable(var_type='_p', var_name='cpL', shape=(1, 1))
            cpV = model.set_variable(var_type='_p', var_name='cpV', shape=(1, 1))
            kEvap = model.set_variable(var_type='_p', var_name='kEvap', shape=(1, 1))
            #U_L = model.set_variable(var_type='_p', var_name='U_L', shape=(1, 1))
            #U_V = model.set_variable(var_type='_p', var_name='U_V', shape=(1, 1))
            hLV = model.set_variable(var_type='_p', var_name='hLV', shape=(1, 1))

            hLV = hLV * 1000
            # i4 sind die parameter in _rhs_fun
        ## define helper functions #####################################################################################
        # define some dummy variables
        x  = SX.sym('x')
        aE = SX.sym('aE')
        TF = SX.sym('TF')
        TS = SX.sym('TS')

        # define helper functions
        ReLU = Function('ReLU', [x], [fmax(x, 0)])
        #ReLU = Function('soft', [x], [(x + sqrt(x**2 + 1e-5))/2])

        mDotE = Function('mDotE', [aE, TF, TS], [ReLU(beta * aE * (TF - TS)/TS)]) # [1/s] evaporation rate based on LEE model
        mDotC = Function('mDotC', [aE, TF, TS], [ReLU(beta * (1-aE) * (TS - TF) / TS)])  # [1/s] condensation rate based on LEE model

        cpF   = Function('cpF', [aE], [aE * cpL + (1-aE) * cpV])  # mixture heat capacity
        rhoF  = Function('rhoF', [aE], [aE * rhoL + (1-aE) * rhoV])  # mixture density
        U_F   = Function('U_F', [aE], [aE * U_L + (1-aE) * U_V])  # mixture heat transfer coefficient

        ## define RHS ##################################################################################################
        # predefine the RHS
        da_E = SX.sym('da_E',  nD_Evap, 1)
        dT_FE = SX.sym('dT_FE', nD_Evap, 1)
        dT_WE = SX.sym('dT_WE', nD_Evap, 1)

        # calculate the heating power from MV
        Qdot_E = Qdot_E * 3600 / 100  # [W]

        # calculate the liquid phase velocity
        v_L = (mdot_N + mdot_W) / rhoL * 1000 / 3600 / A_z_E

        # loop over the discretisation points
        for i in range(0, nD_Evap):
            # compute the mixture properties at the ith - discretisation point
            cpF_i = cpF(a_E[i])
            rhoF_i = rhoF(a_E[i])
            U_F_i = U_F(a_E[i])

            # compute the mass streams due to condensation and evaporation
            mDotE_i = mDotE(a_E[i], T_FE[i], T_sat) * rhoL
            mDotC_i = mDotC(a_E[i], T_FE[i], T_sat) * rhoV

            # get the heating power i
            iZ = i * dz_E
            Qdot_E_i = casadi.if_else((iZ > 0.1105 and iZ < 0.764) or (1.464 < iZ and iZ < 2.164),
                                      Qdot_E * dz_E / 0.636, 0)

            # calculate the spatial derivatives
            if i != 0: # 1-order BDF
                da_dz_i = (a_E[i] - a_E[i-1]) / dz_E
                dTF_dz_i = (T_FE[i] - T_FE[i-1]) / dz_E


            if i != 0 and i != (nD_Evap-1): # 2-order CDF
                dTW_dz_i = (T_WE[i+1] - 2*T_WE[i] + T_WE[i-1]) / dz_E**2


            # volume fraction
            if i != 0:
                da_E[i] = - v_L * da_dz_i + (mDotC_i - mDotE_i) / rhoL

            # fluid temperature
            if i != 0:
                V_i  = A_z_E * dz_E
                delta_i = v_L * A_z_E * (rhoL * cpL - S_L * rhoV * cpV)
                gamma_i = v_L * A_z_E * S_L * rhoV * cpV
                dT_FE[i] = -(a_E[i] * delta_i + gamma_i) * dTF_dz_i / (rhoF_i * cpF_i * A_z_E) \
                    - delta_i * T_FE[i] * da_dz_i / (rhoF_i * cpF_i * A_z_E) \
                    - 1 * T_FE[i] * (cpF_i * (rhoL-rhoV) * V_i + rhoF_i * V_i * (cpL - cpV)) * da_E[i] / (rhoF_i * cpF_i * V_i) \
                    - (mDotE_i - mDotC_i) * hLV / (cpF_i * rhoF_i) \
                    + U_F_i * A_r_E * (T_WE[i] - T_FE[i]) / (cpF_i * rhoF_i * A_z_E * dz_E)

            # wall temperature
            if i != 0 and i != (nD_Evap-1):
                dT_WE[i] = Qdot_E_i / (cpW * rhoW * A_z_W * dz_E) \
                        + D_W * dTW_dz_i / (rhoW * cpW) \
                        - U_F_i * A_r_E * (T_WE[i] - T_FE[i]) / (cpW * rhoW * A_z_W * dz_E) \
                        - kEvap * A_r_W * (T_WE[i] - T_Env) / (cpW * rhoW * A_z_W * dz_E)


        ## Set boundary conditions #####################################################################################

        # volume fraction at i = 0
        da_E[0] = -v_L * (a_E[0] - 1) / dz_E

        # fluid temperature at i = 0
        cpF_i = cpF(a_E[0])
        rhoF_i = rhoF(a_E[0])
        U_F_i  = U_F(a_E[0])

        dT_FE[0] = -v_L * (T_FE[0] - T_Env)/dz_E \
                    + U_F_i * A_r_E * (T_WE[0] - T_FE[0]) / (cpF_i * rhoF_i * A_z_E * dz_E)

        # wall temperature at i = 0
        dT_WE[0] = D_W * 2 * (T_WE[1] - T_WE[0]) / dz_E**2 / (cpW * rhoW ) \
                   - U_F_i * A_r_E * (T_WE[0] - T_FE[0]) / (cpW * rhoW * A_z_W * dz_E) \
                   - kEvap * A_r_W * (T_WE[0] - T_Env) / (cpW * rhoW * A_z_W * dz_E)

        # wall temperature at i = nD_Evap-1
        U_F_i = U_F(a_E[-1])
        dT_WE[-1] = D_W * 2 * (T_WE[-2] - T_WE[-1]) / dz_E**2 / (cpW * rhoW ) \
                   - U_F_i * A_r_E * (T_WE[-1] - T_FE[-1]) / (cpW * rhoW * A_z_W * dz_E) \
                   - kEvap * A_r_W * (T_WE[-1] - T_Env) / (cpW * rhoW * A_z_W * dz_E)


        # set some auxillaryies
        #model.set_expression('v_L', v_L)
        model.set_expression('mDot', mDotE(a_E, T_FE, T_sat))
        model.set_expression('Qdot', Qdot_E * dz_E / 0.636)
        ## setup and return of the model ###############################################################################

        # set RHS in the model object
        model.set_rhs('a_E', da_E)
        model.set_rhs('T_FE', dT_FE)
        model.set_rhs('T_WE', dT_WE)

        # construct model
        model.setup()

        # return object
        return model

def getRawInitialState(model):
    nD = int(model.n_x / len(model._x.keys()))

    x0 = vertcat(np.ones([nD, 1]), np.ones([2*nD, 1]) * 50)

    return x0