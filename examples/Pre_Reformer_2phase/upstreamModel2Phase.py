


import sys
sys.path.append('../../')
import do_mpc
from casadi import *
import numpy as np
import CoolProp.CoolProp as CP


def template_model(system_type='SX', nD_Evap=40, nD_Pipe=10, nD_Super=20, nD_Pre = 30, modelUsage=0):

        model_type = 'continuous'
        model = do_mpc.model.Model(model_type, system_type)

        model.modelUsage = modelUsage
        gammaP = 0.89 # free volume due to packing


        lengthPipe = 2.8 # [m]
        lengthEvap = 2.228  # [m]
        lengthSuper = 1.6  # [m]
        lengthPre = 1.64 # [m]

        dz_E = lengthEvap/nD_Evap  # [m]
        dz_P = lengthPipe/nD_Pipe # [m]
        dz_S = lengthSuper/nD_Super # [m]
        dz_R = lengthPre/nD_Pre # [m]

        dr_P = 0.0015  #[m]
        dr_E = 0.0036  # [m]
        dr_R = 0.004 # [m]
        r_E = 0.0212 - dr_E  # [m]
        r_P = 0.012/2 - dr_P  # [m]
        r_R = 0.0165 - dz_R # [m]
        A_z_E = r_E**2 * np.pi * gammaP  # [m²] pseudo cross sectional area of the free volume inside the pipe
        A_r_E = 2 * r_E * np.pi * dz_E  # [m²] heat transfer area between wall and fluid
        A_z_W = ((r_E + dr_E)**2 - r_E**2) * np.pi  # [m²] cross sectional area of the wall
        A_r_W = (r_E + dr_E) * 2 * np.pi * dz_E  # [m²] transfer area between wall and isolation
        A_z_P = r_E**2 * np.pi * (1-gammaP) # [m²] pseudo cross sectional area of the packing

        A_P_z = r_P ** 2 * np.pi
        A_P_r = r_P * 2 * np.pi * dz_P
        A_P_w = ((r_P + dr_P) ** 2 - r_P**2) * np.pi

        A_S_z = r_E ** 2 * np.pi * gammaP
        A_S_r = r_E * 2 * np.pi * dz_S
        A_S_w = A_z_W
        A_S_wr = (r_E + dr_E) * 2 * np.pi * dz_S

        A_R_z = r_R ** 2 * np.pi * 0.45
        A_R_P = r_R ** 2 * np.pi * (1-0.45)
        A_R_w = ((r_R+dr_R)**2 - r_R**2) * np.pi
        A_R_r = r_R * 2 * np.pi * dz_R
        A_R_wr = (r_R + dr_R) * 2 * np.pi * dz_R

        A_E_P = A_z_E / A_P_z
        A_E_R = A_z_E / A_R_z

        P = 30e5  # [Pa] pressure

        D_W = 15  # [] heat
        rhoW = 7.87e6  # [g/m³] density of the wall
        cpW = 0.5  # [J/g/k] heat capacity of the wall
        T_Env = 20  # [°C] environmental temperature
        kEvap = 2  # [W/m²] heat loss coefficient
        kPipe = 30
        kSuper = 0
        kPre = 0
        S_V = 1000  # [m²/m³] specific surface packing

        rhoL = CP.PropsSI("D", "Q", 0, "P", P, "Water") * 1000  # [g/m³] # density of the liquid phase
        rhoV = CP.PropsSI("D", "Q", 1, "P", P, "Water") * 1000  # [g/m³] # density of the vapor phase
        cpL = CP.PropsSI("C", "Q", 0, "P", P, "Water") / 1000  # [j/g/K] heat capacity of the liquid phase
        cpV = CP.PropsSI("C", "Q", 1, "P", P, "Water") / 1000  # [j/g/K] heat capacity of the vapor phase
        beta = 0.1  # [1/s] Lee-model parameter
        S_L = (rhoL / rhoV) ** (1/1)  # [-] slip ratio
        hL = CP.PropsSI("H", "Q", 0, "P", P, "Water") / 1000  # j/g
        hV = CP.PropsSI("H", "Q", 1, "P", P, "Water") / 1000  # j/g
        hLV = hV - hL  # [j/g] heat of evaporation
        T_sat = CP.PropsSI("T", "Q", 0, "P", P, "Water")-273.15  # [°C] saturation temperature
        U_L = 1000  # [] heat transfer coefficient liquid phase
        U_V = 200   # [] heat transfer coefficient vapor phase

        ## set the variables
        a_E = model.set_variable(var_type='_x', var_name='a_E', shape=(nD_Evap, 1))  # volume fraction of the liquid
        T_FE = model.set_variable(var_type='_x', var_name='T_FE', shape=(nD_Evap, 1))  # pseudo fluid temperature
        T_WE = model.set_variable(var_type='_x', var_name='T_WE', shape=(nD_Evap, 1))  # wall temperature
        T_PE = model.set_variable(var_type='_x', var_name='T_PE', shape=(nD_Evap, 1))  # packing temperature

        a_P  = model.set_variable(var_type='_x', var_name='a_P', shape=(nD_Pipe, 1))
        T_FP = model.set_variable(var_type='_x', var_name='T_FP', shape=(nD_Pipe, 1))  # pseudo fluid temperature
        T_WP = model.set_variable(var_type='_x', var_name='T_WP', shape=(nD_Pipe, 1))  # wall temperature

        a_S = model.set_variable(var_type='_x', var_name='a_S', shape=(nD_Super, 1))  # volume fraction of the liquid
        T_FS = model.set_variable(var_type='_x', var_name='T_FS', shape=(nD_Super, 1))  # pseudo fluid temperature
        T_WS = model.set_variable(var_type='_x', var_name='T_WS', shape=(nD_Super, 1))  # wall temperature
        T_PS = model.set_variable(var_type='_x', var_name='T_PS', shape=(nD_Super, 1))  # packing temperature

        a_R = model.set_variable(var_type='_x', var_name='a_R', shape=(nD_Pre, 1))  # volume fraction of the liquid
        T_FR = model.set_variable(var_type='_x', var_name='T_FR', shape=(nD_Pre, 1))  # pseudo fluid temperature
        T_WR = model.set_variable(var_type='_x', var_name='T_WR', shape=(nD_Pre, 1))  # wall temperature
        #T_PR = model.set_variable(var_type='_x', var_name='T_PS', shape=(nD_Evap, 1))  # packing temperature

        # set the inputs
        Qdot_E = model.set_variable(var_type='_u', var_name='Qdot_E', shape=(1, 1))  # [%] heating power percentage evaporator
        mdot_W = model.set_variable(var_type='_u', var_name='mdot_W', shape=(1, 1))  # [g/s] inlet flow of water
        mdot_N = model.set_variable(var_type='_u', var_name='mdot_N', shape=(1, 1))  # [g/s] inlet flow of naphtha
        Qdot_S = model.set_variable(var_type='_u', var_name='Qdot_P', shape=(1, 1))
        Qdot_R = model.set_variable(var_type='_u', var_name='Qdot_R', shape=(1, 1))

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
        dT_PE = SX.sym('dT_PE', nD_Evap, 1)

        da_P = SX.sym('da_P', nD_Pipe, 1)
        dT_FP = SX.sym('dT_FP', nD_Pipe, 1)
        dT_WP = SX.sym('dT_WP', nD_Pipe, 1)

        da_S = SX.sym('da_E', nD_Super, 1)
        dT_FS = SX.sym('dT_FE', nD_Super, 1)
        dT_WS = SX.sym('dT_WE', nD_Super, 1)
        dT_PS = SX.sym('dT_PE', nD_Super, 1)

        da_R = SX.sym('da_E', nD_Pre, 1)
        dT_FR = SX.sym('dT_FE', nD_Pre, 1)
        dT_WR = SX.sym('dT_WE', nD_Pre, 1)

        # calculate the heating power from MV
        Qdot_E = Qdot_E * 3600 / 100  # [W]
        Qdot_S = Qdot_S * 3600 / 100
        Qdot_R = Qdot_R * 3000 / 100
        #Qdot_E = -0.318 * (Qdot_E ) ** 2 + 64.163 * (Qdot_E) + 275.54
        # calculate the liquid phase velocity
        v_L = (mdot_N + mdot_W) / rhoL * 1000 / 3600 / A_z_E

        # loop over the discretisation points
        for i in range(0, nD_Evap):
            # compute the mixture properties at the ith - discretisation point
            cpF_i = cpF(a_E[i])
            rhoF_i = rhoF(a_E[i])
            U_F_i = U_F(a_E[i])

            if i < nD_Pipe:
                cpP_i = cpF(a_P[i])
                rhoP_i = rhoF(a_P[i])
                U_P_i = U_F(a_P[i])
            if i < nD_Super:
                cpS_i = cpF(a_S[i])
                rhoS_i = rhoF(a_S[i])
                U_S_i = U_F(a_S[i])
            if i < nD_Pre:
                cpR_i = cpF(a_R[i])
                rhoR_i = rhoF(a_R[i])
                U_R_i = U_F(a_R[i])

            # compute the mass streams due to condensation and evaporation
            mDotE_i = mDotE(a_E[i], T_FE[i], T_sat) * rhoL
            mDotC_i = mDotC(a_E[i], T_FE[i], T_sat) * rhoV
            if i < nD_Pipe:
                mDotC_Pi = mDotC(a_P[i], T_FP[i], T_sat) * rhoL
            if i < nD_Super:
                mDotC_Si = mDotC(a_S[i], T_FS[i], T_sat) * rhoV
                mDotE_Si = mDotE(a_S[i], T_FS[i], T_sat) * rhoL
            if i < nD_Pre:
                mDotE_Ri = mDotE(a_R[i], T_FR[i], T_sat) * rhoL
                mDotC_Ri = mDotC(a_R[i], T_FR[i], T_sat) * rhoV

            # get the heating power i
            iZ = i * dz_E
            Qdot_E_i = casadi.if_else((iZ > 0.1105 and iZ < 0.764) or (1.464 < iZ and iZ < 2.164),
                                      Qdot_E * dz_E / 0.636, 0)
            iZ = i * dz_S
            Qdot_S_i = casadi.if_else((iZ > 0.12 and iZ < 1.48), Qdot_S/1.3 * dz_S, 0)

            iZ = i * dz_S
            Qdot_R_i = casadi.if_else((iZ > 0.24 and iZ < 1.64), Qdot_R / 1.4 * dz_R, 0)

            # calculate the spatial derivatives
            if i != 0: # 1-order BDF
                da_dz_i  = (a_E[i] - a_E[i - 1]) / dz_E
                if i < nD_Pipe:
                    daP_d_i  = (a_P[i] - a_P[i - 1]) / dz_P
                if i < nD_Super:
                    daS_dz_i = (a_S[i] - a_S[i - 1]) / dz_S
                if i < nD_Pre:
                    daR_dz_i = (a_R[i] - a_R[i - 1]) / dz_R

                dTF_dz_i   = (T_FE[i] - T_FE[i-1]) / dz_E
                if i < nD_Pipe:
                    dTFP_dz_i  = (T_FP[i] - T_FP[i-1]) / dz_P
                if i < nD_Super:
                    dT_FS_dz_i = (T_FS[i] - T_FS[i-1]) / dz_S
                if i < nD_Pre:
                    dT_FR_dz_i = (T_FR[i] - T_FR[i-1]) / dz_R

            if i != 0 and i != (nD_Evap-1): # 2-order CDF
                dTW_dz_i   = (T_WE[i+1] - 2*T_WE[i] + T_WE[i-1]) / dz_E**2
                if i < (nD_Pipe-1):
                    dTPW_dz_i  = (T_WP[i+1] - 2*T_WP[i] + T_WP[i-1]) / dz_P**2
                if i < (nD_Super-1):
                    dT_WS_dz_i = (T_WS[i+1] - 2*T_WS[i] + T_WS[i-1]) / dz_S**2
                if i < (nD_Pre-1):
                    dT_WR_dz_i = (T_WR[i+1] - 2*T_WR[i] + T_WR[i-1]) / dz_R**2

            # volume fraction
            if i != 0:
                da_E[i] = - v_L * da_dz_i + (mDotC_i - mDotE_i) / rhoL
                if i < nD_Pipe:
                    da_P[i] = - v_L * A_E_P * daP_d_i  + mDotC_Pi / rhoL
                if i < nD_Super:
                    da_S[i] = - v_L * daS_dz_i + (mDotC_Si - mDotE_Si) / rhoL
                if i < nD_Pre:
                    da_R[i] = - v_L * A_E_R * daR_dz_i + (mDotC_Ri - mDotE_Ri) / rhoL

            # fluid temperature
            if i != 0:
                V_i  = A_z_E * dz_E
                delta_i = v_L * A_z_E * (rhoL * cpL - S_L * rhoV * cpV)
                gamma_i = v_L * A_z_E * S_L * rhoV * cpV
                dT_FE[i] = -(a_E[i] * delta_i + gamma_i) * dTF_dz_i / (rhoF_i * cpF_i * A_z_E) \
                    - delta_i * T_FE[i] * da_dz_i / (rhoF_i * cpF_i * A_z_E) \
                    - 1 * T_FE[i] * (cpF_i * (rhoL-rhoV) * V_i + rhoF_i * V_i * (cpL - cpV)) * da_E[i] / (rhoF_i * cpF_i * V_i) \
                    - (mDotE_i - mDotC_i) * hLV / (cpF_i * rhoF_i) \
                    + U_F_i * A_r_E * (T_WE[i] - T_FE[i]) / (cpF_i * rhoF_i * A_z_E * dz_E) \
                    - (U_F_i * S_V * A_z_E * dz_E) / (rhoF_i*cpF_i*V_i) * (T_FE[i] - T_PE[i])

                if i < nD_Pipe:
                    dT_FP[i] = -v_L * ((1-a_P[i]) * S_L + S_L) * A_E_P * dTFP_dz_i \
                            + U_P_i * A_P_r * (T_WP[i] - T_FP[i]) / (cpP_i * rhoP_i * A_P_z * dz_P) \
                            + mDotC_Pi * hL / (cpP_i * rhoP_i)

                if i < nD_Super:
                    dT_FS[i] = -v_L * ((1-a_S[i])*S_L + S_L) * dT_FS_dz_i \
                            + U_S_i * A_S_r * (T_WS[i]-T_FS[i]) / (cpS_i * rhoS_i * A_S_z * dz_S) \
                            - (mDotE_Si - mDotC_Si) * hLV / (cpS_i * rhoS_i) \
                            - (U_S_i * S_V * A_S_z * dz_S) * (T_FS[i] - T_PS[i]) / (rhoS_i * cpS_i * A_S_z * dz_S)

                if i < nD_Pre:
                    dT_FR[i] = -v_L * ((1-a_R[i])*S_L + S_L) * A_E_R * dT_FR_dz_i \
                            + U_R_i * A_R_r * (T_WR[i] - T_FR[i]) / (cpR_i * rhoR_i * A_R_z * dz_R) \
                            - 0*(mDotE_Ri - mDotC_Ri) * hLV / (cpR_i * rhoR_i) \

            # wall temperature
            if i != 0 and i != (nD_Evap-1):
                dT_WE[i] = Qdot_E_i / (cpW * rhoW * A_z_W * dz_E) \
                        + D_W * dTW_dz_i / (rhoW * cpW) \
                        - U_F_i * A_r_E * (T_WE[i] - T_FE[i]) / (cpW * rhoW * A_z_W * dz_E) \
                        - kEvap * A_r_W * (T_WE[i] - T_Env) / (cpW * rhoW * A_z_W * dz_E)
                if i < (nD_Pipe-1):
                    dT_WP[i] = D_W * dTPW_dz_i / (rhoW * cpW) \
                            - U_P_i * A_P_r * (T_WP[i] - T_FP[i]) / (cpW * rhoW * A_P_w * dz_P) \
                            - kPipe * A_P_r * (T_WP[i] - T_Env) / (cpW * rhoW * A_P_w * dz_P)
                if i < (nD_Super-1):
                    dT_WS[i] = D_W * dT_WS_dz_i / (rhoW * cpW) \
                            + Qdot_S_i / (cpW * rhoW * A_S_w * dz_S) \
                            - U_S_i * A_S_r * (T_WS[i] - T_FS[i]) / (cpW * rhoW * A_S_w * dz_S) \
                            - kSuper * A_S_wr * (T_WS[i] - T_Env) / (cpW * rhoW * A_S_w * dz_S)
                if i < (nD_Pre-1):
                    dT_WR[i] = D_W * dT_WR_dz_i / (rhoW * cpW) \
                            + Qdot_R_i / (cpW * rhoW * A_r_W * dz_R) \
                            - U_R_i * A_R_r * (T_WR[i] - T_FR[i]) / (cpW * rhoW * A_R_w * dz_R) \
                            - kPre * A_R_wr * (T_WR[i] - T_Env) / (cpW * rhoW * A_R_w * dz_R)
            # packing temperature
            dT_PE[i] = (U_F_i * S_V * A_z_E * dz_E) / (rhoW*cpW*A_z_W*dz_E) * (T_FE[i] - T_PE[i])

            if i < nD_Super:
                dT_PS[i] = (U_S_i * S_V * A_P_z * dz_S) / (rhoW*cpW*A_z_W*dz_S) * (T_FS[i] - T_PS[i])

        ## Set boundary conditions #####################################################################################

        # volume fraction at i = 0
        da_E[0] = -v_L * (a_E[0] - 1) / dz_E
        da_P[0] = -v_L * A_E_P * (a_P[0] - a_E[-1]) / dz_P
        da_S[0] = -v_L * (a_S[0] - a_P[-1]) / dz_S
        da_R[0] = -v_L * A_E_R * (a_R[0] - a_S[-1]) / dz_R

        # fluid temperature at i = 0
        cpF_i = cpF(a_E[0])
        rhoF_i = rhoF(a_E[0])
        U_F_i  = U_F(a_E[0])

        cpP_i = cpF(a_P[0])
        rhoP_i = rhoF(a_P[0])
        U_P_i = U_F(a_P[0])

        cpS_i = cpF(a_S[0])
        rhoS_i = rhoF(a_S[0])
        U_S_i = U_F(a_S[0])

        cpR_i = cpF(a_R[0])
        rhoR_i = rhoF(a_R[0])
        U_R_i = U_F(a_R[0])

        dT_FE[0] = -v_L * (T_FE[0] - T_Env)/dz_E \
                    + U_F_i * A_r_E * (T_WE[0] - T_FE[0]) / (cpF_i * rhoF_i * A_z_E * dz_E)
        dT_FP[0] = -v_L * ((1-S_L)*a_P[0] + S_L) * A_E_P * (T_FP[0] - T_FE[-1]) / dz_P \
                    + U_P_i * A_P_r * (T_WP[0] - T_FP[0]) / (cpP_i * rhoP_i * A_P_z * dz_P)
        dT_FS[0] = -v_L * ((1-S_L)*a_S[0] + S_L) * (T_FS[0] - T_FP[-1]) / dz_S \
                    + U_S_i * A_S_r * (T_WS[0] - T_FS[0]) / (cpS_i * rhoS_i * A_S_z * dz_S)
        dT_FR[0] = -v_L * ((1-S_L)*a_R[0] + S_L) * A_E_R * (T_FR[0] - T_FS[-1]) / dz_R \
                    + U_R_i * A_R_r * (T_WR[0] - T_FR[0]) / (cpR_i * rhoR_i * A_R_z * dz_R)

        # wall temperature at i = 0
        dT_WE[0] = D_W * 2 * (T_WE[1] - T_WE[0]) / dz_E**2 / (cpW * rhoW ) \
                   - U_F_i * A_r_E * (T_WE[0] - T_FE[0]) / (cpW * rhoW * A_z_W * dz_E) \
                   - kEvap * A_r_W * (T_WE[0] - T_Env) / (cpW * rhoW * A_z_W * dz_E)
        dT_WP[0] = D_W * 2 * (T_WP[1] - T_WP[0]) / dz_P**2 / (cpW * rhoW) \
                   - kPipe * A_P_r * (T_WP[0] - T_Env) / (cpW * rhoW * A_P_w + dz_P) \
                   - U_P_i * A_P_r * (T_WP[0] - T_FP[0]) / (cpW * rhoW * A_P_w * dz_P)
        dT_WS[0] = D_W * 2 * (T_WS[1] - T_WS[0]) / dz_S**2 / (cpW * rhoW) \
                   - kSuper * A_S_wr * (T_WS[0] - T_Env) / (cpW * rhoW * A_S_w * dz_S) \
                   - U_S_i * A_S_r * (T_WS[0] - T_FS[0]) / (cpW * rhoW * A_S_w * dz_S)
        dT_WR[0] = D_W * 2 * (T_WR[1] - T_WR[0]) / dz_R**2 / (cpW * rhoW) \
                   - kPre * A_R_wr * (T_WR[0] - T_Env) / (cpW * rhoW * A_R_w * dz_R) \
                   - U_R_i * A_R_r * (T_WR[0] - T_FR[0]) / (cpW * rhoW * A_R_w * dz_R)

        # wall temperature at i = nD_Evap-1
        U_F_i = U_F(a_E[-1])
        U_P_i = U_F(a_P[-1])
        U_S_i = U_F(a_S[-1])
        U_R_i = U_F(a_R[-1])

        dT_WE[-1] = D_W * 2 * (T_WE[-2] - T_WE[-1]) / dz_E**2 / (cpW * rhoW ) \
                   - U_F_i * A_r_E * (T_WE[-1] - T_FE[-1]) / (cpW * rhoW * A_z_W * dz_E) \
                   - kEvap * A_r_W * (T_WE[-1] - T_Env) / (cpW * rhoW * A_z_W * dz_E)
        dT_WP[-1] = D_W * 2 * (T_WP[-2] - T_WP[-1]) / dz_P**2 / (cpW * rhoW) \
                    - kPipe * A_P_r * (T_WP[-1] - T_Env) / (cpW * rhoW * A_P_w * dz_P) \
                    - U_P_i * A_P_r * (T_WP[-1] - T_FP[-1]) / (cpW * rhoW * A_P_w * dz_P)
        dT_WS[-1] = D_W * 2 * (T_WS[-2] - T_WS[-1]) / dz_S**2 / (cpW * rhoW) \
                    - kSuper * A_S_wr * (T_WS[-1] - T_Env) / (cpW * rhoW * A_S_w * dz_S) \
                    - U_S_i * A_S_r * (T_WS[-1] - T_FS[-1]) / (cpW * rhoW * A_S_w * dz_S)
        dT_WR[-1] = D_W * 2 * (T_WR[-2] - T_WR[-1]) / dz_R**2 / (cpW * rhoW) \
                    - kPre * A_R_wr * (T_WR[-1] - T_Env) / (cpW * rhoW * A_R_w * dz_R) \
                    - U_R_i * A_R_r * (T_WR[-1] - T_FR[-1]) / (cpW * rhoW * A_R_w * dz_R)
        # set some auxillaryies
        #model.set_expression('URI_end', U_F(a_R[-1]))
        #model.set_expression('mDot', mDotE(a_E, T_FE, T_sat))
        #model.set_expression('Qdot', Qdot_S)
        ## setup and return of the model ###############################################################################

        # set RHS in the model object
        model.set_rhs('a_E', da_E)
        model.set_rhs('T_FE', dT_FE)
        model.set_rhs('T_WE', dT_WE)
        model.set_rhs('T_PE', dT_PE)

        model.set_rhs('a_P', da_P)
        model.set_rhs('T_FP', dT_FP)
        model.set_rhs('T_WP', dT_WP)

        model.set_rhs('a_S', da_S)
        model.set_rhs('T_FS', dT_FS)
        model.set_rhs('T_WS', dT_WS)
        model.set_rhs('T_PS', dT_PS)

        model.set_rhs('a_R', da_R)
        model.set_rhs('T_FR', dT_FR)
        model.set_rhs('T_WR', dT_WR)

        # construct model
        model.setup()

        # return object
        return model

def getRawInitialState(model, nD_Evap=40, nD_Pipe=10, nD_Super=20, nD_Pre = 30):
    nD = nD_Evap #int(model.n_x / len(model._x.keys()))
    nD_P = nD_Pipe
    nD_S = nD_Super
    nD_R = nD_Pre

    x0 = vertcat(np.ones([nD, 1]), np.ones([3*nD, 1]) * 50, np.ones([nD_P, 1]), np.ones([2*nD_P, 1]) * 50,
                 np.ones([nD_S, 1]), np.ones([3*nD_S, 1]) * 50,
                 np.ones([nD_R, 1]), np.ones([2*nD_R, 1]) * 50)
    return x0
