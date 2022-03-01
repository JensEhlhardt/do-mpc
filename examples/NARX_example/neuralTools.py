from casadi import *


def createMultiLayerPerceptron(nInputs, nOutputs, layer, actfun="tanh"):
    nodes = np.hstack((nInputs, layer, nOutputs))
    p_list = []
    #b_list = []
    layer_list = []
    x = SX.sym("x_0", nInputs, 1)
    for iLayer in range(1,len(nodes)):
        w = SX.sym("w_"+str(iLayer), nodes[iLayer],nodes[iLayer-1])
        b = SX.sym("b_"+str(iLayer), nodes[iLayer], 1)

        if iLayer == 1:
            x_i = w @ x + b
            p_list = vertcat(w.reshape((w.numel(), 1)), b)
            #b_list = b
        else:
            x_i = w @ x_i + b
            p_list = vertcat(p_list, w.reshape((w.numel(), 1)), b)
            #b_list = vertcat(b_list, b)
        if iLayer < (len(nodes)-1):
            x_i = tanh(x_i)
        layer_list.append(x_i)

#    params = vertcat(w_list, b_list)
    params = p_list
    net_fun = Function('net', [x, params], [x_i])#, ["x", "w"], ["y"])

    net_dict = {"fun":net_fun,
                "inputs":x,
                "layers":layer_list,
                "weights":params,
                }
    return net_dict

def assembleNarxInput(x_old, u_new, narx):
    nD_U = narx["nInputDelay"] + 1
    nD_X = narx["nOutputDelay"] + 1
    n_U = narx["nInputs"]
    n_X = narx["nOutputs"]

    x_new = vertcat(x_old[0:(nD_X * n_X)],
                    u_new,
                    x_old[(nD_X*n_X):-n_U]
                    )
    return x_new

def assembleNarxOutput(x_new, y_new, narx):
    nD_U = narx["nInputDelay"] + 1
    nD_X = narx["nOutputDelay"] + 1
    n_U = narx["nInputs"]
    n_X = narx["nOutputs"]

    y_hat = vertcat(y_new,
                    x_new[0:n_X*(nD_X-1)],
                    x_new[nD_X*n_X:]
                    )
    return y_hat