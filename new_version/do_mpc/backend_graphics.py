#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as maxes
from casadi import *
from casadi.tools import *
import pdb


class backend_graphics:
    """Graphics module to present the results of do-mpc. The module is independent of all other modules but is incorporated in the configuration.
    This allows for a high-level API which is triggered through the configuration.
    At the same time, the module can be used with pickled result files in post-processing for flexible and custom graphics.
    For details about the high-level API, please see the doc string in configuration.setup_graphic().
    Using the low-level API consists of two steps.
    First, the .add_line method is used to define which results are to be plotted. The user passes an existing axes object.
    Note that .add_line does not create a graphic.
    The graphic is obtained with the .plot_results() method, which takes a do-mpc data object as input. Each module (simulator, estimator, optimizer)
    has its own data object.
    Furthermore, the module contains the .plot_predictions() method, which can be used to show the predicted trajectories.
    """
    def __init__(self):
        self.line_list = []
        self.ax_list  = []
        self.color = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def reset_axes(self):
        """Clears the lines on all axes which were passed with .add_line.
        Method is called internally, before each plot.
        """
        for ax_i in self.ax_list:
            ax_i.lines = []
        self.reset_prop_cycle()

    def reset_prop_cycle(self):
        """Resets the property cycle for all axes which were passed with .add_line.
        The matplotlib color cycler is restarted.
        """
        for ax_i in self.ax_list:
            ax_i.set_prop_cycle(None)

    def add_line(self, var_type, var_name, axis, **pltkwargs):
        """add_line is called during setting up the graphics class. This is typically the last step of the configuration of a do-mpc case.
        Each call of .add_line adds a line to the passed axis according to the variable type (_x, _u, _z, _tvp, _p, _aux_expression)
        and its name (as defined in the model).
        Furthermore, all valid matplotlib .plot arguments can be passed as optional keyword arguments, e.g.: 'linewidth', 'color', 'alpha'.

        :param var_type: Variable type to be plotted. Valid arguments are _x, _u, _z, _tvp, _p, _aux_expression.
        :type var_type: string

        :param var_name: Variable name. Must reference the names defined in the model for the given variable type.
        :type var_name: string

        :param axis: Variable name. Must reference the names defined in the model for the given variable type.
        :type axis: matplotlib.axes.Axes object.

        :param pltkwargs: Valid matplotlib pyplot keyword arguments (e.g.: 'linewidth', 'color', 'alpha')
        :type pltkwargs: , optional

        :raises assertion: var_type argument must be a string
        :raises assertion: var_name argument must be a string
        :raises assertion: var_type argument must reference to the valid var_types of do-mpc models.
        :raises assertion: axis argument must be matplotlib axes object.
        """
        assert isinstance(var_type, str), 'var_type argument must be a string. You have: {}'.format(type(var_type))
        assert isinstance(var_name, str), 'var_name argument must be a string. You have: {}'.format(type(var_name))
        assert var_type in ['_x', '_u', '_z', '_tvp', '_p', '_aux_expression'], 'var_type argument must reference to the valid var_types of do-mpc models.'
        assert isinstance(axis, maxes.Axes), 'axis argument must be matplotlib axes object.'

        self.line_list.append(
            {'var_type': var_type, 'var_name': var_name, 'ax': axis, 'reskwargs': pltkwargs, 'predkwargs':pltkwargs.copy()}
        )
        self.ax_list.append(axis)


    def plot_results(self, data, t_ind=None, **pltkwargs):
        """Plots the results stored in the passed data object for the plot configuration.
        Each do-mpc module (simulator, estimator, optimizer) has an individual data object.
        Use the t_ind parameter to plot only until the given time index. This is mostly used in post-processing for animations.
        Optionally pass an arbitrary number of valid pyplot.plot arguments (e.g. 'color', 'linewidth', 'alpha'), which is applied to ALL lines.

        :param data: do-mpc data object. Either from unpickled results or the created modules. The data object is updated at each make_step_... call in the configuration.
        :type data: do-mpc data object

        :param t_ind: Plot results up until this time index.
        :type t_ind: int

        :param pltkwargs: Valid matplotlib pyplot keyword arguments (e.g.: 'linewidth', 'color', 'alpha')
        :type pltkwargs: , optional

        :raises assertion: t_ind argument must be a int
        :raises assertion: t_ind argument must not exceed the length of the results

        :return: All plotted lines on all supplied axes.
        :rtype:  list
        """
        if t_ind is not None:
            assert isinstance(t_ind, int), 'The t_ind param must be of type int. You have: {}'.format(type(t_ind))
            assert t_ind <= data._time.shape[0], 'The t_ind param must not exceed the length of the results. You choose t_ind={}, where only n={} elements are available.'.format(t_ind, data._time.shape[0])
        # Make index "inclusive", if it is passed. This means that for index 1, the elements at 0 AND 1 are plotted.
        if t_ind is not None:
            t_ind+=1

        self.reset_prop_cycle()
        lines = []
        for line_i in self.line_list:
            line_i['reskwargs'].update(pltkwargs)
            time = data._time[:t_ind]
            res_type = getattr(data, line_i['var_type'])
            # The .f() method returns an index of a casadi Struct, given a name.
            var_ind = data.model[line_i['var_type']].f[line_i['var_name']]
            if line_i['var_type'] in ['_u']:
                lines.extend(line_i['ax'].step(time, res_type[:t_ind, var_ind], **line_i['reskwargs']))
            else:
                lines.extend(line_i['ax'].plot(time, res_type[:t_ind, var_ind], **line_i['reskwargs']))

        return lines

    def plot_predictions(self, data, opt_x_num=None, opt_aux_num=None, t_ind=-1, **pltkwargs):
        """Plots the predicted trajectories for the plot configuration.
        The predicted trajectories are part of the optimal solution at each timestep and can be passed either as the optional
        argument (opt_x_num) or they are part of the data structure, if the optimizer was set to store the optimal solution.
        The plot predictions method can only be called with data from the do-mpc optimizer object and raises an error if called with data from other objects.
        Use the t_ind parameter to plot the prediction for the given time instance. This is mostly used in post-processing for animations.
        Optionally pass an arbitrary number of valid pyplot.plot arguments (e.g. 'color', 'linewidth', 'alpha'), which is applied to ALL lines.

        :param data: do-mpc (optimizer) data object. Either from unpickled results or the created modules.
        :type data: do-mpc (optimizer) data object

        :param t_ind: Plot predictions at this time index.
        :type t_ind: int

        :param pltkwargs: Valid matplotlib pyplot keyword arguments (e.g.: 'linewidth', 'color', 'alpha')
        :type pltkwargs: , optional

        :raises assertion: Can only call plot_predictions with data object from do-mpc optimizer
        :raises Exception: Cannot plot predictions if full solution is not stored or supplied when calling the method

        :return: All plotted lines on all supplied axes.
        :rtype:  list
        """
        assert data.dtype == 'optimizer', 'Can only call plot_predictions with data object from do-mpc optimizer.'
        assert isinstance(t_ind, int), 'The t_ind param must be of type int. You have: {}'.format(type(t_ind))

        t_now = data._time[t_ind]
        # These fields only exist, if data type (dtype) os optimizer:
        t_step = data.meta_data['t_step']
        n_horizon = data.meta_data['n_horizon']
        structure_scenario = data.meta_data['structure_scenario']

        # Check if full solution is stored in data, or supplied as optional input. Raise error is neither is the case.
        if opt_x_num is None and data.meta_data['store_full_solution']:
            opt_x_num = data.opt_x(data._opt_x_num[t_ind])
        elif opt_x_num is not None:
            pass
        else:
            raise Exception('Cannot plot predictions if full solution is not stored or supplied when calling the method.')
        if opt_aux_num is None and data.meta_data['store_full_solution']:
            opt_aux_num = data.opt_aux(data._opt_aux_num[t_ind])
        elif opt_aux_num is not None:
            pass
        else:
            raise Exception('Cannot plot predictions if full solution is not stored or supplied when calling the method.')

        # Plot predictions:
        self.reset_prop_cycle()
        lines = []
        for line_i in self.line_list:
            line_i['predkwargs'].update(pltkwargs)
            # Fix color for the robust trajectories according to the current state of the cycler.
            if 'color' not in line_i['predkwargs']:
                color = next(line_i['ax']._get_lines.prop_cycler)['color']
                line_i['predkwargs'].update({'color':color})


            # Choose time array depending on variable type (states with n+1 steps)
            if line_i['var_type'] in ['_x', '_z']:
                t_end = t_now + (n_horizon+1)*t_step
                time = np.linspace(t_now, t_end, n_horizon+1)
            else:
                t_end = t_now + n_horizon*t_step
                time = np.linspace(t_now, t_end, n_horizon)

            # Plot states etc. as continous quantities and inputs as steps.
            if line_i['var_type'] in ['_x', '_z']:
                # pred is a n_horizon x n_branches array.
                pred = vertcat(*opt_x_num[line_i['var_type'],:,lambda v: horzcat(*v),:, -1, line_i['var_name']])
                # sort pred such that each column belongs to one scenario
                pred = pred.full()[range(pred.shape[0]),structure_scenario.T].T
                lines.extend(line_i['ax'].plot(time, pred, **line_i['predkwargs']))
            elif line_i['var_type'] in ['_u']:
                pred = vertcat(*opt_x_num[line_i['var_type'],:,lambda v: horzcat(*v),:,line_i['var_name']])
                pred = pred.full()[range(pred.shape[0]),structure_scenario[:-1,:].T].T
                lines.extend(line_i['ax'].step(time, pred, **line_i['predkwargs']))
            elif line_i['var_type'] in ['_aux_expression']:
                pred = vertcat(*opt_aux_num['_aux',:,lambda v: horzcat(*v),:,line_i['var_name']])
                pred = pred.full()[range(pred.shape[0]),structure_scenario[:-1,:].T].T
                lines.extend(line_i['ax'].plot(time, pred, **line_i['predkwargs']))


        return lines
