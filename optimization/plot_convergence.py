# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import os
import numpy as np
from pylab import grid
import matplotlib.pyplot as plt
from pylab import savefig
import pylab
import pandas as pd
import GPyOpt
import GPy
import seaborn as sns

from bayesopt_v2 import return_reward, load_params_of_all_trials, boundaries, agent


def plot_convergence(Xdata,best_Y, filename = None):
    '''
    Plots to evaluate the convergence of standard Bayesian optimization algorithms
    '''
    n = Xdata.shape[0]
    aux = (Xdata[1:n,:]-Xdata[0:n-1,:])**2
    distances = np.sqrt(aux.sum(axis=1))

    ## Distances between consecutive x's
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(n-1)), distances, '-ro')
    plt.xlabel('Iteration')
    plt.ylabel('d(x[n], x[n-1])')
    plt.title('Distance between consecutive x\'s')
    grid(True)

    # Estimated m(x) at the proposed sampling points
    plt.subplot(1, 2, 2)
    plt.plot(list(range(n)),best_Y,'-o')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration')
    plt.ylabel('Best y')
    grid(True)

    if filename!=None:
        savefig(filename)
    else:
        plt.show()
        
if __name__ == '__main__':
    y_vals = return_reward(True)
    y_vals_norm = return_reward(True,True)
    x_params = load_params_of_all_trials(return_dataframe=True)
    x_params['Training Rewards'] = y_vals
    x_params['Normalized Rewards'] = return_reward(True,True)
    print(x_params)
    
    plot_convergence(x_params.values, y_vals_norm)

    keys = x_params.keys()
    print(keys)
    
    g = sns.PairGrid(data = x_params[list(keys)[:]], diag_sharey=False)
    g.map_upper(sns.kdeplot)
    g.map_lower(sns.scatterplot)
    g.map_diag(sns.kdeplot, lw=3)
    plt.show()
    
    #sns.pairplot(data = x_params[list(keys)[-7:]], diag_kind = 'kde')
    #sns.pairplot(data = x_params[list(keys)[-7:]], x_vars = list(keys)[-7:-2], y_vars = list(keys)[-1], diag_kind='kde')
    #plt.show()
    
    #sns.swarmplot(x = list(keys)[-7:], y = list(keys[-1]), hue = list(keys)[-1], data = x_params)
    #plt.show()
    
    
