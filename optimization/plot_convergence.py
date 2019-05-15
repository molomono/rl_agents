# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import os
import numpy as np
from pylab import grid
import matplotlib.pyplot as plt
from pylab import savefig
import pylab
import pandas as pd

from bayesopt import return_reward, load_params_of_all_trials


home_path = os.path.expanduser('~')


agent = 'ddpg'

#Append new agents to these dictionaries:
agent_preset = {'ddpg': 'ddpg_vrep_opt.py'}
agent_opt_dir = {'ddpg': 'ddpg_opt_2'}


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
    x_params = load_params_of_all_trials()
    print(pd.DataFrame(x_params))
	
    plot_convergence(x_params, return_reward(True))
