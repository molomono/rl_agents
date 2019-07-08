##########################################################
## Author: Brian Armstrong								##
## Company: DEMCON 										##
## Function: Graduation Assignment, RL-AI in robotics	##
## Date: 05-2019                                        ##
## Version 2, contains the addition of batch sampling   ##
##########################################################

import pandas as pd 
import sys
import numpy as np 

import os
import glob
import time

import GPyOpt
import GPy
from GPyOpt.util.mcmc_sampler import AffineInvariantEnsembleSampler
import argparse

''' List of WIP parts of this script
TODO: move all the definitions such as boundaries and paths to a seperate .py file
TODO: move all the functions to a seperate .py file

TODO: In my last test the new inferred datapoints stopped varying for some reason:
	-- After taking a look at the training data it seems that there is too little improvement and too 
	much variance in the environment for the gaussian process to establish a good prediction for better hyperparams. 
	--> Now testing in gym environments --> After validating this script works  fix the envrionment.
IMPORTANT:
	--> It looks like after it takes all the init-datapoints it didn't start the sequence of modeling and infering, it just provided the same prediction step after step 
	I NEED TO FIX THIS^^^
	Weirdly enough it hangs up on predicting the same parameters that were used for the initial trial.
	NOTES:
	After some exra research it seems that EI might be over-confident, this combined with the noisy evaluations is causing poor convergence.
	Possible solutions: 
		1. Rewrite the optimizer to the Modular method
		2. Play with different acquisition and evaluation methods --> Entropy acquistion and Batch sampling methods.
		
	PS: Batch sampling seems computationally 'expensive' however especially the Local Penalty method seems promising to,
	reduce noise and improve convergence.	
	
	See: https://github.com/SheffieldML/GPyOpt/blob/master/manual/GPyOpt_constrained_optimization.ipynb
		 https://github.com/SheffieldML/GPyOpt/blob/master/manual/GPyOpt_entropy_search.ipynb
		
	#i should also implement a Toy equation just to make sure GPyOpt is doing what it should be doing
'''

import sys
 
parser = argparse.ArgumentParser(description='Optimization algorithm arguments.')
parser.add_argument('--algorithm', metavar='str', type=str, help='Choice of machine learning algorithm.')
parser.add_argument('--remove-last', type=str, help='Use this flag to remove the last entery in the hyperparameter log. Use values X (hyperparameters), Y (training log-file) or XY for both')
    
args = parser.parse_args()
agent = args.algorithm

from constants_and_spaces import *
from interfacing_functions import *

if __name__=="__main__":
    ''' Main function instantiates a gaussian process optimizer from the GPyOpt package and performs Bayesian Optimization
    within the search domain defined in the boundaries dict.
    '''
    #Define the number of optimization iterations to run.
    initial_datapoints = 5
    max_iter = 25
    X = None
    Y = None
	
    #If there are alerady .csv files in the project folder load the dataset X, Y and change the initial deisng numtypes to 0
    if glob.glob(home_path + '/experiments/'+ agent_opt_dir[agent] +'/optimization_parameters.csv'):
        
        #TODO: Use the argument --remove-last to remove the last logged iteration
        #Function deals with the problem caused by having an incomplete log file due to crashing of a training iteration.
        remove_param = print(any('X' in str_val for str_val in args.remove_last))
        remove_log = print(any('Y' in str_val for str_val in args.remove_last))
    
        if remove_param or remove_log:
            remove_failed_optimization_iteration(remove_param, remove_log)
        else:
            #print('WARNING---------------------WARNING')
            #print('Important notice, the prior trials are loaded but if a trial was executed during training')
            #print('run this script once with .py script with argument --remove-last param and/or log')
            #time.sleep(10)
            pass
           
        Y = -return_reward(return_all_trials = True, normalize=True)
        X = load_params_of_all_trials()
        
        print('Dimensions X: {},  Y: {}'.format(X.shape,Y.shape))
        if X.shape[0] is 0 and Y.shape[0] is 0:
            X = None
            Y = None
        elif X.shape[0] < initial_datapoints:
            #Subtract the number of existing datapoints if a prior dataset already exists, otherwise set it to zero. 
            initial_datapoints = initial_datapoints - X.shape[0]
        else:
            initial_datapoints = 0
	
    print("Number of initial datapoints before Bayesian Optimization: {}".format(initial_datapoints))
	
	#################################### Initializing the searchspaces and data
	# Construct the optimization space using the boundaries
    param_space = GPyOpt.Design_space(space = boundaries[agent], constraints = None)
	
	# Initial datapoints
    initial_design = GPyOpt.experiment_design.initial_design('latin', param_space, initial_datapoints)
	# Append the new parameters to the X dataset
    print(initial_design)
    print(X)
    if X is None:
        X = initial_design
    else:
        X = np.concatenate([X, initial_design], axis=0)
	
	##################################### Instantiating the BayesOpt objects/functions
	# Choose the objective function:
    objective = GPyOpt.core.task.SingleObjective(run_ai)
	
	# Choose a kernel, either RBF or Matern32/Matern52 usally works well enough
    kern = GPy.kern.RBF(input_dim = X.shape[1], variance = 0.5, lengthscale = 0.1)
	
	# Choose the model type
    model = GPyOpt.models.GPModel(kern, noise_var=1e-1, exact_feval=False, optimize_restarts=10, verbose=False)
	
	# Choose the acquisition optimizer
    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(param_space)
	
	# Entropy search:
    #ei = GPyOpt.acquisitions.AcquisitionEI(model, param_space, optimizer=acquisition_optimizer)
    #proposal_function = lambda x_: np.clip(np.log(ei._compute_acq(x_)), 0., np.PINF)
	
    #sampler = AffineInvariantEnsembleSampler(param_space)
    #acquisition = GPyOpt.acquisitions.AcquisitionEntropySearch(model, param_space, sampler, optimizer = acquisition_optimizer, burn_in_steps=10, num_samples=100, proposal_function = proposal_function)
	
	# Choose the acquisition function
    acquisition = GPyOpt.acquisitions.AcquisitionLCB(model, param_space, optimizer = acquisition_optimizer)
    
	# Adding local penalization to the aquisition function
    #acquisition = GPyOpt.acquisitions.AcquisitionLP(model, param_space, acquisition=acquisition_method, optimizer = acquisition_optimizer)

	# Choose a evaluation method
    evaluator = GPyOpt.core.evaluators.ThompsonBatch(acquisition, batch_size = 5)
    #evaluator = GPyOpt.core.evaluators.LocalPenalization(acquisition, batch_size = 3)
    
	
	##################################### Constructing the actual optimizer object
	# BO Object
    ai_optimizer = GPyOpt.methods.ModularBayesianOptimization(model, param_space, objective, acquisition, evaluator, X_init = X, Y_init = Y)
	
	# Print the AI model after initialization
    print(ai_optimizer.model.model)
    
    ##################################### Running the Optimization
    #Run optimizer
    ai_optimizer.run_optimization(max_iter, verbosity=True)
        
    # Evaluate using ai_optimizer.plot_convergence()
    ai_optimizer.plot_convergence()

    # All the hyperparameters come out of the optimization as float64 set_dtypes corrects the datatypes
    x_optimum = ai_optimizer.x_opt

    #The estimated optimum after training is printed
    print("Parameter names : ", param_names)
    print("Best params     : ", x_optimum)
