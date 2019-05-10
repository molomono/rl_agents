import pandas as pd 
import numpy as np 
import GPyOpt
import GPy
import os
# The logged parameters are writtent o .CSV so i can access these from the python pandas library easily. 
# .jsons are saved to the directory containing the information regarding initialization each agent.
# In the RL scheduler i can set max amount of steps to run, initially this will be set to allow for roughly 1 hour of training.
# The sum of the 'Total Reward' column is a measurement of total performance.
# BayesOpt can be implemented in a completely seperate python file than the AIs themselves, containing domain definitions for the hyperparameter search spaces.
# Use pandas to append the new parameter selection to xxx_parameters_opt.csv file, and create an argument/function that loads the last row from this file for use when running a new trial.

#### Training Sequence
# 
#
agent = 'ddpg'

#Append new agents to these dictionaries:
agent_preset = {'ddpg': 'ddpg_vrep_opt.py'}
agent_opt_dir = {'ddpg': 'ddpg_opt'}


#TODO: Modify the bounds define the bounding box for the hyperparameters
boundaries ={ 'ddpg': 
            [{'name': 'e_d',     'type': 'continuous', 'domain': (0.9,0.9999)},
            {'name': 'e_m',     'type': 'continuous', 'domain': (0.0,0.05)},
            {'name': 'gamma',   'type': 'continuous', 'domain': (0.8,1.0)},
            {'name': 'lr',      'type': 'continuous', 'domain': (0.0,0.05)},
            {'name': 'lr_d',    'type': 'continuous', 'domain': (0.0,0.05)},
            {'name': 'b_m',     'type': 'discrete',   'domain': (16, 32, 64, 128, 256)},
            {'name': 'layer1',  'type': 'discrete',   'domain': (20,24,28,32,36,40)},
            {'name': 'layer2',  'type': 'discrete',   'domain': (20,24,28,32,36,40)},
            {'name': 'layer1_a','type': 'categorical','domain': (0,1,2)}, #['tanh','relu','linear']
            {'name': 'layer2_a','type': 'categorical','domain': (0,1,2)}],
        }

def return_reward():
    ''' Loads the latest ~/expirements/*/worker_xxx.csv from the logged training data and returns the sum of all training rewards for that iteration.
    '''
    pass

def run_ai(param_list):
    ''' Runs the Ai using the parameters defined in the param_list
    '''
    #Load the .csv file
    #Append new parameters to .csv file

    #Start the AI using os.system('python ' + agent_preset[agent]) This python script will wait for the agent to finish. 
    #The AI_opt script will load the parameters from the .csv and perform a training sequence.
    exit_flag = os.system('python ' + agent_preset[agent]) 

    #Provided the exit flag choose an appropriate action (was an error raised or was execution normal)
    if exit_flag == 0:
        # load the .csv file with the previous execution data and return the sum of training rewards
        return -return_reward()
    elif exit_flag != 0: 
        pass
    


def set_dtypes(X):
    ''' Changes the datatypes of the searchspace used by bayes-opt to fit that of tensorflow 
    '''
    [e_decay,e_min,gamma,lr,lr_d,BATCH_max,layer1,layer2,layer1_a,layer2_a] = X

    BATCH_max   =   int(BATCH_max)
    layer1      =   int(layer1)
    layer2      =   int(layer2)
    layer1_a    =   int(layer1_a)
    layer2_a    =   int(layer2_a)
    layer_activations = ['tanh', 'relu', 'linear']

    h_layers = {'layers':[layer1,layer2], 'activation':[layer_activations[layer1_a], layer_activations[layer2_a]], 'initializer':['he_normal', 'he_normal']}

    return [e_decay,e_min,gamma,lr,lr_d,BATCH_max,layer1,layer2,layer1_a,layer2_a], h_layers

################### Example function derived from my other repository
################### Is being altered to support the new AI Agents
if __name__=="__main__":

    #Configure optimizer and set the number of optimization steps
    max_iter = 25
    ai_optimizer = GPyOpt.methods.BayesianOptimization(run_ai, domain=boundaries[agent],
                                                        initial_design_numdata = 8,   # Number of initial datapoints before optimizing
                                                        Initial_design_type = 'latin',
                                                        model_type= 'GP_MCMC',
                                                        acquisition_type='EI_MCMC',
                                                        normalize_Y = True) #http://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/devel/manual/GPyOpt_mixed_domain.ipynb
    #Run optimizer
    ai_optimizer.run_optimization(max_iter)

    # Evaluate using ai_optimizer.plot_convergence()
    ai_optimizer.plot_convergence()

    # All the hyperparameters come out of the optimization as float64 set_dtypes corrects the datatypes
    [x_optimum, h_layers] = set_dtypes(ai_optimizer.x_opt)

    #The estimated optimum is printed and saved to a file
    print("Best performance: ", x_optimum)

    #Save best performance to a file
    file_name = '~/expirements/'+ agent_opt_dir[agent] +'/Optimzed_performance_variables.csv'
    df = pd.DataFrame([x_optimum], columns=bounds[:]['name'])
    df.to_csv(file_name, index=False)