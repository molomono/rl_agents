##########################################################
## Author: Brian Armstrong								##
## Company: DEMCON 										##
## Function: Graduation Assignment, RL-AI in robotics	##
## Date: 05-2019										##
##########################################################

import pandas as pd 
import sys
import numpy as np 
import GPyOpt
import GPy
import os
import glob
import pickle
import time

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
'''
home_path = os.path.expanduser('~')

agent = 'ddpg'

#Append new agents to these dictionaries:
agent_preset = {'ddpg': 'ddpg_vrep_opt.py'}
agent_opt_dir = {'ddpg': 'ddpg_opt_gymtest_4'}


#TODO: Modify the bounds define the bounding box for the hyperparameters
boundaries ={'example':
                [{'name': 'e_d',    'type': 'continuous', 'domain': (0.9,0.9999)},
                {'name': 'e_m',     'type': 'continuous', 'domain': (0.0,0.05)},
                {'name': 'gamma',   'type': 'continuous', 'domain': (0.8,1.0)},
                {'name': 'lr',      'type': 'continuous', 'domain': (0.0,0.05)},
                {'name': 'lr_d',    'type': 'continuous', 'domain': (0.0,0.05)},
                {'name': 'b_m',     'type': 'discrete',   'domain': (16, 32, 64, 128, 256)},
                {'name': 'layer1',  'type': 'discrete',   'domain': (20,24,28,32,36,40)},
                {'name': 'layer2',  'type': 'discrete',   'domain': (20,24,28,32,36,40)},
                {'name': 'layer1_a','type': 'categorical','domain': (0,1,2)}, #['tanh','relu','linear']
                {'name': 'layer2_a','type': 'categorical','domain': (0,1,2)}], 
            'ddpg':
                [{'name': 'actor_layer_1_nodes',    'type': 'discrete',   'domain': (32, 64, 128, 256)}, 
                {'name': 'actor_layer_2_nodes',     'type': 'discrete',   'domain': (0, 32, 64, 128, 256)}, 
                {'name': 'actor_layer_3_nodes',     'type': 'discrete',   'domain': (0, 32, 64, 128, 256)}, 
                {'name': 'critic_layer_1_nodes',    'type': 'discrete',   'domain': (32, 64, 128, 256)}, 
                {'name': 'critic_layer_2_nodes',    'type': 'discrete',   'domain': (32, 64, 128, 256)}, 
                {'name': 'critic_layer_3_nodes',    'type': 'discrete',   'domain': (0, 32, 64, 128, 256)}, 
                {'name': 'discount_factor',         'type': 'continuous', 'domain': (0.8,1.0)}, 
                {'name': 'actor_learning_rate',     'type': 'continuous', 'domain': (0.0001, 0.1)}, 
                {'name': 'critic_learning_rate',    'type': 'continuous', 'domain': (0.0001, 0.1)}, 
                {'name': 'exploration_factor',      'type': 'continuous', 'domain': (0.1,3.0)},
                {'name': 'polyak',      			'type': 'continuous', 'domain': (0.001,0.5)}],
            }

#Retrieve the names of all the parameter variables being tuned:
param_names = []
for i in range(len(boundaries[agent])):
    param_names += [boundaries[agent][i]['name']]


def return_reward(return_all_trials = False, normalize=False):
    ''' Loads the latest ~/experiments/agent_directory/worker_xxx.csv from the logged training data and returns the sum of all training rewards for that iteration.
    '''
    # Load the names of all *.csv files in directory
    file_list = os.listdir(home_path+'/experiments/'+agent_opt_dir[agent])
    # Filter list only log files remain
    file_list = [k for k in file_list if 'main_level' in k]
    # Append the directory location to the file_names
    for i in range(len(file_list)):
        file_list[i] = home_path+'/experiments/'+agent_opt_dir[agent]+'/'+file_list[i]
    #Sort the files based on the time of modification
    file_list.sort(key=os.path.getmtime)
    if return_all_trials:
        #TODO: TEST THIS BRANCH OF THE RETURN REWARD FUNCTION
        Y = []
        for file_location in file_list:				
            newest_training_data_dataframe = pd.read_csv(file_location)
            # Sum-up and return all values in the 'Training Reward' column
            total_reward = newest_training_data_dataframe['Training Reward'].sum()
            # Normalize the Training reward by dividing it by the total number of training iterations
            if normalize:
                n_training_iterations = newest_training_data_dataframe['Training Iter']
                print('Number of training iters: ', n_training_iterations.values[-1])
                total_reward = total_reward/n_training_iterations.values[-1]
            Y += [[total_reward]]
        return np.asarray(Y)
    else:
        # Load most recent edit
        newest_training_data_dataframe = pd.read_csv(file_list[-1])
        # Sum-up and return all values in the 'Training Reward' column
        total_reward = newest_training_data_dataframe['Training Reward'].sum()
        # Normalize the Training reward by dividing it by the total number of training iterations
        if normalize:
            n_training_iterations = newest_training_data_dataframe['Training Iter']
            print('Number of training iters: ', n_training_iterations.values[-1])
            total_reward = total_reward/n_training_iterations.values[-1]
        return total_reward


def run_ai(param_list):
    ''' Runs the Ai created using the parameters defined by the parameter list generated by the bayesian optimizer
    '''
    # Load/create the .csv file as a dataframe
    # Append new parameters to dataframe
    print(param_names)
    print(param_list)
    
    if glob.glob(home_path + '/experiments/'+ agent_opt_dir[agent] +'/optimization_parameters.csv'):
        parameters_dataframe = pd.read_csv(home_path + '/experiments/'+ agent_opt_dir[agent] +'/optimization_parameters.csv')
        parameters_dataframe = parameters_dataframe.append( pd.DataFrame(param_list, columns=param_names), ignore_index=True)
    else:
        if not os.path.exists(home_path + '/experiments/'+ agent_opt_dir[agent]):
            os.mkdir(home_path + '/experiments/'+ agent_opt_dir[agent])
        parameters_dataframe = pd.DataFrame(param_list, columns=param_names)
        
    # Save the dataframe to .csv
    parameters_dataframe.to_csv(home_path + '/experiments/'+ agent_opt_dir[agent] +'/optimization_parameters.csv', index=False)

    # Start the AI using os.system('python ' + agent_preset[agent]) This python script will wait for the agent to finish. 
    # The AI_opt script will load the parameters from the .csv and perform a training sequence.
    exit_flag = os.system('python ../agents/' + agent_preset[agent]) 

    # Provided the exit flag choose an appropriate action (was an error raised or was execution normal)
    if exit_flag == 0:
       # load the .csv file with the previous execution data and return the sum of training rewards
       return -return_reward(normalize = True)
    elif exit_flag != 0: 
        print('An error occured while training the AI Agent')
        remove_failed_optimization_iteration()
        quit()
	
def load_params_of_all_trials(return_dataframe = False):
	''' Load all the hyperparameters from the params.csv file and return them as a numpy array
	'''
	parameters_dataframe = pd.read_csv(home_path + '/experiments/'+ agent_opt_dir[agent] +'/optimization_parameters.csv')
	if return_dataframe:
		return parameters_dataframe
	else:
		return parameters_dataframe.values

def remove_failed_optimization_iteration(remove_param = True, remove_log_arg = False):
    ''' When an iteration fails it leaves behind a log file that isn't complete, it can still be used to draw the sum of total rewards but 
    doing this will skew the hyperparameter predictions in a bad way.
	
    This function is used to delete the incomplete .csv file and alter the parameters.csv file to remove the last row.
    '''
    num_iterations = 0
    if remove_param:
        #Load parameters.csv 
        parameters_dataframe = pd.read_csv(home_path + '/experiments/'+ agent_opt_dir[agent] +'/optimization_parameters.csv')
        pd.DataFrame(parameters_dataframe.values[:-1,:], columns=param_names).to_csv(home_path + '/experiments/'+ agent_opt_dir[agent] +'/optimization_parameters.csv', index=False)
        #Number of iterations remaining after deleting the previous entry
        num_iterations = len(parameters_dataframe)-1
  
    # Load the names of all *.csv files in directory
    file_list = os.listdir(home_path+'/experiments/'+agent_opt_dir[agent])
    # Filter list only log files remain
    file_list = [k for k in file_list if 'main_level' in k]
    num_log_files = len(file_list)
    
    # If the number of log files is more than the parameter entries remove the newest log file
    remove_log = num_log_files > num_iterations
    if remove_log or remove_log_arg:     
        # Append the directory location to the file_names
        for i in range(len(file_list)):
            file_list[i] = home_path+'/experiments/'+agent_opt_dir[agent]+'/'+file_list[i]
        #Sort the files based on the time of modification
        file_list.sort(key=os.path.getmtime)
        #Remove the last log file
        os.remove(file_list[-1])

if __name__=="__main__":
    ''' Main function instantiates a gaussian process optimizer from the GPyOpt package and performs Bayesian Optimization
    within the search domain defined in the boundaries dict.
    '''
    #Define the number of optimization iterations to run.
    initial_datapoints = 5
    max_iter = 20
    X = None
    Y = None
	
    #If there are alerady .csv files in the project folder load the dataset X, Y and change the initial deisng numtypes to 0
    if glob.glob(home_path + '/experiments/'+ agent_opt_dir[agent] +'/optimization_parameters.csv'):
        
        #TODO: Use the argument --remove-last to remove the last logged iteration
        if any('--remove-last' in s for s in sys.argv):
            #Function deals with the problem caused by having an incomplete log file due to crashing of a training iteration.
            remove_param = any('X' in s for s in sys.argv)
            remove_log = any('Y' in s for s in sys.argv)

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
	
    #Configure optimizer and set the number of optimization steps
    ai_optimizer = GPyOpt.methods.BayesianOptimization(run_ai, domain=boundaries[agent],
                                                        initial_design_numdata = initial_datapoints,   # Number of initial datapoints before optimizing
                                                        X = X,
                                                        Y = Y,
                                                        Initial_design_type = 'latin',
                                                        model_type= 'GP_MCMC',
                                                        acquisition_type='EI_MCMC',
                                                        verbosity = True,
                                                        verbosity_model = True,
                                                        normalize_Y = True) #http://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/devel/manual/GPyOpt_mixed_domain.ipynb
    print(ai_optimizer.model.model)
    
    #Run optimizer
    ai_optimizer.run_optimization(max_iter)
        
    # Evaluate using ai_optimizer.plot_convergence()
    ai_optimizer.plot_convergence()

    # All the hyperparameters come out of the optimization as float64 set_dtypes corrects the datatypes
    x_optimum = ai_optimizer.x_opt

    #The estimated optimum after training is printed
    print("Parameter names : ", param_names)
    print("Best params     : ", x_optimum)
