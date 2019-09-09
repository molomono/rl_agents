from collections import OrderedDict
from rl_coach.agents.ddpg_agent import DDPGAgentParameters, DDPGCriticNetworkParameters, DDPGActorNetworkParameters, DDPGAlgorithmParameters
from rl_coach.architectures.layers import Dense, NoisyNetDense, BatchnormActivationDropout
from rl_coach.architectures.middleware_parameters import LSTMMiddlewareParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, EmbedderScheme
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.exploration_policies.ou_process import OUProcessParameters
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import GymVectorEnvironment, mujoco_v2
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.memories.non_episodic.prioritized_experience_replay import PrioritizedExperienceReplayParameters#, EpisodicExperienceReplayParameters
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule, PieceWiseSchedule, ConstantSchedule

import rl_environments
import pandas as pd
import os
import numpy as np

#Define dict containing the information required to construct a schedual
noise_sched = {'lin_schedule': [0.5, 0.05, 15000], 'const_schedule': [0.01, 10000000]}

default_hyper_params = { 'params': ['a_lr', 'c_lr', 'a_l2', 'c_l2', 'a_bs', 'c_bs', 'noise', 'n_heatup', 'discount', 'polyak'], 
						 'values':	[0.0001, 0.0005, 0.000001, 0.00000001, 32, 32, 0.5, 1000, 0.999, 0.0001] }
default_h_params = pd.DataFrame(default_hyper_params).set_index('params')
del default_hyper_params, noise_sched

#################################
# Create layers iteratively		#
#################################
def get_layers_list(layer_nodes_list):
    layers_list = []
    for nodes in layer_nodes_list:
        if nodes is 0:
            break	
        else:
            layers_list += [Dense(nodes)]
    return layers_list

#################################
# Load the newest parameter set #
#################################
home_path = os.path.expanduser('~')

def remove_nan_params():
	#This function replaces all custom 'nan' valued hyperparameters with the default settings
	for parameter in h_params.index:
		if np.isnan(h_params.loc[parameter].values[0]):
			h_params.loc[parameter].values[0] = default_h_params.loc[parameter].values[0]
			
		
print('Use default hyper-parameters? [y/n]')
if input() is 'n':
	print('Using custom parameters, these can be configured using the set_hyperparameters.py script.')
	try:
		h_params = pd.read_csv(home_path+'/hyper_parameter_files/'+'hyper_params.csv', index_col="params")
	except:
		h_params = default_h_params
	remove_nan_params()
else:
	h_params = default_h_params
	
def load_param(param_name):
	return h_params.loc[param_name].values[0]
	
	

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(1500000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(int(load_param('n_heatup')))

#For testing the opt software sequencing run very short cycles
#schedule_params.improve_steps = EnvironmentSteps(40)
#schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
#schedule_params.heatup_steps = EnvironmentSteps(10)

#############
# Algorithm #
#############
algorithm_params = DDPGAlgorithmParameters()
algorithm_params.discount = load_param('discount') #0.934 #opt_params_dict['discount_factor']
algorithm_params.rate_for_copying_weights_to_target = load_param('polyak') #opt_params_dict['polyak']

#########
# Agent #
#########
#Exploration Parameters
exploration_params = AdditiveNoiseParameters()

#exploration_params.noise_schedule = PieceWiseSchedule(
#   [#(LinearSchedule(0.75, 0.25, 15000), EnvironmentSteps(15000)),
#     (LinearSchedule(0.5, 0.05, 15000), EnvironmentSteps(15000)),
#     (ConstantSchedule(0.01), EnvironmentSteps(10000000))] )
exploration_params.noise_schedule = PieceWiseSchedule([(ConstantSchedule(load_param('noise')), EnvironmentSteps(10000000))])
     
#opt_params_dict['exploration_factor']
#exploration_params = ParameterNoise()
#Network Parameters
#Actor Paramters
actor_params = DDPGActorNetworkParameters()
actor_params.learning_rate = load_param('a_lr') #*4*2 #0.002 #opt_params_dict['actor_learning_rate'] # 0.075
actor_params.replace_mse_with_huber_loss = True
#actor_params.learning_rate_decay_rate = 0.75
#actor_params.learning_rate_decay_steps = 15000

#Critic Parameters
critic_params = DDPGCriticNetworkParameters()
critic_params.learning_rate = load_param('c_lr') #*4*2 #0.001 #opt_params_dict['critic_learning_rate']
critic_params.replace_mse_with_huber_loss = True
#critic_params.learning_rate_decay_rate = 0.9985
#critic_params.learning_rate_decay_steps = 100
#Agent Parameters
agent_params = DDPGAgentParameters()
agent_params.algorithm = algorithm_params
agent_params.exploration = exploration_params

#Memory
agent_params.memory = PrioritizedExperienceReplayParameters()
agent_params.memory.beta = LinearSchedule(0.4, 1, 125000)

#Defining the layers
actor_layers = get_layers_list([48,48,24]) 
critic_layers = get_layers_list([48,48,32])

#actor_layers = [NoisyNetDense(48),NoisyNetDense(24),NoisyNetDense(24)]
#Actor 
agent_params.network_wrappers['actor'] = actor_params
agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].scheme = actor_layers[:2]
agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].dropout_rate = 0.5
agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].activation_function = 'selu' # relu

#agent_params.network_wrappers['actor'].middleware_parameters = LSTMMiddlewareParameters(number_of_lstm_cells=64, dropout_rate = 0.5)
agent_params.network_wrappers['actor'].middleware_parameters.scheme = [actor_layers[2]]
agent_params.network_wrappers['actor'].middleware_parameters.dropout_rate = 0.5
agent_params.network_wrappers['actor'].middleware_parameters.activation_function = 'tanh' # tanh

agent_params.network_wrappers['actor'].l2_regularization = load_param('a_l2')
agent_params.network_wrappers['actor'].batch_size = int(load_param('a_bs'))

#Critic
agent_params.network_wrappers['critic'] = critic_params
agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].scheme = critic_layers[:2]
agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].dropout_rate = 0.5
agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].activation_function = 'selu' # relu

agent_params.network_wrappers['critic'].input_embedders_parameters['action'].scheme = critic_layers[:1]
agent_params.network_wrappers['critic'].input_embedders_parameters['action'].dropout_rate = 0.5
agent_params.network_wrappers['critic'].input_embedders_parameters['action'].activation_function = 'selu' # relu

agent_params.network_wrappers['critic'].middleware_parameters = LSTMMiddlewareParameters(number_of_lstm_cells=64, dropout_rate = 0.5)
agent_params.network_wrappers['critic'].middleware_parameters.scheme = [critic_layers[2]]
#agent_params.network_wrappers['critic'].middleware_parameters.scheme = critic_layers
#agent_params.network_wrappers['critic'].middleware_parameters.dropout_rate = 0.5
agent_params.network_wrappers['critic'].middleware_parameters.activation_function = 'selu' # relu

agent_params.network_wrappers['critic'].l2_regularization = load_param('c_l2')
agent_params.network_wrappers['critic'].batch_size = int(load_param('c_bs'))

###############
# Environment #
###############
env_params = GymVectorEnvironment("VrepBalanceBotBalance-v0")

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(render=False))
