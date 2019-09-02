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

import os

'''TODO list for this script:
TODO: Add more variables to the opt_params list
TODO: add more tunable parameters
TODO: Test if this script can be correctly run from the bayesopt script
'''

print("V1.0.1")

def get_layer_nodes_from_categories(category_index):
	return [32, 64, 128, 256, 0][int(category_index)]

log_files_dir = 'ddpg_agent_param_fix'

################################
# Optimizable parameters list: #
################################
opt_params =   ['actor_layer_1_nodes', 
				'actor_layer_2_nodes',
				'actor_layer_3_nodes', 
				'critic_layer_1_nodes', 
				'critic_layer_2_nodes', 
				'critic_layer_3_nodes', 
				'discount_factor', 
				'actor_learning_rate', 
				'critic_learning_rate', 
				'exploration_factor', 
				'polyak',] #Syncrhonized up to this value with the optimizer


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
import pandas as pd
home_path = os.path.expanduser('~')

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(1500000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#For testing the opt software sequencing run very short cycles
#schedule_params.improve_steps = EnvironmentSteps(40)
#schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
#schedule_params.heatup_steps = EnvironmentSteps(10)

#############
# Algorithm #
#############
algorithm_params = DDPGAlgorithmParameters()
algorithm_params.discount = 0.999 #0.934 #opt_params_dict['discount_factor']
algorithm_params.rate_for_copying_weights_to_target = 0.0001 #opt_params_dict['polyak']

#########
# Agent #
#########
#Exploration Parameters
exploration_params = AdditiveNoiseParameters()
exploration_params.noise_schedule = PieceWiseSchedule(
    [#(LinearSchedule(0.75, 0.25, 15000), EnvironmentSteps(15000)),
     (LinearSchedule(0.5, 0.05, 15000), EnvironmentSteps(15000)),
     (ConstantSchedule(0.01), EnvironmentSteps(10000000))]
)
#opt_params_dict['exploration_factor']
#exploration_params = ParameterNoise()
#Network Parameters
#Actor Paramters
actor_params = DDPGActorNetworkParameters()
actor_params.learning_rate = 0.0001#*4*2 #0.002 #opt_params_dict['actor_learning_rate'] # 0.075
actor_params.replace_mse_with_huber_loss = True
actor_params.learning_rate_decay_rate = 0.75
actor_params.learning_rate_decay_steps = 15000

#Critic Parameters
critic_params = DDPGCriticNetworkParameters()
critic_params.learning_rate = 0.0005#*4*2 #0.001 #opt_params_dict['critic_learning_rate']
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
actor_layers = get_layers_list([48,48,24]) # 24 24 24
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

agent_params.network_wrappers['actor'].l2_regularization = 0.000001
agent_params.network_wrappers['actor'].batch_size = 32

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

agent_params.network_wrappers['critic'].l2_regularization = 0.00000001
agent_params.network_wrappers['critic'].batch_size = 32

###############
# Environment #
###############
env_params = GymVectorEnvironment("VrepBalanceBotBalance-v0")

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(render=False))
