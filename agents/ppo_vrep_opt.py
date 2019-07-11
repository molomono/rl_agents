from collections import OrderedDict
from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters, ClippedPPOAlgorithmParameters
from rl_coach.architectures.layers import Dense, NoisyNetDense
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, DistributedCoachSynchronizationType, EmbedderScheme
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import GymVectorEnvironment, mujoco_v2
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.observation.observation_normalization_filter import ObservationNormalizationFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule

import rl_environments

import os

'''TODO list for this script:
TODO: Add more variables to the opt_params list
TODO: add more tunable parameters
TODO: Test if this script can be correctly run from the bayesopt script
'''
import sys
sys.path.insert(0, os.path.abspath('../optimization'))
from constants_and_spaces import *
agent = 'ppo'
log_files_dir = agent_opt_dir[agent]

def get_layer_nodes_from_categories(category_index):
	return [32, 64, 128, 256, 0][int(category_index)]

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
				'gae_lambda', 
				'beta_entropy',
				'clip_likelihood_ratio',] #Syncrhonized up to this value with the optimizer

#################################
# Create layers iteratively		#
#################################
def get_layers_list(layer_nodes_list):
    layers_list = []
    for nodes in layer_nodes_list:
        if nodes is 0:
            break	
        else:
            layers_list += [NoisyNetDense(nodes)]
    return layers_list

#################################
# Load the newest parameter set #
#################################
import pandas as pd
home_path = os.path.expanduser('~')
param_df = pd.read_csv(home_path+'/experiments/'+ log_files_dir +'/optimization_parameters.csv')
opt_params_dict = param_df.tail(1).to_dict('index')
opt_params_dict = opt_params_dict[list(opt_params_dict.keys())[0]] #removes df-index from dict
# example acces parameter value:
# p_val = opt_params_dict['actor_layer_1_nodes']

#Get layer lists:
actor_layers_nodes = [get_layer_nodes_from_categories(opt_params_dict['actor_layer_1_nodes']),
                      get_layer_nodes_from_categories(opt_params_dict['actor_layer_2_nodes']), 
                      get_layer_nodes_from_categories(opt_params_dict['actor_layer_3_nodes'])]
critic_layers_nodes =[get_layer_nodes_from_categories(opt_params_dict['critic_layer_1_nodes']), 
                      get_layer_nodes_from_categories(opt_params_dict['critic_layer_2_nodes']), 
                      get_layer_nodes_from_categories(opt_params_dict['critic_layer_3_nodes'])]

actor_layers = get_layers_list(actor_layers_nodes)
critic_layers = get_layers_list(critic_layers_nodes)


####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(18000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(1000)

####################
# Graph Scheduling #
####################
#schedule_params = ScheduleParameters()
#schedule_params.improve_steps = TrainingSteps(10000000000)
#schedule_params.steps_between_evaluation_periods = EnvironmentSteps(2000)
#schedule_params.evaluation_steps = EnvironmentEpisodes(1)
#schedule_params.heatup_steps = EnvironmentSteps(0)

#############
# Algorithm #
#############
algorithm_params = ClippedPPOAlgorithmParameters()
algorithm_params.gae_lambda = opt_params_dict['gae_lambda']
algorithm_params.discount = opt_params_dict['discount_factor']
algorithm_params.beta_entropy = opt_params_dict['beta_entropy']
algorithm_params.clip_likelihood_ratio_using_epsilon = opt_params_dict['clip_likelihood_ratio']

algorithm_params.num_consecutive_playing_steps = EnvironmentSteps(250)

#########
# Agent #
#########
agent_params = ClippedPPOAgentParameters()


agent_params.network_wrappers['main'].learning_rate = 0.0003
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = EmbedderScheme.Empty
agent_params.network_wrappers['main'].middleware_parameters.scheme = actor_layers
agent_params.network_wrappers['main'].middleware_parameters.activation_function = 'tanh'
agent_params.network_wrappers['main'].batch_size = 64
agent_params.network_wrappers['main'].optimizer_epsilon = 1e-5
agent_params.network_wrappers['main'].adam_optimizer_beta2 = 0.999

agent_params.algorithm.clip_likelihood_ratio_using_epsilon = 0.2
agent_params.algorithm.clipping_decay_schedule = LinearSchedule(1.0, 0, 1000000)
agent_params.algorithm.beta_entropy = 0
agent_params.algorithm.gae_lambda = 0.95
agent_params.algorithm.discount = 0.99
agent_params.algorithm.optimization_epochs = 10
agent_params.algorithm.estimate_state_value_using_gae = True
agent_params.algorithm = algorithm_params
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(250)


# Distributed Coach synchronization type.
agent_params.algorithm.distributed_coach_synchronization_type = DistributedCoachSynchronizationType.SYNC

agent_params.pre_network_filter = InputFilter()
agent_params.pre_network_filter.add_observation_filter('observation', 'normalize_observation',
                                                        ObservationNormalizationFilter(name='normalize_observation'))

###############
# Environment #
###############
env_params = GymVectorEnvironment("VrepBalanceBotNoise-v0")
#env_params = GymVectorEnvironment("RoboschoolInvertedDoublePendulum-v1")
#env_params = GymVectorEnvironment("VrepHopper-v0")
#env_params = GymVectorEnvironment("VrepDoubleCartPoleSwingup-v0")

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(render=False))

import os
from rl_coach.base_parameters import TaskParameters, Frameworks

log_path = home_path+'/experiments/'+log_files_dir
if not os.path.exists(log_path):
    os.makedirs(log_path)
    
task_parameters = TaskParameters(framework_type=Frameworks.tensorflow, 
                                evaluate_only=False,
                                experiment_path=log_path)

task_parameters.__dict__['checkpoint_save_secs'] = 300
task_parameters.__dict__['verbosity'] = 'low'

graph_manager.create_graph(task_parameters)

graph_manager.improve()
graph_manager.close()
