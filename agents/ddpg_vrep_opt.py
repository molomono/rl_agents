from collections import OrderedDict
from rl_coach.agents.ddpg_agent import DDPGAgentParameters, DDPGCriticNetworkParameters, DDPGActorNetworkParameters, DDPGAlgorithmParameters
from rl_coach.architectures.layers import Dense, NoisyNetDense, BatchnormActivationDropout
from rl_coach.architectures.middleware_parameters import LSTMMiddlewareParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, EmbedderScheme
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.exploration_policies.ou_process import OUProcessParameters
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import GymVectorEnvironment, mujoco_v2
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.memories.non_episodic.prioritized_experience_replay import PrioritizedExperienceReplayParameters#, EpisodicExperienceReplayParameters
from rl_coach.graph_managers.graph_manager import ScheduleParameters

import rl_environments

'''TODO list for this script:
TODO: Test the reading .csv script to see if the variables come through correctly
TODO: Add more variables to the opt_params list
TODO: Connect the .csv values with the actual variables in the network
TODO: Test if this script can be correctly run from the bayesopt script
'''

################################
# Optimizable parameters list: #
################################
opt_params =   ['actor_layer_1_nodes', 
				'actor_layer_2_nodes', 
				'critic_layer_1_nodes', 
				'critic_layer_2_nodes', 
				'discount_factor', 
				'actor_learning_rate', 
				'critic_learning_rate', 
				'exploration_factor', #Syncrhonized up to this value with the optimizer
				'actor_layer_1_noisy', # these are boolean and turn on and off the noisynetdense layers
				'actor_layer_2_noisy',
				'critic_layer_1_noisy',
				'critic_layer_2_noisy',]

#################################
# Load the newest parameter set #
#################################
import pandas as pd
param_df = pd.read_csv('~/experiments/ddpg_opt/optimization_parameters.csv')
opt_params_dict = param_df.tail(1).to_dict('index')
opt_params_dict = opt_params_dict[list(opt_params_dict.keys())[0]] #removes df-index from dict
# example acces parameter value:
# p_val = opt_params_dict['actor_layer_1_nodes']

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(1000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#############
# Algorithm #
#############
algorithm_params = DDPGAlgorithmParameters()
algorithm_params.discount = 1.0

#########
# Agent #
#########
#Exploration Parameters
exploration_params = OUProcessParameters()
exploration_params.sigma = 5.0
#Network Parameters
#Actor Paramters
actor_params = DDPGActorNetworkParameters()
actor_params.learning_rate = 0.05 # 0.075
#Critic Parameters
critic_params = DDPGCriticNetworkParameters()
critic_params.learning_rate = 0.05
#Agent Parameters
agent_params = DDPGAgentParameters()
agent_params.algorithm = algorithm_params
agent_params.exploration = exploration_params
#agent_params.network["actor"] = actor_params#, ("critic", critic_params)])
#agent_params.memory = PrioritizedExperienceReplayParameters()
agent_params.network_wrappers['actor'].input_embedders_parameters['observation'].scheme = [NoisyNetDense(48)]
agent_params.network_wrappers['actor'].middleware_parameters.scheme = [NoisyNetDense(64)]

agent_params.network_wrappers['critic'].input_embedders_parameters['observation'].scheme = [NoisyNetDense(48)]
#agent_params.network_wrappers['critic'].middleware_parameters = LSTMMiddlewareParameters()
agent_params.network_wrappers['critic'].middleware_parameters.scheme = [NoisyNetDense(48), Dense(48)]
agent_params.network_wrappers['critic'].input_embedders_parameters['action'].scheme = EmbedderScheme.Empty


###############
# Environment #
###############
env_params = GymVectorEnvironment("VrepBalanceBotNoise-v0")
#env_params = GymVectorEnvironment("VrepHopper-v0")
#env_params = GymVectorEnvironment("VrepDoubleCartPoleSwingup-v0")

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(render=False, tensorboard=True))

import os
from rl_coach.base_parameters import TaskParameters, Frameworks

log_path = '~/experiments/ddpg_opt'
if not os.path.exists(log_path):
    os.makedirs(log_path)
    
task_parameters = TaskParameters(framework_type=Frameworks.tensorflow, 
                                evaluate_only=False,
                                experiment_path=log_path)

task_parameters.__dict__['checkpoint_save_secs'] = 300
task_parameters.__dict__['verbosity'] = 'low'

graph_manager.create_graph(task_parameters)

graph_manager.improve()
