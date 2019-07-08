import os 

home_path = os.path.expanduser('~')

#Append new agents to these dictionaries:
agent_preset = {'ddpg': 'ddpg_vrep_opt.py',
				'ppo': 'ppo_vrep_opt.py'}

#Decleration of the directory name used for each respective algorithm optimization cycle.				
agent_opt_dir = {'ddpg': 'ddpg_vrep_0',
				 'ppo': 'ppo_vrep_opt_0'}

#Boundaries is a dict of each algorithms hyperparameter spaces.
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
                [{'name': 'actor_layer_1_nodes',    'type': 'categorical',   'domain': (0, 1, 2, 3)}, 
                {'name': 'actor_layer_2_nodes',     'type': 'categorical',   'domain': (0, 1, 2, 3, 4)}, 
                {'name': 'actor_layer_3_nodes',     'type': 'categorical',   'domain': (0, 1, 2, 3, 4)}, 
                {'name': 'critic_layer_1_nodes',    'type': 'categorical',   'domain': (0, 1, 2, 3)}, 
                {'name': 'critic_layer_2_nodes',    'type': 'categorical',   'domain': (0, 1, 2, 3)}, 
                {'name': 'critic_layer_3_nodes',    'type': 'categorical',   'domain': (0, 1, 2, 3, 4)}, 
                {'name': 'discount_factor',         'type': 'continuous', 'domain': (0.9,1.0)}, 
                {'name': 'actor_learning_rate',     'type': 'continuous', 'domain': (0.0001, 0.5)}, 
                {'name': 'critic_learning_rate',    'type': 'continuous', 'domain': (0.0001, 0.5)}, 
                {'name': 'exploration_factor',      'type': 'continuous', 'domain': (0.01,3.0)},
                {'name': 'polyak',      			'type': 'continuous', 'domain': (0.0001,0.5)}],
            'ppo':
				[{'name': 'actor_layer_1_nodes',    'type': 'categorical',   'domain': (0, 1, 2, 3)}, 
                {'name': 'actor_layer_2_nodes',     'type': 'categorical',   'domain': (0, 1, 2, 3, 4)}, 
                {'name': 'actor_layer_3_nodes',     'type': 'categorical',   'domain': (0, 1, 2, 3, 4)}, 
                {'name': 'critic_layer_1_nodes',    'type': 'categorical',   'domain': (0, 1, 2, 3)}, 
                {'name': 'critic_layer_2_nodes',    'type': 'categorical',   'domain': (0, 1, 2, 3)}, 
                {'name': 'critic_layer_3_nodes',    'type': 'categorical',   'domain': (0, 1, 2, 3, 4)}, 
                {'name': 'discount_factor',         'type': 'continuous', 'domain': (0.9,1.0)}, 
                {'name': 'actor_learning_rate',     'type': 'continuous', 'domain': (0.0001, 0.05)}, 
                {'name': 'critic_learning_rate',    'type': 'continuous', 'domain': (0.0001, 0.05)}, 
                {'name': 'gae_lambda',    	  		'type': 'continuous', 'domain': (0.8, 0.999)},
                {'name': 'beta_entropy', 			'type': 'continuous', 'domain': (0.0001,0.05)},
                {'name': 'clip_likelihood_ratio', 	'type': 'continuous', 'domain': (0.01, 1.0)},],
			'td3':
				[{'name': 'discount_factor',        'type': 'continuous', 'domain': (0.95,1.0)}, 
                {'name': 'actor_learning_rate',     'type': 'continuous', 'domain': (0.00001, 0.01)}, 
                {'name': 'critic_learning_rate',    'type': 'continuous', 'domain': (0.00001, 0.01)},
                {'name': 'actor_l2',     			'type': 'continuous', 'domain': (0.0000001, 0.01)}, 
                {'name': 'critic_l2',    			'type': 'continuous', 'domain': (0.0000001, 0.01)},
                {'name': 'polyak',      			'type': 'continuous', 'domain': (0.00001,0.1)}],
            }