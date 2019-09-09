'''Make a new python script for this file, that saves the hyperparameters to a external file that can be read on runtime'''
import pandas as pd

# Declare the hyper-parameters that can be altered
param_list = ['a_lr', 'c_lr', 'a_l2', 'c_l2', 'a_bs', 'c_bs', 'noise', 'n_heatup', 'discount', 'polyak' ]
hyper_parameters_dict = { 'params': param_list, 'values':[None] * len(param_list) }
del param_list

change_params = True
while change_params:
	print("What hyperparameter would you like to set:")
	print("[0]: actor learning rate")
	print("[1]: critic learning rate")
	print("[2]: actor l2 regularization")
	print("[3]: critic l2 regularization")
	print("[4]: actor batch size")
	print("[5]: critic batch size")
	print("[6]: action noise")
	print("[7]: number of heatup steps")
	print("[8]: discount factor")
	print("[9]: rate for copying weights to target")
	print("[-]: Exit menu")
	print("Make Selection: "); selection = input()
	
	if selection is not '-':
		selection = int(selection)

	if selection is 0:
		print('Input actor learning rate: ')
		hyper_parameters_dict['values'][selection] = float(input())
	elif selection is 1:
		print('Input critic learning rate: ')
		hyper_parameters_dict['values'][selection] = float(input())
	elif selection is 2:
		print('Input actor l2 regularization: ')
		hyper_parameters_dict['values'][selection] = float(input())
	elif selection is 3:
		print('Input critic l2 regularization: ')
		hyper_parameters_dict['values'][selection] = float(input())
	elif selection is 4:
		print('Input actor batch size: ')
		hyper_parameters_dict['values'][selection] = int(input())
	elif selection is 5:
		print('Input critic batch size: ')
		hyper_parameters_dict['values'][selection] = int(input())
	elif selection is 6:
		print('Input action noise: ')
		hyper_parameters_dict['values'][selection] = float(input())
	elif selection is 7:
		print('Number of heatup steps: ')
		hyper_parameters_dict['values'][selection] = int(input())
	elif selection is 8:
		print('Discount factor: ')
		hyper_parameters_dict['values'][selection] = float(input())
	elif selection is 9:
		print('Rate of copying weights to target: ')
		hyper_parameters_dict['values'][selection] = int(input())
	elif selection is ['-']:
		break
	
	print("Would you like to continue to change params: [y/n]")
	if input() in ('y', 'Y', 'yes', 'YES', 'Yes'): 
		change_params = True
	else:
		change_params = False

dataframe = pd.DataFrame(hyper_parameters_dict)
print(dataframe)
import os
home_path = os.path.expanduser('~')
dataframe.to_csv(home_path+'/hyper_parameter_files/'+'hyper_params.csv', index=False)
