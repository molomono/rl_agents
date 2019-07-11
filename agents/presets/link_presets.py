import sys, os

if len(sys.argv) > 1:
	file_list = [file_str for file_str in os.listdir('.') if '.py' in file_str]
	for file_str in file_list:
		if 'link_presets' not in file_str:
			os.system('echo Made link for: ./{} {}'.format(file_str, sys.argv[1])) 
			os.system('ln -f {} {}'.format(file_str, sys.argv[1]))
else: 
	print('Missing the rl-coach presets file location')
	
