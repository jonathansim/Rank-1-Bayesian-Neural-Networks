'''
This file is used to convert a conda yml file to a requirements.txt file 
'''

import yaml

with open('environment_setup.yml', 'r') as file:
    env = yaml.safe_load(file)

with open('requirements.txt', 'w') as file:
    for dep in env['dependencies']:
        if isinstance(dep, str):
            file.write(dep + '\n')
        elif isinstance(dep, dict) and 'pip' in dep:
            for pip_dep in dep['pip']:
                file.write(pip_dep + '\n')
