import subprocess


for i in range(2):
    num_envs = 100*(i+1)
    record_right_arm = subprocess.run(['python3', f'SKRL_test_performance_Submodule.py', '--num_envs', f'{num_envs}'])
