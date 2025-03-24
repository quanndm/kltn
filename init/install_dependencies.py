import os

def install_packages(path_requires):
    path_req = os.path.join(path_requires, 'requirements.txt')
    os.system(f'pip install -r {path_req}')
    