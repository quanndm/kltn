import os

def install_packages(path_requires):
    path_req = os.path.join(path_requires, 'requirements.txt')
    os.system('pip uninstall -y numpy')
    os.system(f'pip install -r {path_req}')
    os.system('pip install git+https://github.com/lucasb-eyer/pydensecrf.git')