from setuptools import find_packages, setup

HYPEN_E_DASH = '-e .'

def get_requirements(file_path):
    """
    This function reads a requirements file and returns a list of packages.
    """
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPEN_E_DASH in requirements:
            requirements.remove(HYPEN_E_DASH)

    return requirements

setup(
    name="ml_project",
    version="0.0.1",
    author="Özgür",
    author_email="msozgur44@gmail.com",
    packages=find_packages(),  
    install_requires=get_requirements('requirements.txt') 
)
