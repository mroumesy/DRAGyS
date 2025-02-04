from setuptools import setup, find_packages 

setup( 
name='dragys',  # name of package（which will be shown when you type: pip list） 
version="1.0.0",  # version
description='Disk Ring Adjusted Geometry yields Scattering phase function',  # description
author='Maxime Roumesy',  # name of developer
packages= find_packages(),  # list of models 
license='MIT',  # licence 
install_requires=[
        'numpy','matplotlib', "PyQt6", "scipy", "astropy"
        ]
)