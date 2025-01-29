from setuptools import setup

setup(
    description='Disk Ring Adjusted Geometry yields Scattering phase function',
    author='Maxime Roumesy',
    author_email='mroumesy@gmail.com',
    url='https://github.com/mroumesy/DRAGyS',
    project_urls={'Documentation': 'https://github.com/mroumesy/DRAGyS'},
    name='DRAGyS',
    version='1.0.0',
    license='MIT',
    packages=['DRAGyS'],
    install_requires=[
        'numpy','matplotlib', 'scipy', 'pickle', 'PyQt6', 'scipy', 'astropy', 'multiprocess'
        ],
)