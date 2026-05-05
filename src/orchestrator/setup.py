from setuptools import setup
import os

package_name = 'orchestrator'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    description='Orchestrator for managing and controlling TIAGo',
    license='TODO',
    entry_points={
        'console_scripts': [
            'orchestrator_node = orchestrator.orchestrator_node:main',
        ],
    },
)