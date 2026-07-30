from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'robot_ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    package_data={
        "robot_ai": ["prompts/*.txt"],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ali',
    maintainer_email='ali@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'RobotAi = robot_ai.brain:main',
        ],
    },
)
