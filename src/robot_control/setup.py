from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'robot_control'

setup(
    name=package_name,
    version='0.0.0',

    packages=['robot_control'],

    data_files=[

        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),

        (
            'share/' + package_name,
            ['package.xml']
        ),

        (
            os.path.join('share', package_name, 'launch'),
                glob('launch/*.launch.py')
        ),

        (
            os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')
        ),

        (
            os.path.join('share', package_name, 'urdf'),
            glob('my_robot_scripts/*.urdf')
        ),
        

    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='ali',
    maintainer_email='ali@todo.todo',

    description='TODO: Package description',
    license='TODO: License declaration',

    extras_require={
        'test': ['pytest'],
    },

    entry_points={
        'console_scripts': [
            'Robot_Arm = robot_control.Robot_Arm:main',
        ],
    },
)