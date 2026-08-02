from setuptools import find_packages, setup

package_name = 'ultrasonic_mapping'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ali',
    maintainer_email='ali@todo.todo',
    description='TODO',
    license='TODO',
    entry_points={
        'console_scripts': [
            'ultrasonic = ultrasonic_mapping.connections.ultrasonic:main',
        ],
    },
)