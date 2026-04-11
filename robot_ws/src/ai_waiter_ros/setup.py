from setuptools import setup
import os
from glob import glob

package_name = 'ai_waiter_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        
        # This part correctly installs every model folder without flattening them!
        (os.path.join('share', package_name, 'models/delivery_track'), glob('models/delivery_track/*')),
        (os.path.join('share', package_name, 'models/delivery_track_corner'), glob('models/delivery_track_corner/*')),
        (os.path.join('share', package_name, 'models/kitchen_hub'), glob('models/kitchen_hub/*')),
        (os.path.join('share', package_name, 'models/kitchen_monitor'), glob('models/kitchen_monitor/*')),
        (os.path.join('share', package_name, 'models/table'), glob('models/table/*')),
        (os.path.join('share', package_name, 'models/chair'), glob('models/chair/*')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    description='Simulation assets for AI Waiter',
    license='Apache-2.0',
    entry_points={'console_scripts': []},
)
