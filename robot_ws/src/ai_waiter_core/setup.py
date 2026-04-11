from setuptools import setup
import os
from glob import glob

package_name = 'ai_waiter_core'

setup(
    name=package_name,
    version='0.1.0',
    # We use the list we just corrected in pyproject.toml
    packages=[
        package_name,
        package_name + '.core',
        package_name + '.core.schemas',
        package_name + '.core.utils',
        package_name + '.interfaces',
        package_name + '.interfaces.ros_nodes',
        package_name + '.interfaces.websocket',
        package_name + '.orchestrator',
        package_name + '.output',
        package_name + '.perception',
        package_name + '.tools',
        package_name + '.tools.data',
        package_name + '.tools.search_engines'
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Core LLM and Agent logic for AI Waiter',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # NOTE: Update the path below to point to wherever your actual 'main' function is located!
            # If your main function is inside interfaces/ros_nodes/main_node.py:
            # 'ai_brain = ai_waiter_core.interfaces.ros_nodes.main_node:main'
            'ai_brain = ai_waiter_core.interfaces.ros_nodes.ai_brain_node:main'
        ],
    },
)
