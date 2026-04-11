import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # 1. Get the install path to your ROS 2 package
    pkg_ai_waiter_ros = get_package_share_directory('ai_waiter_ros')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # 2. Tell Gazebo where to find your custom models (table, chair, track)
    set_model_path = AppendEnvironmentVariable(
        'IGN_GAZEBO_RESOURCE_PATH',
        os.path.join(pkg_ai_waiter_ros, 'models')
    )

    # 3. Path to your specific world file
    world_file = os.path.join(pkg_ai_waiter_ros, 'worlds', 'ai_waiver_restaurant.sdf')

    # 4. Include the default Gazebo launch file, but pass it your custom world
    start_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        # '-r' means start running the physics immediately
        launch_arguments={'gz_args': f'-r {world_file}'}.items()
    )

    return LaunchDescription([
        set_model_path,
        start_gazebo
    ])
