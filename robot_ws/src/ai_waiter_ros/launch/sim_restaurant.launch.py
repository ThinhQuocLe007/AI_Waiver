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
    # We add both the install path and the source path to be extra safe
    model_path = os.path.join(pkg_ai_waiter_ros, 'models')
    src_model_path = os.path.join(get_package_share_directory('ai_waiter_ros'), '..', '..', '..', 'src', 'ai_waiter_ros', 'models')
    
    # Ignition Gazebo (Fortress) uses this
    set_ign_path = AppendEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', model_path)
    # Newer Gazebo (Garden/Ionic) uses this
    set_gz_path = AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', model_path)

    # 3. Path to your specific world file
    world_file = os.path.join(pkg_ai_waiter_ros, 'worlds', 'ai_waiver_restaurant.sdf')

    # 4. Include the default Gazebo launch file
    start_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items()
    )

    return LaunchDescription([
        set_ign_path,
        set_gz_path,
        start_gazebo
    ])
