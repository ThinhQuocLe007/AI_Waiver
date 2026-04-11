import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

def generate_launch_description():
    # 1. Định vị thư mục workspace
    pkg_share = get_package_share_directory('ai_waiter_ros')
    
    # Bước lùi để tìm thư mục src (Fail-safe)
    ws_root = pkg_share
    for _ in range(4):
        ws_root = os.path.dirname(ws_root)
    
    src_worlds_path = os.path.join(ws_root, 'src', 'ai_waiter_ros', 'worlds')

    # 2. Khai báo tham số world_file (Mặc định là manual_world.sdf)
    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value='manual_world.sdf',
        description='Name of the world file to load from src/ai_waiter_ros/worlds/'
    )

    # 3. Tạo đường dẫn tuyệt đối đến file world được chọn
    # Nếu Gazebo không tìm thấy, nó sẽ báo lỗi thay vì load sảnh trống
    world_path = PathJoinSubstitution([
        src_worlds_path,
        LaunchConfiguration('world_file')
    ])

    # 4. Tìm package gazebo
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # 5. Khởi động Gazebo
    start_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': [ '-r ', world_path ]
        }.items()
    )

    return LaunchDescription([
        world_file_arg,
        start_gazebo
    ])
