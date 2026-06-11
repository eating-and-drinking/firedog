"""
launch/robot_dog_agent.launch.py
ROS 2 Launch 文件：以 ROS 2 节点方式启动 Agent 系统
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="full",
        description="启动模式: full | voice_only | agent_cli",
    )
    mock_arg = DeclareLaunchArgument(
        "mock_robot",
        default_value="false",
        description="是否使用 mock 机器狗",
    )

    agent_node = ExecuteProcess(
        cmd=[
            "python3",
            "scripts/launch.py",
            "--mode", LaunchConfiguration("mode"),
        ],
        output="screen",
        name="robot_dog_agent",
    )

    return LaunchDescription([
        mode_arg,
        mock_arg,
        agent_node,
    ])
