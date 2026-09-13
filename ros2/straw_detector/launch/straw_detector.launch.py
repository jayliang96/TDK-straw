"""只起 straw_node。相機另外起（或用 straw_with_camera.launch.py 一起起）。

realsense-ros 較新版本的話題含兩層 camera 命名空間（/camera/camera/...），
舊版只有一層（/camera/...）。用 camera_namespace 一次改掉三個話題，
先 `ros2 topic list` 確認再覆寫。
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
	share = Path(get_package_share_directory("straw_detector"))
	camera_namespace = LaunchConfiguration("camera_namespace")

	return LaunchDescription([
		DeclareLaunchArgument(
			"camera_namespace",
			default_value="/camera/camera",
			description="realsense-ros 話題的前綴，舊版為 /camera",
		),
		DeclareLaunchArgument(
			"params_file",
			default_value=str(share / "config" / "straw_detector.yaml"),
		),
		DeclareLaunchArgument("use_depth", default_value="true"),
		DeclareLaunchArgument("publish_annotated", default_value="false"),
		Node(
			package="straw_detector",
			executable="straw_node",
			name="straw_detector",
			output="screen",
			parameters=[
				LaunchConfiguration("params_file"),
				{
					"image_topic": [camera_namespace, "/color/image_raw"],
					"depth_topic": [
						camera_namespace, "/aligned_depth_to_color/image_raw"
					],
					"camera_info_topic": [
						camera_namespace, "/color/camera_info"
					],
					"use_depth": LaunchConfiguration("use_depth"),
					"publish_annotated": LaunchConfiguration(
						"publish_annotated"
					),
				},
			],
		),
	])
