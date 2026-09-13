"""一行起完整 pipeline：realsense2_camera 加 straw_node。

相機端兩個設定不能省：
- align_depth.enable 沒開，aligned_depth_to_color 話題根本不存在，
  ApproximateTimeSynchronizer 會靜默地一則都收不到。
- rgb_camera.profile 要對上校正檔的 image_width（目前 1280），否則
  axis_offset_ratio 只在校正時的視角下正確。
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
	realsense_share = Path(get_package_share_directory("realsense2_camera"))
	straw_share = Path(get_package_share_directory("straw_detector"))

	return LaunchDescription([
		DeclareLaunchArgument(
			"rgb_profile",
			default_value="1280x720x30",
			description="彩色串流的 寬x高x幀率，要對上校正檔的 image_width",
		),
		DeclareLaunchArgument("publish_annotated", default_value="false"),
		IncludeLaunchDescription(
			PythonLaunchDescriptionSource(
				str(realsense_share / "launch" / "rs_launch.py")
			),
			# 彩色 profile 的參數名在 realsense-ros 4.51 與 4.55 之間改過
			# （profile -> color_profile）；include 會忽略沒宣告的參數，
			# 兩個都給就能跨版本。深度不必指定，對齊後會重投影到彩色解析度。
			launch_arguments={
				"align_depth.enable": "true",
				"rgb_camera.profile": LaunchConfiguration("rgb_profile"),
				"rgb_camera.color_profile": LaunchConfiguration("rgb_profile"),
			}.items(),
		),
		IncludeLaunchDescription(
			PythonLaunchDescriptionSource(
				str(straw_share / "launch" / "straw_detector.launch.py")
			),
			launch_arguments={
				"camera_namespace": "/camera/camera",
				"use_depth": "true",
				"publish_annotated": LaunchConfiguration("publish_annotated"),
			}.items(),
		),
	])
