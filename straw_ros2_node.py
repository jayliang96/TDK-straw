"""ROS2 節點：訂閱 RealSense 影像，發佈稻草捆的對準誤差。

偵測邏輯全部沿用 straw.py，本檔只負責 ROS2 的收發、參數與深度換算。

訊息格式為 std_msgs/String 承載 JSON，與 `straw.py --emit-json` 的內容
相同，另外多了取自影像 header 的時間戳。選 String 而非自訂訊息是為了
免去 colcon build；若要型別安全的版本，把 publish_payload 換成自訂 .msg
即可，其餘邏輯不必動。

啟用深度後會多出目標中心的相機座標（公尺），控制端就不必自己處理
「同樣的像素偏移在不同距離代表不同實際偏移」這件事。

用法：
    python straw_ros2_node.py --ros-args \
        -p image_topic:=/camera/camera/color/image_raw \
        -p use_depth:=true
"""

import json

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from straw import (DEFAULT_LOWER_HSV, DEFAULT_MIN_AREA,
                   DEFAULT_MORPHOLOGY_KERNEL, DEFAULT_ROBOT_ANGLE,
                   DEFAULT_UPPER_HSV, AxisAngleFilter, process_frame)


class DetectionSettings:
	"""process_frame 需要的設定，欄位名稱與命令列的 args 一致。"""

	def __init__(self, node):
		self.lower_hsv = list(node.get_parameter("lower_hsv").value)
		self.upper_hsv = list(node.get_parameter("upper_hsv").value)
		self.kernel_size = int(node.get_parameter("kernel_size").value)
		self.min_area = int(node.get_parameter("min_area").value)
		self.robot_angle = float(node.get_parameter("robot_angle").value)


class StrawDetectorNode(Node):
	"""訂閱彩色（可選深度）影像，發佈稻草捆相對機器人的誤差。"""

	def __init__(self):
		super().__init__("straw_detector")

		# realsense-ros 較新版本的預設話題含兩層 camera 命名空間；
		# 舊版是 /camera/color/image_raw，用 ros2 topic list 確認後以參數覆寫。
		self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
		self.declare_parameter(
			"depth_topic",
			"/camera/camera/aligned_depth_to_color/image_raw",
		)
		self.declare_parameter(
			"camera_info_topic", "/camera/camera/color/camera_info"
		)
		self.declare_parameter("use_depth", True)
		self.declare_parameter("target_topic", "straw/target")
		self.declare_parameter("annotated_topic", "straw/annotated")
		self.declare_parameter("publish_annotated", False)
		self.declare_parameter("min_confidence", 0.5)
		self.declare_parameter("filter_alpha", 0.25)
		# RealSense 的深度影像是 16UC1，單位為公釐。
		self.declare_parameter("depth_scale", 0.001)
		# 單一像素的深度常常是 0（反光、邊緣、超出量程），取鄰域中位數。
		self.declare_parameter("depth_patch_radius", 6)
		self.declare_parameter("sync_slop", 0.05)
		self.declare_parameter("lower_hsv", list(DEFAULT_LOWER_HSV))
		self.declare_parameter("upper_hsv", list(DEFAULT_UPPER_HSV))
		self.declare_parameter("kernel_size", DEFAULT_MORPHOLOGY_KERNEL)
		self.declare_parameter("min_area", DEFAULT_MIN_AREA)
		self.declare_parameter("robot_angle", DEFAULT_ROBOT_ANGLE)

		self.settings = DetectionSettings(self)
		self.min_confidence = float(self.get_parameter("min_confidence").value)
		self.use_depth = bool(self.get_parameter("use_depth").value)
		self.depth_scale = float(self.get_parameter("depth_scale").value)
		self.depth_patch_radius = int(
			self.get_parameter("depth_patch_radius").value
		)
		self.publish_annotated = bool(
			self.get_parameter("publish_annotated").value
		)

		self.bridge = CvBridge()
		# 跨影格的軸線濾波必須在迴圈外建立一次，每格重建會讓平滑完全失效。
		self.angle_filter = AxisAngleFilter(
			alpha=float(self.get_parameter("filter_alpha").value)
		)
		# 相機內參由 camera_info 提供，收到第一則就夠用。
		self.intrinsics = None

		self.target_publisher = self.create_publisher(
			String, self.get_parameter("target_topic").value, 10
		)
		self.annotated_publisher = None
		if self.publish_annotated:
			self.annotated_publisher = self.create_publisher(
				Image, self.get_parameter("annotated_topic").value, 10
			)

		image_topic = self.get_parameter("image_topic").value
		if self.use_depth:
			self.setup_synchronised_subscriptions(image_topic)
		else:
			# RealSense 以 SensorDataQoS（best effort）發佈影像。訂閱端若用
			# 預設的 reliable QoS，兩邊的 QoS 不相容，會一則訊息都收不到。
			self.create_subscription(
				Image, image_topic, self.on_color_only, qos_profile_sensor_data
			)
			self.get_logger().info("訂閱影像話題: %s（未啟用深度）" % image_topic)

	def setup_synchronised_subscriptions(self, image_topic):
		"""彩色與深度必須時間同步，否則會拿這一格的中心去查上一格的深度。"""
		import message_filters

		depth_topic = self.get_parameter("depth_topic").value
		info_topic = self.get_parameter("camera_info_topic").value

		color_subscriber = message_filters.Subscriber(
			self, Image, image_topic, qos_profile=qos_profile_sensor_data
		)
		depth_subscriber = message_filters.Subscriber(
			self, Image, depth_topic, qos_profile=qos_profile_sensor_data
		)
		# 兩個串流的時間戳不會完全相同，用近似同步搭配容許誤差。
		self.synchroniser = message_filters.ApproximateTimeSynchronizer(
			[color_subscriber, depth_subscriber],
			queue_size=10,
			slop=float(self.get_parameter("sync_slop").value),
		)
		self.synchroniser.registerCallback(self.on_color_and_depth)

		self.create_subscription(
			CameraInfo, info_topic, self.on_camera_info, 10
		)
		self.get_logger().info(
			"訂閱彩色 %s 與深度 %s，內參來自 %s"
			% (image_topic, depth_topic, info_topic)
		)

	def on_camera_info(self, message):
		"""記下相機內參。對齊後的深度與彩色共用同一組內參。"""
		if self.intrinsics is not None:
			return
		fx, _, cx, _, fy, cy = (
			message.k[0], message.k[1], message.k[2],
			message.k[3], message.k[4], message.k[5],
		)
		if fx <= 0.0 or fy <= 0.0:
			self.get_logger().warn("camera_info 的焦距無效，忽略")
			return
		self.intrinsics = (float(fx), float(fy), float(cx), float(cy))
		self.get_logger().info(
			"取得內參 fx=%.1f fy=%.1f cx=%.1f cy=%.1f" % self.intrinsics
		)

	def on_color_only(self, color_message):
		self.handle(color_message, None)

	def on_color_and_depth(self, color_message, depth_message):
		self.handle(color_message, depth_message)

	def handle(self, color_message, depth_message):
		try:
			frame = self.bridge.imgmsg_to_cv2(
				color_message, desired_encoding="bgr8"
			)
		except Exception as error:  # cv_bridge 的例外型別依編碼而異
			self.get_logger().warn("彩色影像轉換失敗: %s" % error)
			return

		outcome = process_frame(frame, self.settings, self.angle_filter)
		if outcome is None:
			self.publish_payload({"valid": False}, color_message)
			return

		payload = outcome["payload"]
		# 可信度不足時視同沒有偵測到：寧可讓控制端維持前一個指令，
		# 也不要送出一個看似合理但方向可能錯 90 度的誤差。
		if payload["confidence"] < self.min_confidence:
			self.publish_payload(
				{
					"valid": False,
					"reason": "low_confidence",
					"confidence": payload["confidence"],
				},
				color_message,
			)
			return

		payload = dict(payload)
		payload.update(
			self.measure_depth(outcome["result"]["center"], depth_message)
		)
		self.publish_payload(payload, color_message)

		if self.annotated_publisher is not None:
			annotated = self.bridge.cv2_to_imgmsg(
				outcome["visualization"], encoding="bgr8"
			)
			annotated.header = color_message.header
			self.annotated_publisher.publish(annotated)

	def measure_depth(self, center_px, depth_message):
		"""把目標中心像素還原成相機座標系的公尺座標。

		回傳的欄位一律存在，has_depth 說明這次有沒有取到有效深度，
		控制端不必用「欄位在不在」來判斷。
		"""
		missing = {"has_depth": False}
		if depth_message is None or self.intrinsics is None:
			return missing

		try:
			depth_image = self.bridge.imgmsg_to_cv2(
				depth_message, desired_encoding="passthrough"
			)
		except Exception as error:
			self.get_logger().warn("深度影像轉換失敗: %s" % error)
			return missing

		depth_metres = self.sample_depth(depth_image, center_px)
		if depth_metres is None:
			return missing

		fx, fy, cx, cy = self.intrinsics
		# 針孔模型反投影。x 向右、y 向下、z 向前，單位公尺。
		x = (float(center_px[0]) - cx) * depth_metres / fx
		y = (float(center_px[1]) - cy) * depth_metres / fy
		return {
			"has_depth": True,
			"distance_m": round(depth_metres, 4),
			"lateral_error_m": round(x, 4),
			"position_m": [round(x, 4), round(y, 4), round(depth_metres, 4)],
		}

	def sample_depth(self, depth_image, center_px):
		"""取中心鄰域的深度中位數，單位公尺；沒有有效值時回傳 None。

		單一像素的深度常常是 0：反光、物體邊緣、超出量程都會造成破洞。
		取鄰域並剔除 0 之後再取中位數，比直接讀一個像素穩定得多。
		"""
		height, width = depth_image.shape[:2]
		u = int(round(float(center_px[0])))
		v = int(round(float(center_px[1])))
		radius = self.depth_patch_radius
		left = max(0, u - radius)
		right = min(width, u + radius + 1)
		top = max(0, v - radius)
		bottom = min(height, v + radius + 1)
		if left >= right or top >= bottom:
			return None

		patch = depth_image[top:bottom, left:right].astype(np.float32)
		valid = patch[patch > 0.0]
		if valid.size == 0:
			return None

		return float(np.median(valid)) * self.depth_scale

	def publish_payload(self, payload, source_message):
		"""發佈一筆結果，時間戳沿用來源影像，供控制端對時。"""
		payload = dict(payload)
		payload["stamp_sec"] = int(source_message.header.stamp.sec)
		payload["stamp_nanosec"] = int(source_message.header.stamp.nanosec)
		self.target_publisher.publish(
			String(data=json.dumps(payload, ensure_ascii=False))
		)


def main(args=None):
	rclpy.init(args=args)
	node = StrawDetectorNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.try_shutdown()


if __name__ == "__main__":
	main()
