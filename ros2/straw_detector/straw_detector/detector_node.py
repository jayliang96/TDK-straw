"""ROS2 節點：訂閱 RealSense 影像，發佈稻草捆的對準誤差。

偵測邏輯全部沿用 straw.py，本檔只負責 ROS2 的收發、參數與深度換算。

訊息為 straw_interfaces/StrawTarget，欄位與 `straw.py --emit-json` 的
JSON 一一對應，header 沿用來源影像的 header 供控制端對時。

啟用深度後會多出目標中心的相機座標（公尺），控制端就不必自己處理
「同樣的像素偏移在不同距離代表不同實際偏移」這件事。

參數集中在 config/straw_detector.yaml，用 launch 檔啟動：
    ros2 launch straw_detector straw_detector.launch.py
"""

import math
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from straw_interfaces.msg import StrawTarget

from .straw import (DEFAULT_CALIBRATION_FILE, DEFAULT_DEPTH_PATCH_RADIUS,
                    DEFAULT_DEPTH_SCALE, DEFAULT_FILTER_ALPHA,
                    DEFAULT_LOWER_HSV, DEFAULT_MIN_AREA,
                    DEFAULT_MIN_CONFIDENCE, DEFAULT_MORPHOLOGY_KERNEL,
                    DEFAULT_UPPER_HSV, AxisAlignment, AxisAngleFilter,
                    build_depth_fields, load_calibration, process_frame)

# 安裝偏差的參數名對應到 AxisAlignment 建構子的關鍵字。這四個參數沒設時
# 用 NaN 當哨兵，才分得出「沒設」與「設成剛好等於預設」。
ALIGNMENT_PARAMETERS = {
	"robot_angle": "robot_angle",
	"axis_offset_m": "offset_m",
	"axis_offset_ratio": "offset_ratio",
	"axis_yaw_deg": "yaw_deg",
}


class DetectionSettings:
	"""process_frame 需要的設定，欄位名稱與命令列的 args 一致。"""

	def __init__(self, node):
		self.lower_hsv = list(node.get_parameter("lower_hsv").value)
		self.upper_hsv = list(node.get_parameter("upper_hsv").value)
		self.kernel_size = int(node.get_parameter("kernel_size").value)
		self.min_area = int(node.get_parameter("min_area").value)
		self.alignment = build_alignment(node)
		# process_frame 會用它決定標註圖上的軸線顏色。
		self.min_confidence = float(
			node.get_parameter("min_confidence").value
		)


def find_calibration_file(node):
	"""決定要載入的校正檔：參數指定 > package 的 config/ > 不載入。

	與 straw.py 相同，預設載入才是安全的那一邊：機器上忘記帶參數會讓
	機器人瞄偏一個相機側偏的距離，而畫面看起來完全正常。
	"""
	if not node.get_parameter("use_calibration").value:
		return None
	explicit = node.get_parameter("calibration_file").value
	if explicit:
		return Path(explicit)
	default = (
		Path(get_package_share_directory("straw_detector"))
		/ "config" / DEFAULT_CALIBRATION_FILE
	)
	return default if default.is_file() else None


def build_alignment(node):
	"""組出相機的安裝偏差：參數 > 校正檔 > 預設值。

	參數宣告時預設為 NaN，只有 YAML 或命令列真的給了值才會蓋過校正檔。
	校正檔的 image_width 與 aim_center_px 也一併傳入，瞄準軸才能依透視
	畫成斜線並在目標所在高度上量橫向誤差。
	"""
	stored = {}
	calibration_file = find_calibration_file(node)
	if calibration_file is not None:
		stored = load_calibration(calibration_file)
		node.get_logger().info("讀入校正檔: %s" % calibration_file)
	else:
		node.get_logger().warn("未載入校正檔，安裝偏差視為零")

	# 缺的鍵不傳，交給 AxisAlignment 的預設值處理。
	kwargs = {}
	for name, keyword in ALIGNMENT_PARAMETERS.items():
		value = float(node.get_parameter(name).value)
		if math.isnan(value):
			value = stored.get(name)
		if value is not None:
			kwargs[keyword] = value

	alignment = AxisAlignment(
		image_width=stored.get("image_width"),
		aim_center_px=stored.get("aim_center_px"),
		**kwargs,
	)
	node.get_logger().info("安裝偏差: %s" % alignment.describe())
	return alignment


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
		self.declare_parameter("min_confidence", DEFAULT_MIN_CONFIDENCE)
		self.declare_parameter("filter_alpha", DEFAULT_FILTER_ALPHA)
		self.declare_parameter("depth_scale", DEFAULT_DEPTH_SCALE)
		self.declare_parameter(
			"depth_patch_radius", DEFAULT_DEPTH_PATCH_RADIUS
		)
		self.declare_parameter("sync_slop", 0.05)
		self.declare_parameter("lower_hsv", list(DEFAULT_LOWER_HSV))
		self.declare_parameter("upper_hsv", list(DEFAULT_UPPER_HSV))
		self.declare_parameter("kernel_size", DEFAULT_MORPHOLOGY_KERNEL)
		self.declare_parameter("min_area", DEFAULT_MIN_AREA)
		for name in ALIGNMENT_PARAMETERS:
			self.declare_parameter(name, float("nan"))
		self.declare_parameter("use_calibration", True)
		# 空字串代表用 package 內 config/ 的校正檔。
		self.declare_parameter("calibration_file", "")

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
			StrawTarget, self.get_parameter("target_topic").value, 10
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

		# 沒有要發佈標註影像時就不必繪圖，那佔整體約四分之一的時間。
		outcome = process_frame(
			frame, self.settings, self.angle_filter, self.publish_annotated
		)
		if outcome is None:
			self.publish_payload(
				{"valid": False, "reason": "no_detection"}, color_message
			)
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

		實際換算與 straw.py 共用，避免命令列與 ROS2 兩邊各寫一份。
		"""
		if depth_message is None or self.intrinsics is None:
			return {"has_depth": False}

		try:
			depth_image = self.bridge.imgmsg_to_cv2(
				depth_message, desired_encoding="passthrough"
			)
		except Exception as error:
			self.get_logger().warn("深度影像轉換失敗: %s" % error)
			return {"has_depth": False}

		return build_depth_fields(
			center_px,
			depth_image,
			self.intrinsics,
			self.depth_patch_radius,
			self.depth_scale,
			self.settings.alignment,
		)

	def publish_payload(self, payload, source_message):
		"""把 straw.py 的 payload dict 攤成 StrawTarget 發佈。

		header 沿用來源影像的，供控制端對時。dict 裡沒有的欄位維持 msg
		的零值；valid=false 時 reason 說明是沒偵測到還是可信度不足。
		"""
		message = StrawTarget()
		message.header = source_message.header
		message.valid = bool(payload.get("valid", False))
		message.reason = str(payload.get("reason", ""))
		message.confidence = float(payload.get("confidence", 0.0))
		if message.valid:
			message.heading_error_deg = float(payload["heading_error_deg"])
			message.lateral_error_px = float(payload["lateral_error_px"])
			message.lateral_error_ratio = float(payload["lateral_error_ratio"])
			message.angle_deg = float(payload["angle_deg"])
			message.angle_sigma_deg = float(payload["angle_sigma_deg"])
			message.center_px = [float(v) for v in payload["center_px"]]
			message.image_size = [int(v) for v in payload["image_size"]]
		message.has_depth = bool(payload.get("has_depth", False))
		if message.has_depth:
			message.distance_m = float(payload["distance_m"])
			message.lateral_error_m = float(payload["lateral_error_m"])
			message.camera_lateral_m = float(payload["camera_lateral_m"])
			x, y, z = payload["position_m"]
			message.position_m.x = float(x)
			message.position_m.y = float(y)
			message.position_m.z = float(z)
		self.target_publisher.publish(message)


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
