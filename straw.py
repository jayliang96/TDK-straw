"""從二值化 mask 偵測最大的稻草捆區域。

此版本先用離線 mask 驗證幾何分析，偵測邏輯不依賴相機或機器人控制器。
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# 與 c1.py 相同的預設 HSV 範圍。
DEFAULT_LOWER_HSV = (21, 65, 67)
DEFAULT_UPPER_HSV = (33, 255, 255)
DEFAULT_MIN_AREA = 5000
DEFAULT_MORPHOLOGY_KERNEL = 11
DEFAULT_RANSAC_THRESHOLD = 12.0
DEFAULT_RANSAC_ITERATIONS = 300
DEFAULT_MIN_AXIS_RATIO = 1.5
DEFAULT_IMAGE_OUTPUT = "output/straw_detection.png"
DEFAULT_VIDEO_OUTPUT = "output/straw_detection.mp4"


def load_image(image_path):
	"""讀取並驗證 BGR 原始影像。"""
	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"找不到圖片，請檢查路徑: {image_path}")
	return image


def create_color_mask(image_bgr, lower_bound, upper_bound, kernel_size=11):
	"""依照 c1.py 的 HSV 色彩範圍建立稻草捆 mask。"""
	image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

	# 與 c1.py 相同：先去除小雜訊，再填補目標區域的小缺口。
	kernel = cv2.getStructuringElement(
		cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
	)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	return mask


def load_mask(mask_path):
	"""讀取外部 mask，保留給離線測試與除錯使用。"""
	mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
	if mask is None:
		raise FileNotFoundError(f"找不到 mask，請檢查路徑: {mask_path}")

	_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
	return mask


def clean_mask(mask, kernel_size=5):
	"""移除小雜訊，並填補稻草捆輪廓中的小缺口。"""
	if kernel_size <= 1:
		return mask.copy()

	kernel = cv2.getStructuringElement(
		cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
	)
	cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
	return cleaned


def select_largest_component(mask, min_area=5000):
	"""找出最大的前景連通區，只保留最可能的稻草捆。"""
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
		mask, connectivity=8
	)

	if num_labels <= 1:
		return None

	component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
	area = int(stats[component_label, cv2.CC_STAT_AREA])
	if area < min_area:
		return None

	target_mask = np.zeros_like(mask)
	target_mask[labels == component_label] = 255

	x = int(stats[component_label, cv2.CC_STAT_LEFT])
	y = int(stats[component_label, cv2.CC_STAT_TOP])
	width = int(stats[component_label, cv2.CC_STAT_WIDTH])
	height = int(stats[component_label, cv2.CC_STAT_HEIGHT])

	return {
		"mask": target_mask,
		"area": area,
		"bbox": (x, y, width, height),
		"centroid": tuple(centroids[component_label]),
	}


def axis_angle_difference(first_angle, second_angle):
	"""計算兩條沒有正反方向之分的軸線，其最小夾角。"""
	difference = abs(first_angle - second_angle) % 180.0
	return min(difference, 180.0 - difference)


class AxisAngleFilter:
	"""針對 180 度週期的物體軸線做連續影格平滑。"""

	def __init__(self, alpha=0.25):
		self.alpha = alpha
		self.filtered_angle = None

	def update(self, angle):
		"""加入一個角度，回傳平滑後的軸線角度。"""
		angle_radians = np.deg2rad(angle * 2.0)
		current = np.array(
			[np.cos(angle_radians), np.sin(angle_radians)], dtype=np.float32
		)

		if self.filtered_angle is None:
			self.filtered_angle = float(angle) % 180.0
			return self.filtered_angle

		filtered_radians = np.deg2rad(self.filtered_angle * 2.0)
		previous = np.array(
			[
				np.cos(filtered_radians),
				np.sin(filtered_radians),
			],
			dtype=np.float32,
		)
		combined = (1.0 - self.alpha) * previous + self.alpha * current
		self.filtered_angle = (
			float(np.degrees(np.arctan2(combined[1], combined[0]))) / 2.0
		) % 180.0
		return self.filtered_angle


def fit_line_ransac(
	points,
	seed_direction,
	threshold=DEFAULT_RANSAC_THRESHOLD,
	iterations=DEFAULT_RANSAC_ITERATIONS,
):
	"""使用 RANSAC 擬合側邊，排除 mask 毛刺與錯誤邊界點。"""
	points = np.asarray(points, dtype=np.float32)
	if len(points) < 5:
		return None

	rng = np.random.default_rng(42)
	best_inliers = None
	best_error = np.inf
	for _ in range(iterations):
		first, second = points[rng.choice(len(points), 2, replace=False)]
		line_vector = second - first
		length = np.linalg.norm(line_vector)
		if length < 1.0:
			continue

		line_vector /= length
		distance = np.abs(
			(line_vector[0] * (points[:, 1] - first[1]))
			- (line_vector[1] * (points[:, 0] - first[0]))
		)
		inliers = distance <= threshold
		inlier_count = int(np.count_nonzero(inliers))
		inlier_error = float(np.mean(distance[inliers])) if inlier_count else np.inf
		if best_inliers is None or (inlier_count, -inlier_error) > (
			int(np.count_nonzero(best_inliers)),
			-best_error,
		):
			best_inliers = inliers
			best_error = inlier_error

	if best_inliers is None or np.count_nonzero(best_inliers) < 5:
		return None

	line = cv2.fitLine(
		points[best_inliers], cv2.DIST_L2, 0, 0.01, 0.01
	)
	direction = np.array([line[0, 0], line[1, 0]], dtype=np.float32)
	direction /= np.linalg.norm(direction)
	seed_direction = np.asarray(seed_direction, dtype=np.float32)
	if np.dot(direction, seed_direction) < 0:
		direction = -direction
	point = np.array([line[2, 0], line[3, 0]], dtype=np.float32)
	return {
		"point": point,
		"direction": direction,
		"inliers": points[best_inliers],
		"inlier_ratio": float(np.mean(best_inliers)),
		"error": best_error,
	}


def fit_side_edges(target_mask, seed_direction):
	"""從 mask 輪廓的左右邊界擬合兩條側邊，並將向量相加。

	seed_direction 只用來建立物體的初始縱向座標，最後方向由左右側邊
	的向量和決定。這可以降低透視造成單一側邊偏移的影響。
	"""
	contours, _ = cv2.findContours(
		target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
	)
	if not contours:
		return None

	contour = max(contours, key=cv2.contourArea)
	contour_points = contour[:, 0, :].astype(np.float32)
	center = contour_points.mean(axis=0)
	axis = np.asarray(seed_direction, dtype=np.float32)
	axis /= np.linalg.norm(axis)
	perpendicular = np.array([-axis[1], axis[0]], dtype=np.float32)

	# 將輪廓點投影到縱向和橫向座標，只使用中間 50%，排除兩端圓弧。
	longitudinal = (contour_points - center) @ axis
	lateral = (contour_points - center) @ perpendicular
	long_min, long_max = np.percentile(longitudinal, [25, 75])
	valid = (longitudinal >= long_min) & (longitudinal <= long_max)
	valid_points = contour_points[valid]
	valid_longitudinal = longitudinal[valid]
	valid_lateral = lateral[valid]
	if len(valid_points) < 20:
		return None

	# 每個縱向切片取邊界附近多個點的中位數，不讓單一毛刺決定邊界。
	bin_edges = np.linspace(long_min, long_max, 21)
	left_points = []
	right_points = []
	for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
		in_bin = (
			(valid_longitudinal >= lower)
			& (valid_longitudinal <= upper)
		)
		if not np.any(in_bin):
			continue
		bin_points = valid_points[in_bin]
		bin_lateral = valid_lateral[in_bin]
		boundary_count = max(3, int(np.ceil(len(bin_points) * 0.08)))
		left_boundary = bin_points[np.argsort(bin_lateral)[:boundary_count]]
		right_boundary = bin_points[np.argsort(bin_lateral)[-boundary_count:]]
		left_points.append(np.median(left_boundary, axis=0))
		right_points.append(np.median(right_boundary, axis=0))

	if len(left_points) < 5 or len(right_points) < 5:
		return None

	left_fit = fit_line_ransac(left_points, axis)
	right_fit = fit_line_ransac(right_points, axis)
	if left_fit is None or right_fit is None:
		return None
	left_point = left_fit["point"]
	left_direction = left_fit["direction"]
	right_point = right_fit["point"]
	right_direction = right_fit["direction"]

	# 兩條邊都已被統一為相同縱向，向量和就是梯形的中心朝向。
	summed_direction = left_direction + right_direction
	if np.linalg.norm(summed_direction) < 1e-6:
		return None
	summed_direction /= np.linalg.norm(summed_direction)

	def line_endpoints(point, direction):
		start = point + direction * (
			long_min - np.dot(point - center, direction)
		)
		end = point + direction * (
			long_max - np.dot(point - center, direction)
		)
		return start, end

	left_start, left_end = line_endpoints(left_point, left_direction)
	right_start, right_end = line_endpoints(right_point, right_direction)
	# 兩條側邊各自取中點，再取兩個中點的中點，作為梯形中心。
	left_middle = (left_start + left_end) / 2.0
	right_middle = (right_start + right_end) / 2.0
	side_center = (left_middle + right_middle) / 2.0
	# 目前側邊只取物體中間 50%，乘以 2 估計整個縱向長度。
	long_side = (
		np.linalg.norm(left_end - left_start)
		+ np.linalg.norm(right_end - right_start)
	) / 2.0 * 2.0
	left_angle = float(np.degrees(np.arctan2(left_direction[1], left_direction[0])))
	right_angle = float(np.degrees(np.arctan2(right_direction[1], right_direction[0])))
	summed_angle = float(
		np.degrees(np.arctan2(summed_direction[1], summed_direction[0]))
	)

	return {
		"left_points": np.asarray(left_points),
		"right_points": np.asarray(right_points),
		"left_start": left_start,
		"left_end": left_end,
		"right_start": right_start,
		"right_end": right_end,
		"center": side_center,
		"long_side": float(long_side),
		"left_direction": left_direction,
		"right_direction": right_direction,
		"direction": summed_direction,
		"left_angle": left_angle,
		"right_angle": right_angle,
		"side_angle_difference": axis_angle_difference(left_angle, right_angle),
		"left_inlier_ratio": left_fit["inlier_ratio"],
		"right_inlier_ratio": right_fit["inlier_ratio"],
		"angle": summed_angle,
	}


def analyze_target(target_mask, min_axis_ratio=DEFAULT_MIN_AXIS_RATIO):
	"""估計目標中心、實際主軸角度、方向向量與可信度。"""
	# 透視會讓稻草捆的矩形變成梯形，因此用所有前景像素做 PCA，
	# 取得整個目標的主要分布方向，作為實際長軸方向。
	points_yx = np.column_stack(np.where(target_mask > 0))
	points_xy = points_yx[:, ::-1].astype(np.float32)
	mean, eigenvectors, eigenvalues = cv2.PCACompute2(
		points_xy, mean=None
	)
	pca_vector = eigenvectors[0]
	pca_angle = float(np.degrees(np.arctan2(pca_vector[1], pca_vector[0])))

	# 目標被畫面邊界截斷時，mask 會接近正方形，主軸方向純粹是雜訊。
	# 形狀不夠細長就不輸出方向，避免給控制端一個看似合理卻完全錯誤的角度。
	variances = eigenvalues.ravel()
	minor_variance = float(variances[1])
	if minor_variance <= 1e-6:
		axis_ratio = float("inf")
	else:
		axis_ratio = float(variances[0]) / minor_variance
	if axis_ratio < min_axis_ratio:
		return None

	# PCA 只用來提供找左右側邊的初始座標系；實際方向由兩側邊向量和決定。
	side_edges = fit_side_edges(target_mask, pca_vector)
	if side_edges is None:
		# 側邊是中心與方向的必要資料，失敗時不能用外接矩形中心冒充。
		return None

	direction = side_edges["direction"]
	actual_angle = side_edges["angle"]
	target_center = tuple(side_edges["center"])

	# 左右邊越平行、RANSAC 內點越多，代表這次方向越可信。
	side_confidence = max(
		0.0, 1.0 - side_edges["side_angle_difference"] / 30.0
	)
	inlier_confidence = (
		side_edges["left_inlier_ratio"]
		+ side_edges["right_inlier_ratio"]
	) / 2.0
	confidence = float(side_confidence * inlier_confidence)

	return {
		"center": target_center,
		"long_side": side_edges["long_side"],
		"angle": float(actual_angle),
		"pca_angle": pca_angle,
		"axis_ratio": axis_ratio,
		"confidence": float(confidence),
		"direction": direction,
		"side_edges": side_edges,
		"eigenvalues": eigenvalues,
		"mean": mean,
	}


def calculate_heading_error(target_angle, robot_angle=0.0):
	"""回傳 [-90, 90) 度內的無方向軸線角度誤差。"""
	return (target_angle - robot_angle + 90.0) % 180.0 - 90.0


def make_visualization(mask, target_mask, result, robot_angle=0.0):
	"""繪製輪廓、左右側邊，以及兩側邊向量和的實際方向。"""
	visualization = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	visualization[target_mask > 0] = (0, 180, 255)

	# 紅色線是從 mask 左右邊界擬合出的兩條側邊。
	side_edges = result["side_edges"]
	if side_edges is not None:
		for start_name, end_name in (
			("left_start", "left_end"),
			("right_start", "right_end"),
		):
			cv2.line(
				visualization,
				tuple(np.round(side_edges[start_name]).astype(int)),
				tuple(np.round(side_edges[end_name]).astype(int)),
				(0, 0, 255),
				6,
			)

	center_x, center_y = result["center"]
	center = (int(round(center_x)), int(round(center_y)))
	cv2.circle(visualization, center, 8, (0, 0, 255), -1)

	# 灰色雙向箭頭是左右側邊向量相加後的圓柱體實際朝向。
	direction = result["direction"]
	arrow_length = max(result["long_side"] * 0.5, 40.0)
	start = np.array(center, dtype=np.float32) - direction * arrow_length
	end = np.array(center, dtype=np.float32) + direction * arrow_length
	cv2.line(
		visualization,
		tuple(np.round(start).astype(int)),
		tuple(np.round(end).astype(int)),
		(180, 180, 180),
		4,
	)
	# 兩端都加上箭頭，表示這是沒有正反方向差異的軸線。
	for point, vector in ((start, direction), (end, -direction)):
		arrow_tip = point + vector * 28.0
		cv2.arrowedLine(
			visualization,
			tuple(np.round(point).astype(int)),
			tuple(np.round(arrow_tip).astype(int)),
			(180, 180, 180),
			4,
			tipLength=0.35,
		)

	heading_error = calculate_heading_error(result["angle"], robot_angle)
	text_lines = [
		f"兩側邊向量和: {result['angle']:.1f} deg",
		f"左/右側邊: {side_edges['left_angle']:.1f} / "
		f"{side_edges['right_angle']:.1f} deg",
		f"左右角度差: {side_edges['side_angle_difference']:.1f} deg",
		f"PCA 初始參考: {result['pca_angle']:.1f} deg",
		f"機器人角度誤差: {heading_error:.1f} deg",
		f"方向可信度: {result['confidence']:.2f}",
	]
	for index, text in enumerate(text_lines):
		cv2.putText(
			visualization,
			text,
			(20, 35 + index * 32),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.8,
			(255, 255, 255),
			2,
			cv2.LINE_AA,
		)

	return visualization, heading_error


def process_frame(frame_bgr, args, angle_filter):
	"""對單一影格執行 mask 建立、目標選取與方向分析。"""
	lower_bound = np.array(args.lower_hsv, dtype=np.uint8)
	upper_bound = np.array(args.upper_hsv, dtype=np.uint8)
	mask = create_color_mask(frame_bgr, lower_bound, upper_bound, args.kernel_size)
	cleaned_mask = clean_mask(mask, 1)

	component = select_largest_component(cleaned_mask, args.min_area)
	if component is None:
		return None

	result = analyze_target(component["mask"], args.min_axis_ratio)
	if result is None:
		return None

	result = apply_angle_filter(result, angle_filter)
	visualization, heading_error = make_visualization(
		cleaned_mask, component["mask"], result, args.robot_angle
	)
	return {
		"visualization": visualization,
		"result": result,
		"heading_error": heading_error,
		"component": component,
	}


def run_on_stream(capture, args):
	"""逐格讀取影片或相機畫面，即時偵測並可選擇顯示/儲存結果。"""
	angle_filter = AxisAngleFilter(alpha=0.25)
	writer = None
	window_name = "TDK Straw Detection"
	output_path = None if args.no_save or not args.output else Path(args.output)

	try:
		while True:
			ok, frame = capture.read()
			if not ok:
				break

			outcome = process_frame(frame, args, angle_filter)
			if outcome is None:
				display = frame
			else:
				result = outcome["result"]
				print(
					f"目標中心: ({result['center'][0]:.1f}, {result['center'][1]:.1f})  "
					f"角度: {result['angle']:.1f} deg  "
					f"誤差: {outcome['heading_error']:.1f} deg  "
					f"可信度: {result['confidence']:.2f}"
				)
				display = outcome["visualization"]

			if output_path is not None:
				if writer is None:
					output_path.parent.mkdir(parents=True, exist_ok=True)
					fps = capture.get(cv2.CAP_PROP_FPS)
					if not fps or fps <= 1e-2:
						fps = 20.0
					height, width = display.shape[:2]
					fourcc = cv2.VideoWriter_fourcc(*"mp4v")
					writer = cv2.VideoWriter(
						str(output_path), fourcc, fps, (width, height)
					)
				writer.write(display)

			if not args.no_display:
				cv2.imshow(window_name, display)
				key = cv2.waitKey(1) & 0xFF
				if key in (27, ord("q")):
					break
	finally:
		capture.release()
		if writer is not None:
			writer.release()
		if not args.no_display:
			cv2.destroyAllWindows()

	if output_path is not None:
		print(f"輸出影片已儲存: {output_path}")


def build_input_mask(args):
	"""依照參數從原始影像或外部檔案取得 mask。"""
	if args.mask is not None:
		return load_mask(args.mask)

	image = load_image(args.image)
	lower_bound = np.array(args.lower_hsv, dtype=np.uint8)
	upper_bound = np.array(args.upper_hsv, dtype=np.uint8)
	return create_color_mask(
		image, lower_bound, upper_bound, args.kernel_size
	)


def apply_angle_filter(result, angle_filter):
	"""套用跨影格軸線濾波，並更新繪圖與控制使用的方向。"""
	filtered_angle = angle_filter.update(result["angle"])
	result["filtered_angle"] = filtered_angle
	result["angle"] = filtered_angle
	filtered_radians = np.deg2rad(filtered_angle)
	result["direction"] = np.array(
		[np.cos(filtered_radians), np.sin(filtered_radians)],
		dtype=np.float32,
	)
	return result


def save_outputs(
	output_path, visualization, mask, target_mask, save_masks, save_full_mask
):
	"""儲存標註圖；除非明確要求，否則不額外輸出 mask。"""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(output_path), visualization)
	if not save_masks:
		return

	if save_full_mask:
		cv2.imwrite(str(output_path.with_name("straw_hsv_mask.png")), mask)
	cv2.imwrite(
		str(output_path.with_name("straw_target_mask.png")), target_mask
	)


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--image",
		default="IMG_3231.JPEG",
		help="原始 BGR 圖片路徑，預設使用 c1.py 的測試圖片",
	)
	parser.add_argument(
		"--mask",
		default=None,
		help="可選的外部二值 mask；指定後會略過 HSV mask 建立（僅適用於單張圖片）",
	)
	parser.add_argument(
		"--video",
		default=None,
		help="影片檔案路徑；提供後以影片作為輸入來源（優先於 --image）",
	)
	parser.add_argument(
		"--camera",
		type=int,
		nargs="?",
		const=0,
		default=None,
		help="相機裝置編號（預設 0）；提供後以相機作為輸入來源（優先於 --video/--image）",
	)
	parser.add_argument(
		"--no-display",
		action="store_true",
		help="不開啟即時顯示視窗，適合無頭環境",
	)
	parser.add_argument(
		"--save-masks",
		action="store_true",
		help="額外儲存目標 mask 與完整 HSV mask，供離線除錯使用",
	)
	parser.add_argument(
		"--no-save",
		action="store_true",
		help="不儲存影片/相機模式的輸出結果",
	)
	parser.add_argument(
		"--output",
		default=DEFAULT_IMAGE_OUTPUT,
		help="標註輸出的路徑；影片/相機模式預設輸出 mp4",
	)
	parser.add_argument(
		"--min-area",
		type=int,
		default=DEFAULT_MIN_AREA,
		help="有效目標連通區的最小面積",
	)
	parser.add_argument(
		"--min-axis-ratio",
		type=float,
		default=DEFAULT_MIN_AXIS_RATIO,
		help="PCA 主軸與次軸的最小變異比；低於此值視為形狀太接近方形，不輸出方向",
	)
	parser.add_argument(
		"--kernel-size",
		type=int,
		default=DEFAULT_MORPHOLOGY_KERNEL,
		help="HSV mask 的形態學核心大小，與 c1.py 預設值一致",
	)
	parser.add_argument(
		"--lower-hsv",
		type=int,
		nargs=3,
		default=DEFAULT_LOWER_HSV,
		metavar=("H", "S", "V"),
		help="HSV 下界，與 c1.py 預設值一致",
	)
	parser.add_argument(
		"--upper-hsv",
		type=int,
		nargs=3,
		default=DEFAULT_UPPER_HSV,
		metavar=("H", "S", "V"),
		help="HSV 上界，與 c1.py 預設值一致",
	)
	parser.add_argument(
		"--robot-angle",
		type=float,
		default=0.0,
		help="機器人前進方向在影像座標中的角度",
	)
	return parser.parse_args()


def main():
	# 讀取參數，依序完成 mask 清理、目標選取、方向分析與結果輸出。
	args = parse_args()

	if args.camera is not None or args.video is not None:
		source = args.video if args.camera is None else args.camera
		capture = cv2.VideoCapture(source)
		if not capture.isOpened():
			raise RuntimeError(f"無法開啟輸入來源: {source}")

		if args.output == DEFAULT_IMAGE_OUTPUT:
			args.output = DEFAULT_VIDEO_OUTPUT

		run_on_stream(capture, args)
		return

	mask = build_input_mask(args)

	cleaned_mask = clean_mask(mask, 1)
	component = select_largest_component(cleaned_mask, args.min_area)

	if component is None:
		raise RuntimeError(
			"No valid target found. Check the mask or lower --min-area."
		)

	result = analyze_target(component["mask"], args.min_axis_ratio)
	if result is None:
		raise RuntimeError(
			"目標形狀不夠細長（主軸比 < "
			f"{args.min_axis_ratio}，可用 --min-axis-ratio 調整）"
			"或無法可靠擬合左右側邊，本幀不輸出中心與方向。"
		)

	# 相機連續取像時，應在影像迴圈外建立並重複使用同一個 filter。
	angle_filter = AxisAngleFilter(alpha=0.25)
	result = apply_angle_filter(result, angle_filter)

	visualization, heading_error = make_visualization(
		cleaned_mask,
		component["mask"],
		result,
		args.robot_angle,
	)

	save_outputs(
		args.output,
		visualization,
		mask,
		component["mask"],
		args.save_masks,
		args.mask is None,
	)

	print(f"目標面積: {component['area']} px")
	print(f"目標中心: ({result['center'][0]:.1f}, {result['center'][1]:.1f})")
	print(f"兩側邊估計長度: {result['long_side']:.1f} px")
	print(f"兩側邊向量和角度: {result['angle']:.1f} deg")
	print(f"機器人角度誤差: {heading_error:.1f} deg")
	print(f"主軸細長比: {result['axis_ratio']:.2f}")
	print(f"方向可信度: {result['confidence']:.2f}")
	print(f"標註圖已儲存: {args.output}")


if __name__ == "__main__":
	main()
