"""從二值化 mask 偵測最大的稻草捆區域。

此版本先用離線 mask 驗證幾何分析，偵測邏輯不依賴相機或機器人控制器。
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# 與 c1.py 相同的預設 HSV 範圍。
# DEFAULT_LOWER_HSV = (21, 65, 67)
# DEFAULT_UPPER_HSV = (33, 255, 255)
DEFAULT_LOWER_HSV = (16, 84, 42)     # 原本 (21, 65, 67)
DEFAULT_UPPER_HSV = (33, 255, 255)   # 原本 (33, 255, 255)
DEFAULT_MIN_AREA = 5000
DEFAULT_MORPHOLOGY_KERNEL = 11
DEFAULT_RANSAC_THRESHOLD = 12.0
DEFAULT_RANSAC_ITERATIONS = 300
DEFAULT_BORDER_MARGIN = 2
# 影像座標中畫面的縱向為 90 度。機器人對準稻草捆長軸時，
# 目標軸線應與畫面縱向重合，此時角度誤差為 0。
DEFAULT_ROBOT_ANGLE = 90.0
# RealSense 深度影像為 16UC1，單位公釐。
DEFAULT_DEPTH_SCALE = 0.001
DEFAULT_DEPTH_PATCH_RADIUS = 6
# 可信度三項證據的模糊區間端點，依實測值設定（見 calculate_confidence）。
DEFAULT_STRAIGHT_FLOOR = 0.5
DEFAULT_STRAIGHT_TARGET = 0.85
DEFAULT_ANGLE_SIGMA_LIMIT = 2.0
DEFAULT_SEED_MARGIN_TARGET = 0.25
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

	# 所有取樣配對一次算完。候選數和點數都只有數十，(候選 x 點) 的距離
	# 矩陣很小，用矩陣運算取代逐次迴圈可避開 numpy 的單次呼叫開銷。
	rng = np.random.default_rng(42)
	count = len(points)
	first_index = rng.integers(0, count, size=iterations)
	# 加上 1..count-1 的位移再取模，保證配對的兩點不同。
	second_index = (
		first_index + rng.integers(1, count, size=iterations)
	) % count

	first_points = points[first_index]
	line_vectors = points[second_index] - first_points
	lengths = np.linalg.norm(line_vectors, axis=1)
	usable = lengths >= 1.0
	if not np.any(usable):
		return None

	first_points = first_points[usable]
	line_vectors = line_vectors[usable] / lengths[usable, None]

	# 每個候選線到每個點的垂直距離，形狀為 (候選數, 點數)。
	deltas = points[None, :, :] - first_points[:, None, :]
	distances = np.abs(
		line_vectors[:, None, 0] * deltas[:, :, 1]
		- line_vectors[:, None, 1] * deltas[:, :, 0]
	)
	inlier_masks = distances <= threshold
	inlier_counts = inlier_masks.sum(axis=1)
	inlier_sums = np.where(inlier_masks, distances, 0.0).sum(axis=1)
	errors = np.where(
		inlier_counts > 0,
		inlier_sums / np.maximum(inlier_counts, 1),
		np.inf,
	)

	# 先比內點數（多者優先），內點數相同再比平均殘差（小者優先）。
	best = int(np.lexsort((errors, -inlier_counts))[0])
	best_inliers = inlier_masks[best]
	best_error = float(errors[best])
	if np.count_nonzero(best_inliers) < 5:
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
		"inlier_mask": best_inliers,
		"inlier_ratio": float(np.mean(best_inliers)),
		"error": best_error,
	}


def fit_straight_segment(
	points,
	seed_direction,
	threshold=DEFAULT_RANSAC_THRESHOLD,
	iterations=DEFAULT_RANSAC_ITERATIONS,
):
	"""從依縱向排序的邊界點中，找出圓柱體真正的直線側邊。

	圓柱體兩端是圓弧，它們位於序列的頭尾。先用 RANSAC 找出符合同一
	條直線的點，再取其中最長的一段連續內點重新擬合：圓弧會持續偏離
	直線，不可能落在連續內點區段裡，因此自然被截掉。

	取「連續」而不是只取內點，是因為兩端圓弧偶爾會正好擦過直線，
	造成兩段直邊中間夾著圓弧的假直線。
	"""
	points = np.asarray(points, dtype=np.float32)
	initial_fit = fit_line_ransac(points, seed_direction, threshold, iterations)
	if initial_fit is None:
		return None

	# 搜尋最長的連續內點區段；末尾補一個 False 讓最後一段也能收尾。
	best_start = 0
	best_length = 0
	run_start = None
	for index, is_inlier in enumerate(np.append(initial_fit["inlier_mask"], False)):
		if is_inlier:
			if run_start is None:
				run_start = index
		elif run_start is not None:
			if index - run_start > best_length:
				best_start = run_start
				best_length = index - run_start
			run_start = None

	if best_length < 5:
		return None

	segment = points[best_start:best_start + best_length]
	segment_fit = fit_line_ransac(segment, seed_direction, threshold, iterations)
	if segment_fit is None:
		return None

	# inlier_ratio 保留整體擬合的數值，才能反映這個種子方向好不好；
	# 重新擬合後的內點率幾乎恆為 1，沒有區別力。
	segment_fit["inlier_ratio"] = initial_fit["inlier_ratio"]
	segment_fit["segment"] = segment
	segment_fit["straight_ratio"] = best_length / float(len(points))
	return segment_fit


def extract_contour_points(target_mask, border_margin=DEFAULT_BORDER_MARGIN):
	"""取出目標輪廓，並剔除貼齊畫面邊界的點。

	結果只取決於 mask，與種子方向無關，因此每幀只需計算一次，
	供兩個種子方向共用。

	回傳 (輪廓點, 貼邊比例)；輪廓不存在或點數不足時回傳 None。
	"""
	contours, _ = cv2.findContours(
		target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
	)
	if not contours:
		return None

	contour = max(contours, key=cv2.contourArea)
	contour_points = contour[:, 0, :].astype(np.float32)

	# 目標超出畫面時，輪廓會沿著影像邊界走一整段。
	# 那是裁切痕跡，不是稻草捆的真實邊緣，必須先剔除，
	# 否則會被當成側邊候選點，讓 RANSAC 擬出一條沿著畫面邊緣的假側邊。
	height, width = target_mask.shape[:2]
	on_border = (
		(contour_points[:, 0] <= border_margin)
		| (contour_points[:, 0] >= width - 1 - border_margin)
		| (contour_points[:, 1] <= border_margin)
		| (contour_points[:, 1] >= height - 1 - border_margin)
	)
	border_ratio = float(np.mean(on_border))
	contour_points = contour_points[~on_border]
	if len(contour_points) < 20:
		return None

	return contour_points, border_ratio


def fit_side_edges(contour_points, border_ratio, seed_direction):
	"""從輪廓的左右邊界擬合兩條側邊，並將向量相加。

	seed_direction 只用來建立物體的初始縱向座標，最後方向由左右側邊
	的向量和決定。這可以降低透視造成單一側邊偏移的影響。
	"""
	center = contour_points.mean(axis=0)
	axis = np.asarray(seed_direction, dtype=np.float32)
	axis /= np.linalg.norm(axis)
	perpendicular = np.array([-axis[1], axis[0]], dtype=np.float32)

	# 將輪廓點投影到縱向和橫向座標，涵蓋整個縱向範圍。
	# 不再用固定的中間 50%：那是依點數分位而非座標範圍，而圓弧上的
	# 輪廓點比直邊密集，分位區間會被拉往圓弧那一側。圓弧的排除
	# 改由 fit_straight_segment 負責。
	longitudinal = (contour_points - center) @ axis
	lateral = (contour_points - center) @ perpendicular
	long_min = float(longitudinal.min())
	long_max = float(longitudinal.max())
	if long_max - long_min < 1.0:
		return None

	# 每個縱向切片取邊界附近多個點的中位數，不讓單一毛刺決定邊界。
	bin_edges = np.linspace(long_min, long_max, 41)
	left_points = []
	right_points = []
	for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
		in_bin = (longitudinal >= lower) & (longitudinal <= upper)
		if not np.any(in_bin):
			continue
		bin_points = contour_points[in_bin]
		bin_lateral = lateral[in_bin]
		boundary_count = max(3, int(np.ceil(len(bin_points) * 0.08)))
		left_boundary = bin_points[np.argsort(bin_lateral)[:boundary_count]]
		right_boundary = bin_points[np.argsort(bin_lateral)[-boundary_count:]]
		left_points.append(np.median(left_boundary, axis=0))
		right_points.append(np.median(right_boundary, axis=0))

	if len(left_points) < 5 or len(right_points) < 5:
		return None

	left_fit = fit_straight_segment(left_points, axis)
	right_fit = fit_straight_segment(right_points, axis)
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

	def line_endpoints(fit):
		"""將擬合線截在這條側邊實際的直線段範圍內。"""
		projections = (fit["segment"] - center) @ axis
		point = fit["point"]
		direction = fit["direction"]
		offset = np.dot(point - center, direction)
		start = point + direction * (float(projections.min()) - offset)
		end = point + direction * (float(projections.max()) - offset)
		return start, end

	left_start, left_end = line_endpoints(left_fit)
	right_start, right_end = line_endpoints(right_fit)
	# 兩條側邊各自取中點，再取兩個中點的中點，作為梯形中心。
	left_middle = (left_start + left_end) / 2.0
	right_middle = (right_start + right_end) / 2.0
	side_center = (left_middle + right_middle) / 2.0
	# 側邊已是實際量到的直線段，直接取兩邊平均長度，不再估算全長。
	long_side = (
		np.linalg.norm(left_end - left_start)
		+ np.linalg.norm(right_end - right_start)
	) / 2.0
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
		"border_ratio": border_ratio,
		"straight_ratio": (
			left_fit["straight_ratio"] + right_fit["straight_ratio"]
		) / 2.0,
		"left_straight_ratio": left_fit["straight_ratio"],
		"right_straight_ratio": right_fit["straight_ratio"],
		"left_error": left_fit["error"],
		"right_error": right_fit["error"],
		"left_sample_count": int(len(left_fit["segment"])),
		"right_sample_count": int(len(right_fit["segment"])),
		"left_length": float(np.linalg.norm(left_end - left_start)),
		"right_length": float(np.linalg.norm(right_end - right_start)),
		"left_inlier_ratio": left_fit["inlier_ratio"],
		"right_inlier_ratio": right_fit["inlier_ratio"],
		"angle": summed_angle,
	}


def side_straightness(side_edges):
	"""兩條側邊中較差的那條的直線段佔比。

	取最小而非平均：一邊擬得完美、另一邊是垃圾，不應該被平均成看起來
	可以接受的分數。
	"""
	return min(
		side_edges["left_straight_ratio"], side_edges["right_straight_ratio"]
	)


def estimate_angle_sigma(side_edges):
	"""估計最終角度的標準誤，單位為度。

	直線擬合的斜率標準誤約為 殘差 * sqrt(12) / (線段長 * sqrt(點數))：線段
	越長、取樣點越多、點越貼合直線，方向就釘得越緊。左右兩條獨立擬合後
	取平均，誤差再除以 sqrt(2)。
	"""
	sigmas = []
	for side in ("left", "right"):
		length = max(side_edges["%s_length" % side], 1.0)
		count = max(side_edges["%s_sample_count" % side], 2)
		sigmas.append(
			np.degrees(
				side_edges["%s_error" % side]
				* np.sqrt(12.0)
				/ (length * np.sqrt(count))
			)
		)
	return float(np.mean(sigmas) / np.sqrt(2.0))


def calculate_confidence(side_edges, seed_margin):
	"""以三項獨立證據評估方向可信度，取最弱的一項。

	證據：兩側都找到夠長的直線段，代表擬的是真正的側邊而不是圓弧。
	精度：角度本身的標準誤，回答「這個角度釘得多緊」。
	種子：主軸與次軸兩個候選的分數差距，守住 90 度翻轉這個最壞的失效
	模式；目標若是正矩形，兩個方向都有直邊，前兩項都不會示警。

	取最小值而非相乘：相乘會讓三項都尚可的結果被壓到 0.6 以下，失去可
	讀性；取最小值則能直接看出是哪一項在拖。

	刻意不使用左右側邊夾角：稻草捆的兩側邊在透視下本來就會收斂，實測
	正確偵測的夾角反而比錯誤的大（11~20 度 vs 2~11 度），拿它當懲罰項
	會壓低正確結果的分數。
	"""
	evidence = (side_straightness(side_edges) - DEFAULT_STRAIGHT_FLOOR) / (
		DEFAULT_STRAIGHT_TARGET - DEFAULT_STRAIGHT_FLOOR
	)
	precision = (
		1.0 - estimate_angle_sigma(side_edges) / DEFAULT_ANGLE_SIGMA_LIMIT
	)
	margin = seed_margin / DEFAULT_SEED_MARGIN_TARGET
	terms = {
		"evidence": float(np.clip(evidence, 0.0, 1.0)),
		"precision": float(np.clip(precision, 0.0, 1.0)),
		"seed_margin": float(np.clip(margin, 0.0, 1.0)),
	}
	return float(min(terms.values())), terms


def analyze_target(target_mask):
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

	# 主軸與次軸的變異比，接近 1 代表形狀接近方形。純粹是診斷資訊：
	# 近似方形造成的 90 度翻轉改由雙種子擬合處理，可信度的種子項會反映。
	variances = eigenvalues.ravel()
	minor_variance = float(variances[1])
	if minor_variance <= 1e-6:
		axis_ratio = float("inf")
	else:
		axis_ratio = float(variances[0]) / minor_variance

	# PCA 只用來提供找左右側邊的初始座標系；實際方向由兩側邊向量和決定。
	# 目標接近方形時 PCA 分不出長短軸，主軸可能剛好指向側邊的垂直
	# 方向，側邊就會汿成上下兩端。主軸與次軸各擬合一次，取內點率高者：
	# 內點率能分辨真實側邊與非側邊，左右夾角則不行（透視收斂會讓正確
	# 解的夾角反而較大）。
	perpendicular_vector = np.array(
		[-pca_vector[1], pca_vector[0]], dtype=np.float32
	)
	extracted = extract_contour_points(target_mask)
	if extracted is None:
		return None
	contour_points, border_ratio = extracted

	candidates = []
	for seed_vector in (pca_vector, perpendicular_vector):
		fitted = fit_side_edges(contour_points, border_ratio, seed_vector)
		if fitted is not None:
			candidates.append((side_straightness(fitted), fitted))
	if not candidates:
		# 側邊是中心與方向的必要資料，失敗時不能用外接矩形中心冒充。
		return None

	candidates.sort(key=lambda item: item[0], reverse=True)
	best_score, side_edges = candidates[0]
	runner_up_score = candidates[1][0] if len(candidates) > 1 else 0.0
	seed_margin = (
		(best_score - runner_up_score) / best_score
		if best_score > 1e-6
		else 0.0
	)

	direction = side_edges["direction"]
	actual_angle = side_edges["angle"]
	target_center = tuple(side_edges["center"])

	confidence, confidence_terms = calculate_confidence(
		side_edges, seed_margin
	)

	return {
		"center": target_center,
		"long_side": side_edges["long_side"],
		"angle": float(actual_angle),
		"pca_angle": pca_angle,
		"axis_ratio": axis_ratio,
		"border_ratio": side_edges["border_ratio"],
		"straight_ratio": side_edges["straight_ratio"],
		"confidence": float(confidence),
		"confidence_terms": confidence_terms,
		"angle_sigma": estimate_angle_sigma(side_edges),
		"seed_margin": float(seed_margin),
		"direction": direction,
		"side_edges": side_edges,
		"eigenvalues": eigenvalues,
		"mean": mean,
	}


def calculate_heading_error(target_angle, robot_angle=DEFAULT_ROBOT_ANGLE):
	"""回傳 [-90, 90) 度內的無方向軸線角度誤差。"""
	return (target_angle - robot_angle + 90.0) % 180.0 - 90.0


def sample_depth_patch(
	depth_image,
	center_px,
	radius=DEFAULT_DEPTH_PATCH_RADIUS,
	depth_scale=DEFAULT_DEPTH_SCALE,
):
	"""取中心鄰域的深度中位數，單位公尺；沒有有效值時回傳 None。

	單一像素的深度常常是 0：反光、物體邊緣、超出量程都會造成破洞。
	取鄰域並剔除 0 之後再取中位數，比直接讀一個像素穩定得多。
	"""
	height, width = depth_image.shape[:2]
	column = int(round(float(center_px[0])))
	row = int(round(float(center_px[1])))
	left = max(0, column - radius)
	right = min(width, column + radius + 1)
	top = max(0, row - radius)
	bottom = min(height, row + radius + 1)
	if left >= right or top >= bottom:
		return None

	patch = depth_image[top:bottom, left:right].astype(np.float32)
	valid = patch[patch > 0.0]
	if valid.size == 0:
		return None

	return float(np.median(valid)) * depth_scale


def build_depth_fields(
	center_px,
	depth_image,
	intrinsics,
	radius=DEFAULT_DEPTH_PATCH_RADIUS,
	depth_scale=DEFAULT_DEPTH_SCALE,
):
	"""把目標中心像素還原成相機座標系的公尺座標。

	intrinsics 為 (fx, fy, cx, cy)。回傳的欄位一律存在，has_depth 說明
	這次有沒有取到有效深度，控制端不必用「欄位在不在」來判斷。
	"""
	missing = {"has_depth": False}
	if depth_image is None or intrinsics is None:
		return missing

	depth_metres = sample_depth_patch(
		depth_image, center_px, radius, depth_scale
	)
	if depth_metres is None:
		return missing

	fx, fy, cx, cy = intrinsics
	# 針孔模型反投影。x 向右、y 向下、z 向前，單位公尺。
	x = (float(center_px[0]) - cx) * depth_metres / fx
	y = (float(center_px[1]) - cy) * depth_metres / fy
	return {
		"has_depth": True,
		"distance_m": round(depth_metres, 4),
		"lateral_error_m": round(x, 4),
		"position_m": [round(x, 4), round(y, 4), round(depth_metres, 4)],
	}


def calculate_control_errors(result, image_width, robot_angle=DEFAULT_ROBOT_ANGLE):
	"""計算要回傳給機器人的兩個控制量。

	角度誤差：目標長軸與機器人前進方向的夾角，0 代表已對正。
	正值代表目標頂端偏向畫面右側。

	橫向誤差：目標中心相對畫面中線的水平位移，0 代表已對中。
	正值代表目標位於中線右側。相機裝在機器人中線上，畫面中線即機器人中線。

	橫向誤差同時提供像素值與正規化值。正規化值以半個畫面寬為單位，
	範圍約 [-1, 1]，不受解析度影響，控制端不必知道相機規格即可使用。
	像素值要換算成實際距離則需要相機標定與目標距離，本程式不提供。
	"""
	heading_error = calculate_heading_error(result["angle"], robot_angle)
	half_width = image_width / 2.0
	lateral_error = float(result["center"][0]) - half_width
	return {
		"heading_error": float(heading_error),
		"lateral_error": lateral_error,
		"lateral_error_ratio": lateral_error / half_width,
	}


def build_robot_payload(result, errors, image_shape):
	"""組成傳給機器人中心電腦的一筆資料。

	valid 為 False 時代表本影格沒有可用的偵測，控制端應維持前一個指令
	或停止，不可把缺值當成 0 誤差。
	"""
	height, width = image_shape[:2]
	return {
		"valid": True,
		"heading_error_deg": round(errors["heading_error"], 2),
		"lateral_error_px": round(errors["lateral_error"], 1),
		"lateral_error_ratio": round(errors["lateral_error_ratio"], 4),
		"angle_deg": round(float(result["angle"]), 2),
		"center_px": [
			round(float(result["center"][0]), 1),
			round(float(result["center"][1]), 1),
		],
		"confidence": round(float(result["confidence"]), 3),
		"angle_sigma_deg": round(float(result["angle_sigma"]), 3),
		"image_size": [int(width), int(height)],
	}


def make_visualization(mask, target_mask, result, errors):
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

	# 綠色垂直線是機器人中線，橫線標出目標中心離中線多遠。
	middle_x = visualization.shape[1] // 2
	cv2.line(
		visualization,
		(middle_x, 0),
		(middle_x, visualization.shape[0]),
		(0, 255, 0),
		2,
	)
	cv2.line(
		visualization,
		(middle_x, center[1]),
		(center[0], center[1]),
		(0, 255, 0),
		4,
	)

	text_lines = [
		f"兩側邊向量和: {result['angle']:.1f} deg",
		f"左/右側邊: {side_edges['left_angle']:.1f} / "
		f"{side_edges['right_angle']:.1f} deg",
		f"左右角度差: {side_edges['side_angle_difference']:.1f} deg",
		f"PCA 初始參考: {result['pca_angle']:.1f} deg",
		f"機器人角度誤差: {errors['heading_error']:.1f} deg",
		f"橫向誤差: {errors['lateral_error']:.0f} px "
		f"({errors['lateral_error_ratio']:+.2f})",
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

	return visualization


def process_frame(frame_bgr, args, angle_filter, build_visualization=True):
	"""對單一影格執行 mask 建立、目標選取與方向分析。

	build_visualization 為 False 時不繪製標註圖，visualization 會是 None。
	機器運行模式用不到標註圖，而繪圖佔整體約四分之一的時間。
	"""
	lower_bound = np.array(args.lower_hsv, dtype=np.uint8)
	upper_bound = np.array(args.upper_hsv, dtype=np.uint8)
	mask = create_color_mask(frame_bgr, lower_bound, upper_bound, args.kernel_size)
	cleaned_mask = clean_mask(mask, 1)

	component = select_largest_component(cleaned_mask, args.min_area)
	if component is None:
		return None

	result = analyze_target(component["mask"])
	if result is None:
		return None

	result = apply_angle_filter(result, angle_filter)
	errors = calculate_control_errors(
		result, frame_bgr.shape[1], args.robot_angle
	)
	visualization = None
	if build_visualization:
		visualization = make_visualization(
			cleaned_mask, component["mask"], result, errors
		)
	return {
		"visualization": visualization,
		"result": result,
		"errors": errors,
		"payload": build_robot_payload(result, errors, frame_bgr.shape),
		"component": component,
	}


def build_robot_command(payload):
	"""機器運行模式的輸出：只留控制迴圈真正會用到的量。

	欄位固定不變，控制端不必判斷欄位在不在。沒有深度時 lateral_error_m
	為 null，此時改用 lateral_error_ratio（以半個畫面寬為單位）。
	"""
	if not payload.get("valid"):
		return {
			"valid": False,
			"heading_error_deg": None,
			"lateral_error_m": None,
			"lateral_error_ratio": None,
		}

	return {
		"valid": True,
		"heading_error_deg": payload["heading_error_deg"],
		"lateral_error_m": (
			payload["lateral_error_m"] if payload.get("has_depth") else None
		),
		"lateral_error_ratio": payload["lateral_error_ratio"],
	}


def emit_payload(payload):
	"""輸出一筆 JSON 給機器人中心電腦。

	一行一筆（JSON Lines），並立即 flush：控制端多半是逐行讀取，
	留在緩衝區裡的資料對即時控制沒有意義。
	"""
	print(json.dumps(payload, ensure_ascii=False), flush=True)


class RealSenseCapture:
	"""把 pyrealsense2 的串流包成 cv2.VideoCapture 的介面。

	OpenCV 的 UVC 路徑抓不到 RealSense 的彩色串流：裝置會註冊多個 UVC
	節點，MSMF 能開啟卻取不到影格（錯誤 0xC00D3704）。改用官方 SDK 才
	可靠，順帶也拿得到深度與內參。

	介面刻意與 VideoCapture 相同，run_on_stream 因此不必區分輸入來源。
	"""

	def __init__(self, width, height, fps, use_depth=True, timeout_ms=10000):
		import pyrealsense2 as rs

		self.rs = rs
		self.fps = fps
		self.use_depth = use_depth
		self.timeout_ms = timeout_ms
		self.depth_image = None
		self.intrinsics = None

		self.pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(
			rs.stream.color, width, height, rs.format.bgr8, fps
		)
		if use_depth:
			config.enable_stream(
				rs.stream.depth, width, height, rs.format.z16, fps
			)
		self.profile = self.pipeline.start(config)
		# 深度對齊到彩色，兩者才會共用同一組內參與同一個像素座標。
		self.aligner = rs.align(rs.stream.color) if use_depth else None

	def read(self):
		"""回傳 (是否成功, BGR 影像)，並記下同一格的深度。"""
		try:
			frames = self.pipeline.wait_for_frames(self.timeout_ms)
		except RuntimeError as error:
			# 最常見的原因是相機仍被別的程式佔用：realsense_test.py 還開著，
			# 或先前用 OpenCV 開過同一個 UVC 節點而未完全釋放。
			raise RuntimeError(
				"RealSense 影格逾時（%s）。請確認相機沒有被其他程式佔用。"
				% error
			)
		if self.aligner is not None:
			frames = self.aligner.process(frames)

		color_frame = frames.get_color_frame()
		if not color_frame:
			return False, None

		if self.intrinsics is None:
			profile = color_frame.get_profile().as_video_stream_profile()
			values = profile.get_intrinsics()
			self.intrinsics = (
				values.fx, values.fy, values.ppx, values.ppy
			)

		self.depth_image = None
		if self.use_depth:
			depth_frame = frames.get_depth_frame()
			if depth_frame:
				self.depth_image = np.asanyarray(depth_frame.get_data())

		return True, np.asanyarray(color_frame.get_data())

	def describe_depth(self, center_px):
		"""目標中心的公尺座標，供 run_on_stream 併進輸出。"""
		return build_depth_fields(
			center_px, self.depth_image, self.intrinsics
		)

	def get(self, prop):
		if prop == cv2.CAP_PROP_FPS:
			return float(self.fps)
		return 0.0

	def isOpened(self):
		return True

	def release(self):
		self.pipeline.stop()


def run_on_stream(capture, args):
	"""逐格讀取影片或相機畫面，即時偵測並可選擇顯示/儲存結果。"""
	angle_filter = AxisAngleFilter(alpha=0.25)
	writer = None
	window_name = "TDK Straw Detection"
	output_path = None if args.no_save or not args.output else Path(args.output)
	# 機器運行模式不需要標註圖，跳過繪圖省下約四分之一的處理時間。
	draw = not args.robot

	try:
		while True:
			ok, frame = capture.read()
			if not ok:
				break

			outcome = process_frame(frame, args, angle_filter, draw)
			if outcome is None:
				display = frame
				# 明確送出無效值，控制端才能區分「沒偵測到」與「誤差為 0」。
				if args.robot:
					emit_payload(build_robot_command({"valid": False}))
				elif args.emit_json:
					emit_payload({"valid": False})
			else:
				result = outcome["result"]
				payload = outcome["payload"]
				# RealSense 輸入才有深度，其餘來源沒有這個方法。
				if hasattr(capture, "describe_depth"):
					payload = dict(payload)
					payload.update(capture.describe_depth(result["center"]))
				# 可信度不足時視同沒有偵測到：寧可讓控制端維持前一個指令，
				# 也不要送出一個看似合理但方向可能錯 90 度的誤差。
				if result["confidence"] < args.min_confidence:
					payload = {"valid": False}
				if args.robot:
					emit_payload(build_robot_command(payload))
				elif args.emit_json:
					emit_payload(payload)
				else:
					print(
						f"角度誤差: {outcome['errors']['heading_error']:+6.1f} deg  "
						f"橫向誤差: {outcome['errors']['lateral_error']:+7.1f} px  "
						f"({outcome['errors']['lateral_error_ratio']:+.2f})  "
						f"可信度: {result['confidence']:.2f}"
						+ (
							f"  距離: {payload['distance_m']:.3f} m"
							if payload.get("has_depth")
							else ""
						)
					)
				display = outcome["visualization"]
				if display is None:
					display = frame

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
		"--realsense",
		action="store_true",
		help="以 pyrealsense2 直接讀取 RealSense（OpenCV 的 --camera 抓不到其彩色串流）",
	)
	parser.add_argument(
		"--realsense-size",
		type=int,
		nargs=2,
		metavar=("W", "H"),
		default=(1280, 720),
		help="RealSense 串流解析度",
	)
	parser.add_argument(
		"--realsense-fps",
		type=int,
		default=30,
		help="RealSense 串流影格率",
	)
	parser.add_argument(
		"--no-realsense-depth",
		action="store_true",
		help="RealSense 模式下不啟用深度串流",
	)
	parser.add_argument(
		"--no-display",
		action="store_true",
		help="不開啟即時顯示視窗，適合無頭環境",
	)
	parser.add_argument(
		"--robot",
		action="store_true",
		help="機器運行模式：只輸出角度與橫向誤差，跳過繪圖與顯示",
	)
	parser.add_argument(
		"--min-confidence",
		type=float,
		default=None,
		help="低於此可信度視同沒有偵測到；機器運行模式預設 0.5，其餘預設不過濾",
	)
	parser.add_argument(
		"--emit-json",
		action="store_true",
		help="以 JSON Lines 輸出控制量給機器人中心電腦，取代人類可讀的輸出",
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
		default=DEFAULT_ROBOT_ANGLE,
		help="機器人前進方向在影像座標中的角度",
	)
	return parser.parse_args()


def main():
	# 讀取參數，依序完成 mask 清理、目標選取、方向分析與結果輸出。
	args = parse_args()

	if args.min_confidence is None:
		# 機器運行模式的錯誤會直接變成錯誤的動作，預設就要過濾；
		# 其他模式維持原本不過濾的行為，以免影響既有用法。
		args.min_confidence = 0.5 if args.robot else 0.0
	if args.robot:
		# 機器上沒有螢幕，也不需要錄影。
		args.no_display = True
		args.no_save = True

	if args.realsense:
		width, height = args.realsense_size
		capture = RealSenseCapture(
			width, height, args.realsense_fps, not args.no_realsense_depth
		)
		if args.output == DEFAULT_IMAGE_OUTPUT:
			args.output = DEFAULT_VIDEO_OUTPUT
		run_on_stream(capture, args)
		return

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

	result = analyze_target(component["mask"])
	if result is None:
		raise RuntimeError(
			"無法可靠擬合左右側邊，本幀不輸出中心與方向。"
		)

	# 相機連續取像時，應在影像迴圈外建立並重複使用同一個 filter。
	angle_filter = AxisAngleFilter(alpha=0.25)
	result = apply_angle_filter(result, angle_filter)

	errors = calculate_control_errors(
		result, mask.shape[1], args.robot_angle
	)
	visualization = make_visualization(
		cleaned_mask,
		component["mask"],
		result,
		errors,
	)

	save_outputs(
		args.output,
		visualization,
		mask,
		component["mask"],
		args.save_masks,
		args.mask is None,
	)

	if args.emit_json:
		emit_payload(build_robot_payload(result, errors, mask.shape))
		return

	print(f"目標面積: {component['area']} px")
	print(f"目標中心: ({result['center'][0]:.1f}, {result['center'][1]:.1f})")
	print(f"兩側邊估計長度: {result['long_side']:.1f} px")
	print(f"兩側邊向量和角度: {result['angle']:.1f} deg")
	print(f"機器人角度誤差: {errors['heading_error']:+.1f} deg")
	print(
		f"橫向誤差: {errors['lateral_error']:+.1f} px "
		f"({errors['lateral_error_ratio']:+.3f} 半畫面寬)"
	)
	print(f"主軸細長比: {result['axis_ratio']:.2f}")
	print(f"輪廓貼齊畫面邊界比例: {result['border_ratio']:.1%}")
	print(f"側邊直線段佔比: {result['straight_ratio']:.1%}")
	print(f"角度標準誤: {result['angle_sigma']:.2f} deg")
	terms = result["confidence_terms"]
	print(
		f"可信度細項: 證據 {terms['evidence']:.2f} / "
		f"精度 {terms['precision']:.2f} / 種子 {terms['seed_margin']:.2f}"
	)
	print(f"方向可信度: {result['confidence']:.2f}")
	print(f"標註圖已儲存: {args.output}")


if __name__ == "__main__":
	main()
