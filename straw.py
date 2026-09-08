"""偵測稻草捆，輸出機器人對準所需的角度誤差與橫向誤差。

支援圖片、影片、一般相機與 RealSense 四種輸入；以 --robot 或
--emit-json 輸出 JSON Lines 給控制端。詳見 README.md。
"""

import argparse
import collections
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


# 稻草捆的 HSV 色彩範圍，依實際光線調整。
DEFAULT_LOWER_HSV = (16, 84, 42)
DEFAULT_UPPER_HSV = (33, 255, 255)
DEFAULT_MIN_AREA = 5000
DEFAULT_MORPHOLOGY_KERNEL = 11
DEFAULT_RANSAC_THRESHOLD = 12.0
DEFAULT_RANSAC_ITERATIONS = 300
DEFAULT_BORDER_MARGIN = 2
# 左右側邊夾角的上限。真實的兩條側邊接近平行，透視收斂實測不超過 30 度；
# 超過此值代表兩條線擬在同一段圓弧上，不是物體的兩側。
DEFAULT_MAX_SIDE_ANGLE_DIFFERENCE = 45.0
# 種子方向的修正次數上限。方向收斂就提前停止，因此好的偵測通常只花一次。
DEFAULT_SEED_REFINE_STEPS = 4
# 種子修正的收斂容許角度；方向變化小於此值就停止，省下一次擬合。
DEFAULT_SEED_CONVERGED_DEGREES = 2.0
# 候選要與最佳解相差這麼多度，才算「另一種解讀」而構成 90 度翻轉的疑慮。
DEFAULT_SEED_RIVAL_MIN_ANGLE = 20.0
# 競爭者的側邊還要有最佳解的這個比例長。長方形的兩個方向都有直邊，
# 但側邊明顯較短的那個描述的是短軸，不是長軸的另一種解讀。
DEFAULT_SEED_RIVAL_MIN_LENGTH = 0.7
# 影像座標中畫面的縱向為 90 度。機器人對準稻草捆長軸時，
# 目標軸線應與畫面縱向重合，此時角度誤差為 0。
DEFAULT_ROBOT_ANGLE = 90.0
# 相機相對機器人瞄準軸的安裝偏差，預設為零（相機裝在中軸線上正對前方）。
# 三者由 --calibrate 一次量出，見 AxisAlignment。
DEFAULT_AXIS_OFFSET_M = 0.0
DEFAULT_AXIS_OFFSET_RATIO = 0.0
DEFAULT_AXIS_YAW_DEG = 0.0
# 校正取樣：影片取最後這幾秒，即時來源收集這麼多格。
DEFAULT_CALIBRATION_SECONDS = 1.0
DEFAULT_CALIBRATION_FRAMES = 30
# 校正離散度上限。超過代表這段畫面裡的姿態根本不穩，寧可不寫檔：
# 一個看似合理的壞校正會讓之後每一格都偏，而且不會有任何徵兆。
DEFAULT_CALIBRATION_MAX_ANGLE_SPREAD = 5.0
DEFAULT_CALIBRATION_MAX_RATIO_SPREAD = 0.10
DEFAULT_FILTER_ALPHA = 0.25
# 機器運行模式的可信度門檻；錯誤的讀數會直接變成錯誤的動作。
DEFAULT_MIN_CONFIDENCE = 0.5
# 標註圖的軸線顏色（BGR）。通過門檻用橘色，未通過維持灰色。
# 目標區塊本身也是橘色，因此軸線一律先描一圈深色外框才有對比。
AXIS_COLOUR_PASS = (0, 140, 255)
AXIS_COLOUR_FAIL = (180, 180, 180)
AXIS_OUTLINE_COLOUR = (40, 40, 40)
# RealSense 深度影像為 16UC1，單位公釐。
DEFAULT_DEPTH_SCALE = 0.001
DEFAULT_DEPTH_PATCH_RADIUS = 6
# 可信度三項證據的模糊區間端點，依實測值設定（見 calculate_confidence）。
DEFAULT_STRAIGHT_FLOOR = 0.5
DEFAULT_STRAIGHT_TARGET = 0.85
DEFAULT_ANGLE_SIGMA_LIMIT = 2.0
DEFAULT_SEED_MARGIN_TARGET = 0.25
# 沒指定 --calibration 時自動找這個檔。在機器上忘記帶參數會讓機器人
# 瞄偏一個相機側偏的距離，而畫面看起來完全正常；預設載入才是安全的
# 那一邊。載入時一律在 stderr 印出實際採用的偏差，不會無聲生效。
DEFAULT_CALIBRATION_FILE = "axis_calibration.json"
DEFAULT_IMAGE_OUTPUT = "output/straw_detection.png"
DEFAULT_VIDEO_OUTPUT = "output/straw_detection.mp4"


def load_image(image_path):
	"""讀取並驗證 BGR 原始影像。"""
	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"找不到圖片，請檢查路徑: {image_path}")
	return image


def create_color_mask(
	image_bgr, lower_bound, upper_bound, kernel_size=DEFAULT_MORPHOLOGY_KERNEL
):
	"""以 HSV 色彩範圍建立稻草捆 mask，並做形態學去雜訊。"""
	image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

	# 先去除小雜訊，再填補目標區域的小缺口。
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


def select_largest_component(mask, min_area=DEFAULT_MIN_AREA):
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

	return {
		"mask": target_mask,
		"area": area,
	}


def axis_angle_difference(first_angle, second_angle):
	"""計算兩條沒有正反方向之分的軸線，其最小夾角。"""
	difference = abs(first_angle - second_angle) % 180.0
	return min(difference, 180.0 - difference)


class AxisAngleFilter:
	"""針對 180 度週期的物體軸線做連續影格平滑。"""

	def __init__(self, alpha=DEFAULT_FILTER_ALPHA):
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
		"inlier_mask": best_inliers,
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


def fit_side_edges(
	contour_points,
	border_ratio,
	seed_direction,
	max_side_angle_difference=DEFAULT_MAX_SIDE_ANGLE_DIFFERENCE,
):
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
	left_angle = float(
		np.degrees(np.arctan2(left_direction[1], left_direction[0]))
	)
	right_angle = float(
		np.degrees(np.arctan2(right_direction[1], right_direction[0]))
	)
	side_angle_difference = axis_angle_difference(left_angle, right_angle)

	# 目標只露出一小截時，兩條線可能都擬在同一段端點圓弧上，成為該弧的
	# 兩條切線 —— 此時夾角會遠大於透視造成的收斂。真實側邊接近平行，
	# 實測正常情形不超過 30 度，因此夾角過大就判定這個種子方向失敗。
	# 這是物理合理性的硬性否決，與可信度的漸進評分是兩回事：可信度
	# 刻意不看夾角，因為在正常範圍內夾角大反而代表擬到了真正的側邊。
	if side_angle_difference > max_side_angle_difference:
		return None

	summed_angle = float(
		np.degrees(np.arctan2(summed_direction[1], summed_direction[0]))
	)

	return {
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
		"side_angle_difference": side_angle_difference,
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
		"angle": summed_angle,
	}


def mean_side_length(side_edges):
	"""兩條側邊的平均長度，用來分辨長軸與短軸。"""
	return (side_edges["left_length"] + side_edges["right_length"]) / 2.0


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
	angle_sigma = estimate_angle_sigma(side_edges)
	precision = 1.0 - angle_sigma / DEFAULT_ANGLE_SIGMA_LIMIT
	margin = seed_margin / DEFAULT_SEED_MARGIN_TARGET
	terms = {
		"evidence": float(np.clip(evidence, 0.0, 1.0)),
		"precision": float(np.clip(precision, 0.0, 1.0)),
		"seed_margin": float(np.clip(margin, 0.0, 1.0)),
	}
	# 一併回傳，呼叫端不必再算一次。
	return float(min(terms.values())), terms, angle_sigma


def refine_side_edges(
	contour_points,
	border_ratio,
	seed_direction,
	max_steps=DEFAULT_SEED_REFINE_STEPS,
):
	"""以上一輪的擬合結果當新種子反覆擬合，直到方向收斂。

	目標只露出一小截時 mask 接近方形，PCA 主軸可能偏離真實長軸數十度。
	種子只用來決定分箱的縱向座標，偏太多會讓左右邊界都落在同一側 ——
	實測有一張圖的兩側中點只相距 14 px，而目標寬度是 260 px。
	用上一輪算出的方向重新分箱可以逐步修正，通常兩三次就穩定。

	回傳各輪中分數最高者而非最後一輪：迭代可能在兩個方向間震盪，
	取最佳比取最後可靠。
	"""
	best = None
	best_score = -1.0
	seed = np.asarray(seed_direction, dtype=np.float32)
	for _ in range(max_steps):
		fitted = fit_side_edges(contour_points, border_ratio, seed)
		if fitted is None:
			# 這一輪失敗不代表前幾輪無效，保留既有的最佳結果。
			break

		score = side_straightness(fitted)
		if score > best_score:
			best = fitted
			best_score = score

		# 直線段已經夠長就不必再修正；修正是為了救回種子偏掉的情況，
		# 對本來就擬得好的影格只是多花一次擬合。
		if score >= DEFAULT_STRAIGHT_TARGET:
			break

		next_seed = np.asarray(fitted["direction"], dtype=np.float32)
		# 方向已經接近種子就不必再試（軸線無正反之分，取絕對值）。
		if abs(float(np.dot(next_seed, seed))) > np.cos(
			np.deg2rad(DEFAULT_SEED_CONVERGED_DEGREES)
		):
			break
		seed = next_seed

	return best


def analyze_target(target_mask):
	"""估計目標中心、實際主軸角度、方向向量與可信度。"""
	extracted = extract_contour_points(target_mask)
	if extracted is None:
		return None
	contour_points, border_ratio = extracted

	# 以面積加權的二階中央動差求主軸。這與對全部前景像素做 PCA 在數學上
	# 等價（實測主軸角度差 0.000 度），但 OpenCV 直接在影像上累加，不必先把
	# 十幾萬個像素座標展開成陣列 —— 實測 3.3 ms vs 7.4 ms。
	# 主軸只用來當找側邊的初始方向，實際方向由兩側邊向量和決定。
	moments = cv2.moments(target_mask, binaryImage=True)
	if moments["m00"] <= 0.0:
		return None
	mu20 = moments["mu20"] / moments["m00"]
	mu02 = moments["mu02"] / moments["m00"]
	mu11 = moments["mu11"] / moments["m00"]

	pca_angle = float(np.degrees(0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)))
	pca_radians = np.deg2rad(pca_angle)
	pca_vector = np.array(
		[np.cos(pca_radians), np.sin(pca_radians)], dtype=np.float32
	)

	# 共變異矩陣的兩個特徵值，比值接近 1 代表形狀接近方形。純粹是診斷
	# 資訊：近似方形造成的 90 度翻轉改由雙種子擬合與種子修正處理。
	spread = np.sqrt(4.0 * mu11 * mu11 + (mu20 - mu02) ** 2)
	major_variance = (mu20 + mu02 + spread) / 2.0
	minor_variance = (mu20 + mu02 - spread) / 2.0
	if minor_variance <= 1e-6:
		axis_ratio = float("inf")
	else:
		axis_ratio = float(major_variance / minor_variance)

	# PCA 只用來提供找左右側邊的初始座標系；實際方向由兩側邊向量和決定。
	# 目標接近方形時 PCA 分不出長短軸，主軸可能剛好指向側邊的垂直
	# 方向，側邊就會汿成上下兩端。主軸與次軸各擬合一次，取內點率高者：
	# 內點率能分辨真實側邊與非側邊，左右夾角則不行（透視收斂會讓正確
	# 解的夾角反而較大）。
	perpendicular_vector = np.array(
		[-pca_vector[1], pca_vector[0]], dtype=np.float32
	)
	candidates = []
	for seed_vector in (pca_vector, perpendicular_vector):
		fitted = refine_side_edges(contour_points, border_ratio, seed_vector)
		if fitted is not None:
			candidates.append((side_straightness(fitted), fitted))
	if not candidates:
		# 側邊是中心與方向的必要資料，失敗時不能用外接矩形中心冒充。
		return None

	candidates.sort(key=lambda item: item[0], reverse=True)
	best_score, side_edges = candidates[0]
	# 只有指向不同軸線、且側邊長度相當的候選才算競爭者。
	# 兩個種子收斂到同一個方向代表彼此印證，是最可靠的情況；而側邊
	# 明顯較短的候選描述的是短軸 —— 長方形的兩個方向都有直邊，若不看
	# 長度，短軸會一直被誤判成勢均力敵的對手而無謂地壓低可信度。
	best_length = mean_side_length(side_edges)
	rival_score = 0.0
	for score, fitted in candidates[1:]:
		if (
			axis_angle_difference(fitted["angle"], side_edges["angle"])
			<= DEFAULT_SEED_RIVAL_MIN_ANGLE
		):
			continue
		if mean_side_length(fitted) < best_length * DEFAULT_SEED_RIVAL_MIN_LENGTH:
			continue
		rival_score = max(rival_score, score)
	seed_margin = (
		(best_score - rival_score) / best_score if best_score > 1e-6 else 0.0
	)

	direction = side_edges["direction"]
	actual_angle = side_edges["angle"]
	target_center = tuple(side_edges["center"])

	confidence, confidence_terms, angle_sigma = calculate_confidence(
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
		"angle_sigma": angle_sigma,
		"seed_margin": float(seed_margin),
		"direction": direction,
		"side_edges": side_edges,
	}


def calculate_heading_error(target_angle, robot_angle=DEFAULT_ROBOT_ANGLE):
	"""回傳 [-90, 90) 度內的無方向軸線角度誤差。"""
	return (target_angle - robot_angle + 90.0) % 180.0 - 90.0


class AxisAlignment:
	"""相機相對機器人瞄準軸的安裝偏差。

	預設全為零，等同相機裝在中軸線上正對前方 —— 與加入本機制之前的
	行為完全相同，既有用法不受影響。

	`offset_m` 是相機的側偏（公尺，正為右）。相機在軸線右側 d 公尺時，
	目標落在畫面正中央代表它其實也在軸線右側 d 公尺，所以換算是**加**
	不是減。有深度時走這條路，與距離無關。

	`offset_ratio` 是同一件事在校正距離上的像素表現（以半畫面寬為單位）。
	相機側偏在畫面上不是固定的像素位移，而是 fx·d/Z，隨距離變動；沒有
	深度就只能固定在校正距離上，收斂後的殘留偏差為 d·(1 - Z/Z_ref)，
	在校正距離為零。因此校正應該在機器人真正要停的位置做。

	兩者由同一次校正一起產生，在校正距離上必定一致，深度掉格時控制端
	在 lateral_error_m 與 lateral_error_ratio 之間切換不會看到跳變。

	`yaw_deg` 是相機的偏航（正為朝右）。忽略它的殘留偏差是
	(Z - Z_ref)·tan(yaw)，同樣在校正距離為零，因此預設為 0；要補的話
	在兩個距離各量一次 lateral_error_m，斜率就是 tan(yaw)。
	"""

	def __init__(
		self,
		robot_angle=DEFAULT_ROBOT_ANGLE,
		offset_m=DEFAULT_AXIS_OFFSET_M,
		offset_ratio=DEFAULT_AXIS_OFFSET_RATIO,
		yaw_deg=DEFAULT_AXIS_YAW_DEG,
		image_width=None,
	):
		self.robot_angle = float(robot_angle)
		self.offset_m = float(offset_m)
		self.offset_ratio = float(offset_ratio)
		self.yaw_deg = float(yaw_deg)
		# 校正時的畫面寬度，用來擋下「換了解析度卻沿用同一個 ratio」。
		self.image_width = (
			int(image_width) if image_width is not None else None
		)
		self.warned_width = False

	def to_camera_axis(self, x_metres, z_metres):
		"""把相機座標的橫向位移換算成相對機器人瞄準軸的位移。"""
		yaw = np.deg2rad(self.yaw_deg)
		return (
			x_metres * np.cos(yaw) + z_metres * np.sin(yaw) + self.offset_m
		)

	def aim_x(self, image_width):
		"""瞄準軸在畫面上的像素橫座標；偏差為零時就是畫面中線。

		基準刻意維持畫面中線而非光心 cx：校正量到的是「瞄準點在哪」，
		光心偏移已經含在裡面，換基準不會改變結果，反而會讓沒有內參的
		來源（影片、一般相機）與 RealSense 用不同基準。
		"""
		self.check_image_width(image_width)
		half_width = image_width / 2.0
		return half_width + self.offset_ratio * half_width

	def check_image_width(self, image_width):
		"""換了解析度卻沿用同一個 offset_ratio 時警告一次。

		ratio 以半畫面寬為單位，純粹縮放（1280x720 -> 640x360）不受影響，
		但改變長寬比會換掉感測器的裁切範圍，水平視角跟著變 —— 實測同一
		台 D435 在 640x480 量到 0.478、1280x720 量到 0.359。此時 ratio
		會靜靜地錯掉，而 axis_offset_m 不受影響。
		"""
		if self.warned_width or self.image_width is None:
			return
		if self.offset_ratio == 0.0 or int(image_width) == self.image_width:
			return
		self.warned_width = True
		print(
			"警告：校正時畫面寬 %d，現在是 %d。axis_offset_ratio 只在原本的"
			"視角下正確，換長寬比會失準（axis_offset_m 不受影響）。"
			"請以現在的解析度重新校正。"
			% (self.image_width, int(image_width)),
			file=sys.stderr,
		)

	def is_default(self):
		"""是否等同「相機裝在中軸線上正對前方」。"""
		return (
			self.robot_angle == DEFAULT_ROBOT_ANGLE
			and self.offset_m == DEFAULT_AXIS_OFFSET_M
			and self.offset_ratio == DEFAULT_AXIS_OFFSET_RATIO
			and self.yaw_deg == DEFAULT_AXIS_YAW_DEG
		)

	def describe(self):
		return (
			"robot_angle=%.2f deg  axis_offset_m=%+.3f m  "
			"axis_offset_ratio=%+.4f  axis_yaw_deg=%+.2f deg"
			% (
				self.robot_angle,
				self.offset_m,
				self.offset_ratio,
				self.yaw_deg,
			)
		)


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
	alignment=None,
):
	"""把目標中心像素還原成相機座標系的公尺座標。

	intrinsics 為 (fx, fy, cx, cy)。回傳的欄位一律存在，has_depth 說明
	這次有沒有取到有效深度，控制端不必用「欄位在不在」來判斷。

	alignment 為相機的安裝偏差；lateral_error_m 是**相對機器人瞄準軸**
	的位移，position_m 與 camera_lateral_m 則維持原始相機座標，校正與
	除錯要看未補償的值時用後者。
	"""
	missing = {"has_depth": False}
	if depth_image is None or intrinsics is None:
		return missing

	depth_metres = sample_depth_patch(
		depth_image, center_px, radius, depth_scale
	)
	if depth_metres is None:
		return missing

	if alignment is None:
		alignment = AxisAlignment()
	fx, fy, cx, cy = intrinsics
	# 針孔模型反投影。x 向右、y 向下、z 向前，單位公尺。
	x = (float(center_px[0]) - cx) * depth_metres / fx
	y = (float(center_px[1]) - cy) * depth_metres / fy
	lateral = float(alignment.to_camera_axis(x, depth_metres))
	return {
		"has_depth": True,
		"distance_m": round(depth_metres, 4),
		"lateral_error_m": round(lateral, 4),
		"camera_lateral_m": round(x, 4),
		"position_m": [round(x, 4), round(y, 4), round(depth_metres, 4)],
	}


def calculate_control_errors(result, image_width, alignment=None):
	"""計算要回傳給機器人的兩個控制量。

	角度誤差：目標長軸與機器人前進方向的夾角，0 代表已對正。
	正值代表目標頂端偏向畫面右側。

	橫向誤差：目標中心相對機器人瞄準軸的水平位移，0 代表已對中。
	正值代表目標位於瞄準軸右側。相機裝在中軸線上時瞄準軸就是畫面中線；
	相機偏離中軸線時由 alignment.offset_ratio 把瞄準點移到正確的位置。

	橫向誤差同時提供像素值與正規化值。正規化值以半個畫面寬為單位，
	範圍約 [-1, 1]，不受解析度影響，控制端不必知道相機規格即可使用。
	像素值要換算成實際距離則需要相機標定與目標距離，本程式不提供。
	"""
	if alignment is None:
		alignment = AxisAlignment()
	heading_error = calculate_heading_error(
		result["angle"], alignment.robot_angle
	)
	half_width = image_width / 2.0
	aim_x = alignment.aim_x(image_width)
	lateral_error = float(result["center"][0]) - aim_x
	return {
		"heading_error": float(heading_error),
		"lateral_error": lateral_error,
		"lateral_error_ratio": lateral_error / half_width,
		"aim_x": aim_x,
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


def display_confidence_threshold(args):
	"""標註圖上判定「通過」所用的門檻。

	--min-confidence 在非機器運行模式預設為 0（不過濾輸出），但標註圖
	若用 0 會讓每一格都顯示通過，失去意義。因此未過濾時退回
	DEFAULT_MIN_CONFIDENCE，也就是機器運行模式實際採用的標準。
	"""
	if args.min_confidence > 0.0:
		return args.min_confidence
	return DEFAULT_MIN_CONFIDENCE


def make_visualization(
	mask, target_mask, result, errors, min_confidence=DEFAULT_MIN_CONFIDENCE
):
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

	# 雙向箭頭是左右側邊向量相加後的圓柱體實際朝向。
	# 橘色代表可信度通過門檻、這一格會被控制端採用；灰色代表未通過。
	passed = result["confidence"] >= min_confidence
	axis_colour = AXIS_COLOUR_PASS if passed else AXIS_COLOUR_FAIL
	direction = result["direction"]
	arrow_length = max(result["long_side"] * 0.5, 40.0)
	start = np.array(center, dtype=np.float32) - direction * arrow_length
	end = np.array(center, dtype=np.float32) + direction * arrow_length

	# 先粗後細畫兩次；深色外框讓橘色軸線在同樣是橘色的目標上仍有對比。
	for colour, thickness in ((AXIS_OUTLINE_COLOUR, 8), (axis_colour, 4)):
		cv2.line(
			visualization,
			tuple(np.round(start).astype(int)),
			tuple(np.round(end).astype(int)),
			colour,
			thickness,
			lineType=cv2.LINE_AA,
		)
		# 兩端都加上箭頭，表示這是沒有正反方向差異的軸線。
		for point, vector in ((start, direction), (end, -direction)):
			arrow_tip = point + vector * 28.0
			cv2.arrowedLine(
				visualization,
				tuple(np.round(point).astype(int)),
				tuple(np.round(arrow_tip).astype(int)),
				colour,
				thickness,
				tipLength=0.35,
			)

	# 綠色垂直線是機器人瞄準軸，橫線標出目標中心離它多遠。
	# 相機不在中軸線上時瞄準軸會離開畫面中線，此時另外用細灰線標出
	# 畫面中線，才看得出補償了多少。
	aim_x = int(round(errors.get("aim_x", visualization.shape[1] / 2.0)))
	middle_x = visualization.shape[1] // 2
	if aim_x != middle_x:
		cv2.line(
			visualization,
			(middle_x, 0),
			(middle_x, visualization.shape[0]),
			(120, 120, 120),
			1,
		)
	cv2.line(
		visualization,
		(aim_x, 0),
		(aim_x, visualization.shape[0]),
		(0, 255, 0),
		2,
	)
	cv2.line(
		visualization,
		(aim_x, center[1]),
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
		f"方向可信度: {result['confidence']:.2f} / 門檻 {min_confidence:.2f}"
		f"  {'通過' if passed else '未通過'}",
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
	mask = create_color_mask(
		frame_bgr, lower_bound, upper_bound, args.kernel_size
	)
	component = select_largest_component(mask, args.min_area)
	if component is None:
		return None

	result = analyze_target(component["mask"])
	if result is None:
		return None

	result = apply_angle_filter(result, angle_filter)
	errors = calculate_control_errors(
		result, frame_bgr.shape[1], args.alignment
	)
	visualization = None
	if build_visualization:
		visualization = make_visualization(
			mask,
			component["mask"],
			result,
			errors,
			display_confidence_threshold(args),
		)
	return {
		"visualization": visualization,
		"result": result,
		"errors": errors,
		"payload": build_robot_payload(result, errors, frame_bgr.shape),
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
		try:
			self.profile = self.pipeline.start(config)
		except RuntimeError as error:
			# SDK 只會說 "Couldn't resolve requests"，看不出原因。
			# 最常見的是相機插在 USB 2.0 埠：D435 在 USB 2.1 模式下
			# 深度最高只到 640x480，1280x720 無法成立。
			raise RuntimeError(
				"無法以 %dx%d@%d 啟動 RealSense（%s）。%s"
				% (width, height, fps, error, self.describe_device())
			)
		# 深度對齊到彩色，兩者才會共用同一組內參與同一個像素座標。
		self.aligner = rs.align(rs.stream.color) if use_depth else None

	def describe_device(self):
		"""回報連線型態，USB 2.x 會大幅限制可用的解析度。"""
		try:
			devices = list(self.rs.context().query_devices())
			if not devices:
				return "找不到 RealSense 裝置。"
			usb = devices[0].get_info(
				self.rs.camera_info.usb_type_descriptor
			)
		except Exception:
			return "無法讀取裝置資訊。"

		if usb.startswith("2"):
			return (
				"目前為 USB %s 連線，此模式下深度最高只到 640x480；"
				"請改插 USB 3 埠，或加上 --realsense-size 640 480。" % usb
			)
		return "目前為 USB %s 連線，請確認相機沒有被其他程式佔用。" % usb

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

	def describe_depth(self, center_px, alignment=None):
		"""目標中心的公尺座標，供 run_on_stream 併進輸出。"""
		return build_depth_fields(
			center_px,
			self.depth_image,
			self.intrinsics,
			alignment=alignment,
		)

	def get(self, prop):
		if prop == cv2.CAP_PROP_FPS:
			return float(self.fps)
		return 0.0

	def isOpened(self):
		return True

	def release(self):
		self.pipeline.stop()


class LatestFrameCapture:
	"""在背景執行緒持續讀取，永遠只保留最新的一格。

	即時控制拿到的誤差必須反映當下。處理一格的期間相機仍在產出影格，
	若逐格排隊處理，延遲會不斷累積，控制端收到的永遠是過期的狀態。
	舊影格對控制沒有價值，直接丟棄。

	另一個好處是讀取與處理可以重疊：RealSense 的深度對齊本身就要花時間，
	放到背景執行緒後，總耗時從「讀取 + 處理」變成「兩者取大」。

	只用於 cv2.VideoCapture 的相機來源：該路徑的驅動會累積佇列，實測
	處理耗時 100 ms 時 read() 只花 0.2 ms 就回傳，代表拿到的是過期影格。
	RealSense 不需要，其 pipeline 已經只保留最新的 frameset。
	影片檔更不可使用：丟格等於跳過內容。

	介面與 cv2.VideoCapture 相同，run_on_stream 不必區分。
	"""

	def __init__(self, capture):
		self.capture = capture
		self.lock = threading.Lock()
		self.frame = None
		self.dropped = 0
		self.error = None
		self.finished = False
		self.running = True
		self.thread = threading.Thread(target=self.reader, daemon=True)
		self.thread.start()

	def reader(self):
		"""背景讀取；新影格直接覆蓋尚未被取用的舊影格。"""
		while self.running:
			try:
				ok, frame = self.capture.read()
			except Exception as error:
				# 例外要帶回主執行緒丟出，否則會靜默地停止供應影格。
				with self.lock:
					self.error = error
					self.finished = True
				return

			if not ok:
				with self.lock:
					self.finished = True
				return

			with self.lock:
				if self.frame is not None:
					self.dropped += 1
				self.frame = frame

	def read(self):
		"""取出最新的一格。沒有新影格時等待，不重複回傳同一格。"""
		while True:
			with self.lock:
				if self.error is not None:
					raise self.error
				if self.frame is not None:
					frame = self.frame
					self.frame = None
					return True, frame
				if self.finished:
					return False, None
			time.sleep(0.001)

	def get(self, prop):
		return self.capture.get(prop)

	def isOpened(self):
		return self.capture.isOpened()

	def release(self):
		self.running = False
		# 讀取執行緒可能正阻塞在 read()，等一下就好，它是 daemon。
		self.thread.join(timeout=1.0)
		self.capture.release()


def wrap_live_capture(capture, args):
	"""相機來源才套用丟格讀取；影片檔丟格等於跳過內容。"""
	if args.no_frame_drop:
		return capture
	return LatestFrameCapture(capture)


def run_on_stream(capture, args):
	"""逐格讀取影片或相機畫面，即時偵測並可選擇顯示/儲存結果。"""
	angle_filter = AxisAngleFilter(alpha=DEFAULT_FILTER_ALPHA)
	writer = None
	window_name = "TDK Straw Detection"
	output_path = None if args.no_save or not args.output else Path(args.output)
	# 機器運行模式不需要標註圖，跳過繪圖省下約四分之一的處理時間。
	draw = not args.robot
	# 包裝層會把 describe_depth 轉給底層，但沒有深度的來源不該被誤判。
	source = getattr(capture, "capture", capture)
	has_depth_source = hasattr(source, "describe_depth")

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
				if has_depth_source:
					payload = dict(payload)
					payload.update(
						capture.describe_depth(
							result["center"], args.alignment
						)
					)
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

	dropped = getattr(capture, "dropped", 0)
	if dropped and not args.robot:
		# 丟格是預期行為，但數量能看出處理速度跟不跟得上相機。
		print(f"為了取得最新畫面，丟棄了 {dropped} 格")


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
	"""套用跨影格軸線濾波，並更新繪圖與控制使用的方向。

	angle_filter 為 None 時不平滑，直接沿用本格的原始角度。校正走這條
	路：濾波的暖機過渡會混進取樣視窗，而多格取中位數本來就比指數平滑
	更能抗離群值。
	"""
	if angle_filter is None:
		return result

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


def axis_angle_median(angles):
	"""軸線角度的中位數。

	軸線角度是 180 度週期的：179 度與 1 度其實只差 2 度，直接取中位數
	會在繞回處算出離譜的值。先以第一筆為參考把每個角度展開到 ±90 度內，
	取完中位數再折回 [0, 180)。
	"""
	reference = float(angles[0])
	unwrapped = [
		reference + ((float(angle) - reference + 90.0) % 180.0 - 90.0)
		for angle in angles
	]
	return float(np.median(unwrapped)) % 180.0


def axis_angle_spread(angles, centre):
	"""這批軸線角度相對 centre 的全距，單位為度。"""
	deltas = [
		(float(angle) - centre + 90.0) % 180.0 - 90.0 for angle in angles
	]
	return float(max(deltas) - min(deltas))


def collect_calibration_samples(
	capture, args, keep, stop_when_full=False, max_frames=None
):
	"""逐格量測校正用的原始值，只留最後 keep 格。

	只收可信度過門檻的影格。校正一旦寫進檔案，之後每一格都會用到它，
	讓一次錯誤的偵測去定義瞄準軸是最糟的失敗方式 —— 結果看起來完全
	正常，只是每一格都偏。

	影格取樣不套用軸線濾波（angle_filter 傳 None）：濾波的暖機過渡會
	混進取樣視窗，而多格取中位數本來就比指數平滑更能抗離群值。
	"""
	threshold = display_confidence_threshold(args)
	has_depth_source = hasattr(capture, "describe_depth")
	samples = collections.deque(maxlen=keep)
	frames_read = 0
	reported = 0

	while True:
		ok, frame = capture.read()
		if not ok:
			break
		frames_read += 1
		outcome = process_frame(frame, args, None, False)
		if outcome is not None:
			payload = outcome["payload"]
			if payload["confidence"] >= threshold:
				if has_depth_source:
					payload = dict(payload)
					payload.update(
						capture.describe_depth(outcome["result"]["center"])
					)
				samples.append(payload)
		if stop_when_full:
			if len(samples) >= 10 and len(samples) // 10 > reported:
				reported = len(samples) // 10
				print(f"已收集 {len(samples)}/{keep} 格")
			if len(samples) == keep:
				break
		if max_frames is not None and frames_read >= max_frames:
			break

	return list(samples), frames_read


def solve_calibration(samples, image_width, source_name):
	"""從校正取樣解出相機的安裝偏差。

	一律取中位數而非平均：偶爾一格會把軸線判成 90 度翻轉，平均會被
	這種離群值拉走，中位數不會。

	正確姿態下目標相對瞄準軸的位移應為零，因此量到多少位移，相機就是
	往反方向偏了多少 —— 所以 offset 取負號。
	"""
	angles = [sample["angle_deg"] for sample in samples]
	robot_angle = axis_angle_median(angles)

	half_width = image_width / 2.0
	ratios = [
		(float(sample["center_px"][0]) - half_width) / half_width
		for sample in samples
	]
	offset_ratio = float(np.median(ratios))

	# 有深度的取樣才能定出與距離無關的公尺側偏。兩個 offset 取自同一批
	# 影格，因此在校正距離上必定一致，深度掉格時控制端在
	# lateral_error_m 與 lateral_error_ratio 之間切換不會看到跳變。
	depth_samples = [sample for sample in samples if sample.get("has_depth")]
	if depth_samples:
		offset_m = -float(
			np.median(
				[sample["camera_lateral_m"] for sample in depth_samples]
			)
		)
		reference_distance = round(
			float(
				np.median([sample["distance_m"] for sample in depth_samples])
			),
			4,
		)
	else:
		offset_m = DEFAULT_AXIS_OFFSET_M
		reference_distance = None

	return {
		"robot_angle": round(robot_angle, 3),
		"axis_offset_m": round(offset_m, 4),
		"axis_offset_ratio": round(offset_ratio, 5),
		# yaw 不由單一姿態決定：它與側偏在單一距離上完全簡併。要補的話
		# 在兩個距離各量一次 lateral_error_m，斜率就是 tan(yaw)。
		"axis_yaw_deg": DEFAULT_AXIS_YAW_DEG,
		"depth_calibrated": bool(depth_samples),
		"reference_distance_m": reference_distance,
		"image_width": int(image_width),
		"source": source_name,
		"samples": len(samples),
		"angle_spread_deg": round(axis_angle_spread(angles, robot_angle), 3),
		"ratio_spread": round(float(max(ratios) - min(ratios)), 5),
	}


def check_calibration(calibration):
	"""離散度過大時回傳拒絕的理由，通過則回傳 None。"""
	if calibration["angle_spread_deg"] > DEFAULT_CALIBRATION_MAX_ANGLE_SPREAD:
		return (
			"軸線角度在取樣視窗內散佈 %.1f 度，超過上限 %.1f 度"
			% (
				calibration["angle_spread_deg"],
				DEFAULT_CALIBRATION_MAX_ANGLE_SPREAD,
			)
		)
	if calibration["ratio_spread"] > DEFAULT_CALIBRATION_MAX_RATIO_SPREAD:
		return (
			"目標中心在取樣視窗內散佈 %.3f 個半畫面寬，超過上限 %.3f"
			% (
				calibration["ratio_spread"],
				DEFAULT_CALIBRATION_MAX_RATIO_SPREAD,
			)
		)
	return None


def save_calibration(path, calibration):
	"""把校正結果寫成 JSON。"""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)


def load_calibration(path):
	"""讀入校正檔。只有四個參數欄位參與運算，其餘是診斷用的紀錄。"""
	data = json.loads(Path(path).read_text(encoding="utf-8"))
	if not data.get("depth_calibrated", True):
		# 沒有深度的來源量不出公尺側偏。此時若拿去跑 --realsense，
		# lateral_error_m 不會被補償，會與已補償的 lateral_error_ratio
		# 不一致，正是本機制想避免的跳變。
		print(
			"警告：%s 是在沒有深度的來源上校正的，axis_offset_m 仍為 0。"
			"有深度時 lateral_error_m 不會被補償，會與已補償的 "
			"lateral_error_ratio 不一致；請用 --realsense 重新校正，"
			"或手動填入 axis_offset_m。" % path,
			file=sys.stderr,
		)
	return data


def report_calibration(calibration, args, frames_read):
	"""印出校正結果並決定要不要寫檔。"""
	print(f"取樣來源: {calibration['source']}")
	print(f"讀取 {frames_read} 格，其中 {calibration['samples']} 格可用")
	print(
		f"角度離散: {calibration['angle_spread_deg']:.2f} deg  "
		f"中心離散: {calibration['ratio_spread']:.4f} 半畫面寬"
	)
	print(f"robot_angle: {calibration['robot_angle']:.2f} deg")
	print(f"axis_offset_ratio: {calibration['axis_offset_ratio']:+.5f}")
	if calibration["depth_calibrated"]:
		print(f"axis_offset_m: {calibration['axis_offset_m']:+.4f} m")
		print(f"校正距離: {calibration['reference_distance_m']:.3f} m")
	else:
		print("axis_offset_m: 無深度來源，未能量出（維持 0）")

	reason = check_calibration(calibration)
	if reason is not None:
		raise RuntimeError(
			"校正未寫入：%s。\n"
			"這段畫面裡的姿態不夠穩定，寫進去的會是一個看似合理的錯誤"
			"瞄準軸。請確認取樣視窗內機器人與稻草捆都靜止，或改用 "
			"--calibrate-seconds 縮短視窗。" % reason
		)

	save_calibration(args.calibrate, calibration)
	print(f"校正檔已儲存: {args.calibrate}")
	if not calibration["depth_calibrated"]:
		print(
			"提醒：這份校正只補償像素/比例路徑，且只在校正距離上準確。"
			"要與距離無關的補償，請用 --realsense 重新校正一次。"
		)


def run_calibration(capture, args, source_name, live):
	"""收集取樣、解出安裝偏差並寫檔。

	校正量的是「瞄準點絕對在哪」，因此取樣期間一律用零偏差；沿用既有的
	校正檔會讓這次的結果變成相對於上一次的增量。
	"""
	args.alignment = AxisAlignment()
	try:
		if live:
			keep = args.calibrate_frames
			print(f"請保持在正確姿態，收集 {keep} 格可用影格…")
			samples, frames_read = collect_calibration_samples(
				capture, args, keep, True, max_frames=keep * 30
			)
			image_width = None
		else:
			fps = capture.get(cv2.CAP_PROP_FPS)
			if not fps or fps <= 1e-2:
				fps = 30.0
			keep = max(1, int(round(fps * args.calibrate_seconds)))
			frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
			if frame_count and frame_count > keep:
				# 只要最後 keep 格，把前面全部解碼是白費工。seek 可能落在
				# 稍早的關鍵格上，但 deque 只留最後 keep 格，取樣視窗不受
				# 影響；seek 失敗也只是退回從頭讀，結果一樣。
				capture.set(
					cv2.CAP_PROP_POS_FRAMES, float(frame_count - keep)
				)
			print(
				f"取影片最後 {args.calibrate_seconds:.1f} 秒（{keep} 格）"
				"作為正確姿態"
			)
			samples, frames_read = collect_calibration_samples(
				capture, args, keep
			)
			image_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
	finally:
		capture.release()

	if not samples:
		raise RuntimeError(
			"取樣視窗內沒有任何可信度達 %.2f 的影格，無法校正。"
			% display_confidence_threshold(args)
		)
	if image_width is None:
		image_width = samples[-1]["image_size"][0]

	calibration = solve_calibration(samples, image_width, source_name)
	report_calibration(calibration, args, frames_read)


def run_image_calibration(args):
	"""單張參考圖的校正：整段流程只有一格取樣。"""
	if args.mask is not None:
		raise RuntimeError(
			"--calibrate 需要原始影像才能量出瞄準點，不支援 --mask。"
		)

	args.alignment = AxisAlignment()
	image = load_image(args.image)
	outcome = process_frame(image, args, None, False)
	if outcome is None:
		raise RuntimeError("參考圖上找不到可用的目標，無法校正。")

	payload = outcome["payload"]
	threshold = display_confidence_threshold(args)
	if payload["confidence"] < threshold:
		raise RuntimeError(
			"參考圖的可信度只有 %.2f，未達 %.2f，不足以定義瞄準軸。"
			% (payload["confidence"], threshold)
		)

	calibration = solve_calibration([payload], image.shape[1], str(args.image))
	report_calibration(calibration, args, 1)


def parse_args():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--image",
		default="IMG_3231.JPEG",
		help="原始 BGR 圖片路徑",
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
		"--no-frame-drop",
		action="store_true",
		help="即時來源不丟格，逐格處理（延遲會累積，僅供除錯）",
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
		help="HSV mask 的形態學核心大小",
	)
	parser.add_argument(
		"--lower-hsv",
		type=int,
		nargs=3,
		default=DEFAULT_LOWER_HSV,
		metavar=("H", "S", "V"),
		help="HSV 下界",
	)
	parser.add_argument(
		"--upper-hsv",
		type=int,
		nargs=3,
		default=DEFAULT_UPPER_HSV,
		metavar=("H", "S", "V"),
		help="HSV 上界",
	)
	# 以下四個安裝偏差參數的預設值都是 None，用來區分「沒指定」與
	# 「指定成預設值」：沒指定才會退回校正檔，指定了就一律優先。
	parser.add_argument(
		"--robot-angle",
		type=float,
		default=None,
		help="機器人前進方向在影像座標中的角度（預設 %g）"
		% DEFAULT_ROBOT_ANGLE,
	)
	parser.add_argument(
		"--axis-offset-m",
		type=float,
		default=None,
		help="相機相對機器人瞄準軸的側偏（公尺，正為右）；有深度時使用",
	)
	parser.add_argument(
		"--axis-offset-ratio",
		type=float,
		default=None,
		help="同上，但以半畫面寬為單位；沒有深度時使用，只在校正距離上準確",
	)
	parser.add_argument(
		"--axis-yaw-deg",
		type=float,
		default=None,
		help="相機的偏航（度，正為朝右）",
	)
	parser.add_argument(
		"--calibration",
		default=None,
		help="讀入 --calibrate 產生的校正檔；個別參數若明確指定則優先。"
		"未指定時自動找 %s" % DEFAULT_CALIBRATION_FILE,
	)
	parser.add_argument(
		"--no-calibration",
		action="store_true",
		help="不要自動載入 %s" % DEFAULT_CALIBRATION_FILE,
	)
	parser.add_argument(
		"--calibrate",
		default=None,
		help="進入校正模式：把目前輸入來源量到的安裝偏差寫到這個路徑",
	)
	parser.add_argument(
		"--calibrate-seconds",
		type=float,
		default=DEFAULT_CALIBRATION_SECONDS,
		help="影片來源取最後幾秒作為正確姿態",
	)
	parser.add_argument(
		"--calibrate-frames",
		type=int,
		default=DEFAULT_CALIBRATION_FRAMES,
		help="即時來源要收集幾格可用影格",
	)
	args = parser.parse_args()
	if args.min_confidence is None:
		# 機器運行模式的錯誤會直接變成錯誤的動作，預設就要過濾；
		# 其他模式維持原本不過濾的行為，以免影響既有用法。
		args.min_confidence = DEFAULT_MIN_CONFIDENCE if args.robot else 0.0
	args.alignment = resolve_alignment(args)
	return args


def resolve_alignment(args):
	"""決定這次要用的安裝偏差：命令列 > 校正檔 > 預設值。

	沒指定 --calibration 時會自動找 DEFAULT_CALIBRATION_FILE，找不到就
	維持零偏差。校正模式本身在量絕對值，不套用任何既有校正。
	"""
	stored = {}
	path = args.calibration
	if path is None and not args.no_calibration and not args.calibrate:
		default_path = Path(DEFAULT_CALIBRATION_FILE)
		if default_path.is_file():
			path = default_path
	if path is not None:
		stored = load_calibration(path)

	def pick(explicit, key, fallback):
		if explicit is not None:
			return explicit
		return stored.get(key, fallback)

	return AxisAlignment(
		robot_angle=pick(args.robot_angle, "robot_angle", DEFAULT_ROBOT_ANGLE),
		offset_m=pick(args.axis_offset_m, "axis_offset_m", DEFAULT_AXIS_OFFSET_M),
		offset_ratio=pick(
			args.axis_offset_ratio,
			"axis_offset_ratio",
			DEFAULT_AXIS_OFFSET_RATIO,
		),
		yaw_deg=pick(args.axis_yaw_deg, "axis_yaw_deg", DEFAULT_AXIS_YAW_DEG),
		image_width=stored.get("image_width"),
	)


def main():
	# 讀取參數，依序完成 mask 清理、目標選取、方向分析與結果輸出。
	args = parse_args()

	if args.robot:
		# 機器上沒有螢幕，也不需要錄影。
		args.no_display = True
		args.no_save = True

	if not args.alignment.is_default() and not args.calibrate:
		# 走 stderr，才不會混進 --robot / --emit-json 的 JSON Lines。
		print("安裝偏差: %s" % args.alignment.describe(), file=sys.stderr)

	if args.realsense:
		width, height = args.realsense_size
		# RealSense 的 pipeline 本身就只保留最新的 frameset，不需要再包一層
		# 丟格讀取；實測兩者延遲相同（中位 56 vs 55 ms）。
		capture = RealSenseCapture(
			width, height, args.realsense_fps, not args.no_realsense_depth
		)
		if args.calibrate:
			run_calibration(capture, args, "realsense", live=True)
			return
		if args.output == DEFAULT_IMAGE_OUTPUT:
			args.output = DEFAULT_VIDEO_OUTPUT
		run_on_stream(capture, args)
		return

	if args.camera is not None or args.video is not None:
		source = args.video if args.camera is None else args.camera
		capture = cv2.VideoCapture(source)
		if not capture.isOpened():
			raise RuntimeError(f"無法開啟輸入來源: {source}")

		if args.calibrate:
			# 校正不丟格：影片要精準取到最後那段視窗，相機則寧可等
			# 也不要漏掉可用的影格。
			run_calibration(
				capture, args, str(source), live=args.camera is not None
			)
			return

		# 相機是即時來源，可以丟格；影片檔必須逐格處理。
		if args.camera is not None:
			capture = wrap_live_capture(capture, args)

		if args.output == DEFAULT_IMAGE_OUTPUT:
			args.output = DEFAULT_VIDEO_OUTPUT

		run_on_stream(capture, args)
		return

	if args.calibrate:
		run_image_calibration(args)
		return

	mask = build_input_mask(args)
	component = select_largest_component(mask, args.min_area)

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
	angle_filter = AxisAngleFilter(alpha=DEFAULT_FILTER_ALPHA)
	result = apply_angle_filter(result, angle_filter)

	errors = calculate_control_errors(
		result, mask.shape[1], args.alignment
	)
	visualization = make_visualization(
		mask,
		component["mask"],
		result,
		errors,
		display_confidence_threshold(args),
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
