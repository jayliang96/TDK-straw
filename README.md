# TDK-straw

從影像、影片或即時相機畫面中偵測稻草捆，輸出**中心座標**、**長軸角度**與**方向可信度**，供機器人對準與夾取使用。

偵測不依賴機器學習模型，全部以 HSV 色彩門檻加幾何分析完成，可在無 GPU 的嵌入式平台上執行。

## 環境需求

```bash
pip install opencv-python numpy
```

## 快速開始

三種輸入來源，優先順序為 `--camera` > `--video` > `--image`。

```bash
# 單張圖片
python straw.py --image data/IMG_4214.JPEG

# 影片檔（標註結果預設輸出成 output/straw_detection.mp4）
python straw.py --video data/IMG_4215.MOV

# 相機（--camera 不帶數字代表裝置 0）
python straw.py --camera
python straw.py --camera 1

# RealSense（走官方 SDK，不是 OpenCV 的 --camera）
python straw.py --realsense
```

影片與相機模式會開啟即時預覽視窗，按 `q` 或 `Esc` 結束。無頭環境請加 `--no-display`。

> `--image` 的預設值是 `IMG_3231.JPEG`，該檔案不在本專案內，使用圖片模式時請明確指定路徑。
> `data/*.MOV` 因體積過大不進版控，測試影片請自行放入 `data/`。

### RealSense 要用 `--realsense`，不要用 `--camera`

OpenCV 的 UVC 路徑抓不到 RealSense 的彩色串流。D435 會註冊多個 UVC
節點，深度/紅外線那幾個 MSMF 能開啟卻取不到影格，典型錯誤是：

```
videoio(MSMF): can't grab frame. Error: -1072875772
```

`--realsense` 改用 `pyrealsense2` 官方 SDK，順帶也拿得到深度與內參：

```bash
python straw.py --realsense --emit-json
python straw.py --realsense --realsense-size 640 480 --realsense-fps 30
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `--realsense` | 關 | 以 pyrealsense2 讀取 RealSense |
| `--realsense-size` | `1280 720` | 串流解析度 |
| `--realsense-fps` | `30` | 串流影格率 |
| `--no-realsense-depth` | 關 | 不啟用深度串流 |

啟用深度時會把深度對齊到彩色，輸出多出 `has_depth`、`distance_m`、
`lateral_error_m`、`position_m`（欄位意義見 ROS2 節點那一節）。

相機同時只能被一個程式佔用。若出現「影格逾時」，先確認
`realsense_test.py` 或其他程式沒有還開著。

## 處理流程

```
輸入影格
  └─ HSV 色彩門檻 (--lower-hsv / --upper-hsv)
  └─ 形態學開閉運算去雜訊、補洞 (--kernel-size)
  └─ 取最大連通區當作目標 (--min-area)
  └─ 全前景像素做 PCA → 主軸、次軸兩個候選種子方向
  └─ 各自擬合左右側邊，取直線段佔比高者
  └─ 左右側邊方向向量相加 → 長軸角度
  └─ 跨影格軸線平滑 (AxisAngleFilter)
  └─ 中心 / 角度 / 角度誤差 / 可信度
```

### 幾個關鍵設計

**側邊向量相加，而非單邊或外接矩形。** 透視會讓稻草捆的矩形投影成梯形，兩側邊各自傾斜的方向相反，相加可抵消大部分透視偏移。

**剔除貼齊畫面邊界的輪廓點。** 目標超出畫面時，輪廓會沿影像邊界走一整段完美直線。那是裁切痕跡不是稻草邊緣，而 RANSAC 天生偏好這種零殘差的直線，不剔除就會擬出一條沿著畫面邊緣的假側邊。剔除範圍由 `DEFAULT_BORDER_MARGIN`（2 px）控制。

**取最長連續內點區段，而非全部內點。** 圓柱體兩端是圓弧，位於邊界點序列的頭尾。圓弧會持續偏離直線，不可能落在連續內點區段內，因此自然被截掉；只取內點則可能讓兩端圓弧「擦過」直線，形成中間夾著圓弧的假直邊。

**主軸與次軸都試一次。** 目標接近方形時 PCA 分不出長短軸，主軸可能剛好指向側邊的垂直方向，導致側邊擬成上下兩端（90 度翻轉）。兩個方向各擬合一次取較佳者，並用兩者的分數差距當作可信度的一項證據。

## 輸出說明

### 標註圖層

| 顏色 | 意義 |
|---|---|
| 灰階底圖 | 形態學清理後的完整 HSV mask |
| 橘色區塊 | 被選為目標的連通區 |
| 紅色線段 | 擬合出的左右兩條側邊 |
| 紅色圓點 | 估計的目標中心 |
| 灰色雙向箭頭 | 兩側邊向量和的長軸方向（軸線無正反之分，故兩端皆有箭頭） |
| 綠色垂直線 | 機器人中線（畫面中線）；橫線標出目標中心離中線的位移 |

### 終端輸出

| 欄位 | 意義 |
|---|---|
| 目標面積 | 選中連通區的像素數 |
| 目標中心 | 兩條側邊中點的中點 |
| 兩側邊估計長度 | 側邊直線段長度，透視下僅供參考 |
| 兩側邊向量和角度 | 長軸角度，影像座標系，畫面縱向為 90 度 |
| 機器人角度誤差 | 目標軸線與 `--robot-angle` 的差，範圍 [-90, 90)，0 代表已對準 |
| 橫向誤差 | 目標中心相對畫面中線的水平位移，0 代表已對中 |
| 主軸細長比 | PCA 主軸與次軸的變異比 |
| 輪廓貼齊畫面邊界比例 | 越高代表目標被截斷越嚴重 |
| 側邊直線段佔比 | 兩側較差者，越高代表擬到的是真實側邊 |
| 角度標準誤 | 角度本身的不確定度，單位為度 |
| 可信度細項 | 證據 / 精度 / 種子 三項分數 |
| 方向可信度 | 三項取最小值 |

### 可信度怎麼讀

三項獨立證據取**最小值**（而非相乘），這樣能直接看出是哪一項在拖後腿：

- **證據** — 兩側都找到夠長的直線段，代表擬的是真正的側邊而不是圓弧。
- **精度** — 角度的標準誤，回答「這個角度釘得多緊」。
- **種子** — 主軸與次軸兩個候選的分數差距，守住 90 度翻轉這個最壞的失效模式。

刻意**不使用**左右側邊夾角當作懲罰項：稻草捆的兩側邊在透視下本來就會收斂，實測正確偵測的夾角（11~20 度）反而比錯誤偵測（2~11 度）更大。

### 給機器人中心電腦的輸出

加上 `--emit-json` 後，每格輸出一行 JSON（JSON Lines），並立即 flush：

```json
{"valid": true, "heading_error_deg": -0.17, "lateral_error_px": -35.1,
 "lateral_error_ratio": -0.0366, "angle_deg": 89.83,
 "center_px": [924.9, 422.6], "confidence": 0.857,
 "angle_sigma_deg": 0.19, "image_size": [1920, 1080]}
```

| 欄位 | 意義 |
|---|---|
| `valid` | `false` 代表本格沒有可用偵測 |
| `heading_error_deg` | 角度誤差，0 代表已對正；**正值代表目標頂端偏向右側** |
| `lateral_error_px` | 橫向誤差（像素），0 代表已對中；**正值代表目標在中線右側** |
| `lateral_error_ratio` | 同上，但以半個畫面寬為單位，約 [-1, 1]，不受解析度影響 |
| `angle_deg` | 目標長軸角度（影像座標，畫面縱向為 90 度） |
| `center_px` | 目標中心像素座標 |
| `confidence` | 方向可信度，見上一節 |
| `angle_sigma_deg` | 角度的標準誤 |
| `image_size` | `[寬, 高]`，供控制端換算 |

控制端使用時有三點要注意：

1. **`valid: false` 不等於誤差為 0。** 沒有偵測到目標時應維持前一個指令或停止，把缺值當成 0 會讓機器人以為已經對準。
2. **可信度要設門檻。** `IMG_4215.MOV` 全片 641 格中，有 121 格（19%）可信度低於 0.3，那些多半是抓到被畫面切掉的碎片。建議先以 0.5 為界試調。
3. **像素不是距離。** `lateral_error_px` 要換算成公釐需要相機標定與目標距離；同樣的像素偏移，目標越遠代表的實際偏移越大。若只做閉迴路對中（把誤差收斂到 0），直接用 `lateral_error_ratio` 當比例控制的輸入即可。

```bash
# 交給控制程式逐行讀取
python straw.py --camera --no-display --emit-json | python robot_control.py
```

## 參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `--image` | `IMG_3231.JPEG` | 輸入圖片路徑 |
| `--video` | 無 | 輸入影片路徑，優先於 `--image` |
| `--camera` | 無 | 相機裝置編號，不帶數字為 0，優先於 `--video` |
| `--mask` | 無 | 外部二值 mask，指定後略過 HSV 建立（僅圖片模式） |
| `--output` | `output/straw_detection.png` | 標註輸出路徑；影片/相機模式未指定時自動改為 `.mp4` |
| `--no-display` | 關 | 不開預覽視窗（僅影片/相機模式） |
| `--no-save` | 關 | 不儲存輸出（僅影片/相機模式） |
| `--robot` | 關 | 機器運行模式：只輸出角度與橫向誤差，跳過繪圖與顯示 |
| `--no-frame-drop` | 關 | 相機來源不丟格，逐格處理（延遲會累積，僅供除錯） |
| `--min-confidence` | 機器模式 0.5，其餘不過濾 | 低於此可信度視同沒有偵測到 |
| `--emit-json` | 關 | 以 JSON Lines 輸出控制量，取代人類可讀輸出 |
| `--save-masks` | 關 | 額外輸出目標 mask 與完整 HSV mask（僅圖片模式） |
| `--min-area` | 5000 | 有效連通區的最小像素面積 |
| `--kernel-size` | 11 | 形態學核心大小 |
| `--lower-hsv` | 21 65 67 | HSV 下界 |
| `--upper-hsv` | 33 255 255 | HSV 上界 |
| `--robot-angle` | 90.0 | 機器人前進方向在影像座標中的角度 |

## ROS2 節點

`straw_ros2_node.py` 訂閱 RealSense 的彩色與深度話題，發佈對準誤差。
偵測邏輯直接沿用 `straw.py`，本檔只負責 ROS2 的收發與深度換算。

```bash
# 相依：rclpy、cv_bridge（ROS2 桌面版已包含）
python straw_ros2_node.py --ros-args \
    -p image_topic:=/camera/camera/color/image_raw \
    -p use_depth:=true \
    -p min_confidence:=0.5

# 看輸出
ros2 topic echo /straw/target
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `image_topic` | `/camera/camera/color/image_raw` | 來源影像話題 |
| `depth_topic` | `/camera/camera/aligned_depth_to_color/image_raw` | 對齊後的深度話題 |
| `camera_info_topic` | `/camera/camera/color/camera_info` | 相機內參來源 |
| `use_depth` | `true` | 是否訂閱深度並換算公尺座標 |
| `depth_scale` | `0.001` | 深度單位換算（RealSense 為公釐） |
| `depth_patch_radius` | `6` | 深度取樣的鄰域半徑（像素） |
| `sync_slop` | `0.05` | 彩色與深度的時間同步容許誤差（秒） |
| `target_topic` | `straw/target` | 發佈的結果話題（`std_msgs/String` 承載 JSON） |
| `annotated_topic` | `straw/annotated` | 標註影像話題，供 rviz 除錯 |
| `publish_annotated` | `false` | 是否發佈標註影像 |
| `min_confidence` | `0.5` | 低於此值視同沒有偵測到 |
| `filter_alpha` | `0.25` | 跨影格軸線平滑係數 |
| `lower_hsv` / `upper_hsv` / `kernel_size` / `min_area` / `robot_angle` | 同命令列 | 偵測參數 |

訊息內容與 `--emit-json` 相同，另外多了 `stamp_sec` / `stamp_nanosec`
（取自來源影像的 header，供控制端對時）。

### 影格新鮮度

`--camera` 走 OpenCV 的 `VideoCapture`，驅動會累積影格佇列：實測處理
耗時 100 ms 時 `read()` 只花 0.2 ms 就回傳，代表拿到的是過期影格，
而且延遲會隨時間累積。因此相機來源預設以背景執行緒讀取，永遠處理
最新的一格，舊格直接丟棄（`--no-frame-drop` 可關閉）。

**`--realsense` 不套用這個機制。** RealSense 的 pipeline 本身就只保留
最新的 frameset，實測加不加背景執行緒的端到端延遲相同（中位 56 vs
55 ms，量測基準為 SDK 的 `time_of_arrival`），多包一層只是增加複雜度。

影片檔一律不丟格 —— 丟格等於跳過內容。

### 機器運行模式

`--robot` 是給實際跑在機器人上的模式：只輸出控制迴圈會用到的量，
跳過繪圖與顯示，並自動套用可信度門檻。

```bash
python straw.py --realsense --robot | python robot_control.py
```

```json
{"valid": true, "heading_error_deg": 1.26, "lateral_error_m": -0.01, "lateral_error_ratio": -0.0093}
```

**欄位固定不變**，控制端不必判斷欄位在不在。`valid` 為 `false` 時其餘
三個欄位都是 `null` —— 沒有偵測到目標時應維持前一個指令或停止。
沒有深度來源時 `lateral_error_m` 為 `null`，改用 `lateral_error_ratio`。

與其他模式的差別：

| | 一般 | `--emit-json` | `--robot` |
|---|---|---|---|
| 輸出 | 人類可讀 | 完整 JSON | 僅角度與橫向誤差 |
| 繪圖 | 有 | 有 | **無** |
| 顯示視窗 / 存檔 | 有 | 有 | **強制關閉** |
| 可信度過濾 | 無 | 無 | **預設 0.5** |
| 速度（1920x1080） | 46.2 ms | 46.2 ms | **34.0 ms** |

跳過繪圖省下約四分之一的時間（21.6 → 29.4 fps）。可信度門檻用
`--min-confidence` 調整，設 0 可關閉；機器運行模式的錯誤會直接變成
錯誤的動作，因此預設就會過濾。

### 深度：把像素換成公尺

`use_depth` 開啟後（預設開啟），節點會同步訂閱對齊後的深度影像，
把目標中心反投影成相機座標系的公尺座標，訊息多出這幾個欄位：

| 欄位 | 意義 |
|---|---|
| `has_depth` | 這次有沒有取到有效深度；為 `false` 時下面三個欄位不存在 |
| `distance_m` | 目標中心到相機的距離（公尺） |
| `lateral_error_m` | 目標中心相對相機光軸的水平位移（公尺），正值為右 |
| `position_m` | `[x, y, z]`，x 向右、y 向下、z 向前，單位公尺 |

有了 `lateral_error_m`，控制端就不必處理「同樣的像素偏移在不同距離
代表不同實際偏移」這件事，也不需要自己做相機標定。

兩個實作細節：

- **彩色與深度做時間同步**（`ApproximateTimeSynchronizer`，容許誤差
  `sync_slop`）。兩個串流的時間戳不會完全相同，不同步的話會拿這一格的
  目標中心去查上一格的深度。
- **深度取鄰域中位數而非單一像素**（半徑 `depth_patch_radius`）。反光、
  物體邊緣、超出量程都會讓單一像素的深度變成 0，取鄰域並剔除 0 之後
  再取中位數穩定得多。整塊都無效時回傳 `has_depth: false`。

深度不可用時（尚未收到 `camera_info`、深度全為 0、轉換失敗）只會讓
`has_depth` 為 `false`，角度與像素誤差仍照常輸出。

### 兩個容易踩的坑

**QoS 必須用 sensor data。** RealSense 以 `SensorDataQoS`（best effort）
發佈影像；訂閱端若用 rclpy 預設的 reliable QoS，兩邊不相容，會一則訊息
都收不到，而且不會報錯。本節點已使用 `qos_profile_sensor_data`。

**話題名稱依 realsense-ros 版本而異。** 較新版本是
`/camera/camera/color/image_raw`（兩層 camera 命名空間），舊版是
`/camera/color/image_raw`。先用 `ros2 topic list` 確認，再用參數覆寫。

## 效能

1920x1080 影片、`IMG_4215.MOV` 60 格實測（CPU，無 GPU）：

| 階段 | 時間 |
|---|---|
| HSV mask + 形態學 | 9.5 ms |
| 連通區選取 | 9.2 ms |
| `analyze_target` | 18.4 ms |
| 繪圖 | 11.5 ms |
| **整體 `process_frame`** | **48.6 ms（20.6 fps）** |

降低輸入解析度幾乎沒有幫助：只有前兩個階段隨像素數縮放，側邊擬合處理的是 40 個切片點，與解析度無關。640x360 只快約 1.4 倍。

RANSAC 的取樣迴圈已改為矩陣運算一次算完所有候選，`analyze_target` 因此快了 4.3 倍。300 次迭代在向量化後成本很低，故保留此值以維持最大餘裕（理論上內點率 0.5 時 24 次即可 99.9% 收斂）。

無頭執行（`--no-display` 且不寫檔）可省下繪圖的 11.5 ms。

## 已知限制

以下是目標貼齊畫面邊界時仍未處理完的部分（`IMG_4215.MOV` 全片 641 格實測）：

- **目標選取仍以純面積排序。** 有 70 格的最大連通區是被畫面切掉的碎片（例如 230k px 的角落三角形），而畫面中另有完整可見的稻草捆（67k px）被忽略。改進方向是把排序改成 `面積 × 完整度權重`，讓重度截斷的元件失去優先權。
- **沒有單邊退化模式。** 目前只要有一側擬合失敗就整格放棄。實際上單側仍可提供方向（可信度打折），只是中心不可求 —— 物體真正的中心可能在畫面外，用可見部分的形心冒充會系統性偏移。
- **未區分縱向與橫向截斷。** 端點被切時兩條側邊仍完好，方向可用，只有長度是下限；側邊被切才會真正破壞方向。兩者目前一視同仁。
- **種子方向未利用時間資訊。** 截斷時 PCA 會亂跳，可改用上一格的平滑角度當種子（並在與 PCA 差距過大時回退，避免濾波器鎖死在錯誤角度）。

## 檔案

- `straw.py` — 全部偵測邏輯與命令列介面
- `straw_ros2_node.py` — ROS2 節點，訂閱 RealSense 影像並發佈對準誤差
- `data/` — 測試素材（`.MOV` 不進版控）
- `output/` — 程式產生的標註結果，不進版控
