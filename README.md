# TDK-straw

從影像、影片或即時相機畫面偵測稻草捆，輸出機器人對準所需的**角度誤差**與**橫向誤差**。

偵測不依賴機器學習模型，全部以 HSV 色彩門檻加幾何分析完成，可在無 GPU 的嵌入式平台上執行。1920x1080 在一般 CPU 上約 20 fps，機器運行模式約 29 fps。

## 環境需求

```bash
pip install opencv-python numpy
```

依使用情境另外需要：

| 情境 | 額外相依 |
|---|---|
| `--realsense`（直連 RealSense） | `pyrealsense2` |
| ROS2 節點（`ros2/` 下的 package） | ROS2 Humble、`ros-humble-realsense2-camera`；或直接用 Docker，見〈ROS2 package〉 |

## 快速開始

```bash
# 單張圖片
python straw.py --image data/IMG_4214.JPEG

# 影片檔
python straw.py --video data/IMG_4215.MOV

# RealSense 深度相機
python straw.py --realsense

# 實際跑在機器人上
python straw.py --realsense --robot | python robot_control.py
```

影片與相機模式會開啟即時預覽視窗，按 `q` 或 `Esc` 結束。無頭環境加 `--no-display`。

> `--image` 沒有可用的預設值，圖片模式請明確指定路徑。`data/*.MOV` 與 `data/*.mp4` 因體積過大不進版控，測試影片請自行放入 `data/`。

---

## 輸入來源

優先順序為 `--realsense` > `--camera` > `--video` > `--image`。

| 來源 | 參數 | 深度 | 丟格 |
|---|---|---|---|
| 單張圖片 | `--image PATH` | — | — |
| 外部二值 mask | `--mask PATH` | — | — |
| 影片檔 | `--video PATH` | 無 | 永不丟格 |
| 一般相機 | `--camera [N]` | 無 | 預設丟格 |
| RealSense | `--realsense` | 有 | 不需要 |

### RealSense 要用 `--realsense`，不能用 `--camera`

OpenCV 的 UVC 路徑抓不到 RealSense 的彩色串流：D435 會註冊多個 UVC 節點，深度/紅外線那幾個 MSMF 能開啟卻取不到影格（`can't grab frame. Error: -1072875772`），換 index 或換 DSHOW 後端都無效。`--realsense` 改用 `pyrealsense2` 官方 SDK，順帶也拿得到深度與內參。

**務必插在 USB 3 埠。** D435 在 USB 2.x 模式下深度最高只到 640x480，預設的 1280x720 會啟動失敗（SDK 只會回報 `Couldn't resolve requests`）。本程式會偵測連線型態並在錯誤訊息中指出，此時改用 `--realsense-size 640 480` 可先跑起來 —— 但換解析度會動到校正，見〈兩個會靜靜出錯的地方〉。

相機同時只能被一個程式佔用。若出現「影格逾時」，先確認 `realsense_test.py` 或其他程式沒有還開著。

### 錄一段素材

相機不在手邊時要有東西可以重跑。`realsense_test.py --record data/0908.mp4` 會把當下的原始彩色串流錄下來，直接餵給 `--video`；副檔名改 `.bag` 連深度一起錄（一分鐘數百 MB，且 `straw.py` 還沒有讀 bag 的來源）。掉格不會補格，收檔時會印出掉格數。

### 影格新鮮度

`--camera` 走 OpenCV 的 `VideoCapture`，驅動會累積影格佇列，拿到的是過期影格且延遲隨時間累積（實測處理耗時 100 ms 時 `read()` 只花 0.2 ms 就回傳）。因此相機來源預設以背景執行緒讀取，永遠處理最新的一格（`--no-frame-drop` 可關閉）。`--realsense` 不套用這個機制，RealSense 的 pipeline 本身就只保留最新的 frameset，實測延遲不累積。影片檔一律不丟格，丟格等於跳過內容。

---

## 輸出模式

三種模式，依用途選擇：

| | 預設 | `--emit-json` | `--robot` |
|---|---|---|---|
| 用途 | 人工除錯 | 記錄與分析 | 機器人控制 |
| 格式 | 人類可讀文字 | 完整 JSON Lines | 精簡 JSON Lines |
| 繪圖 | 有 | 有 | **無** |
| 顯示視窗 / 存檔 | 有 | 有 | **強制關閉** |
| 可信度過濾 | 無 | 無 | **預設 0.5** |
| 速度（1920x1080） | 46.2 ms | 46.2 ms | **34.0 ms** |

### 機器運行模式 `--robot`

只輸出控制迴圈真正會用到的量：

```json
{"valid": true, "heading_error_deg": 1.26, "lateral_error_m": -0.01, "lateral_error_ratio": -0.0093}
```

**欄位固定不變**，控制端不必判斷欄位在不在。`valid` 為 `false` 時其餘三個欄位都是 `null`。沒有深度來源時 `lateral_error_m` 為 `null`，改用 `lateral_error_ratio`。

### 完整輸出 `--emit-json`

```json
{"valid": true, "heading_error_deg": -0.17, "lateral_error_px": -35.1,
 "lateral_error_ratio": -0.0366, "angle_deg": 89.83,
 "center_px": [924.9, 422.6], "confidence": 0.857,
 "angle_sigma_deg": 0.19, "image_size": [1920, 1080]}
```

| 欄位 | 意義 |
|---|---|
| `valid` | `false` 代表本格沒有可用偵測 |
| `heading_error_deg` | 角度誤差，0 代表已對正 |
| `lateral_error_px` | 橫向誤差（像素），0 代表已對中 |
| `lateral_error_ratio` | 同上，以半個畫面寬為單位，約 [-1, 1]，不受解析度影響 |
| `angle_deg` | 目標長軸角度 |
| `center_px` | 目標中心像素座標 |
| `confidence` | 方向可信度 |
| `angle_sigma_deg` | 角度的標準誤 |
| `image_size` | `[寬, 高]` |

啟用深度時另有 `has_depth`、`distance_m`、`lateral_error_m`、`position_m`。

### 座標與正負號

影像座標系，x 向右、y 向下，**畫面縱向為 90 度**。相機裝在機器人中線上時畫面中線即機器人中線；裝在別處請見〈相機不在中軸線上〉。

| 量 | 正值代表 | 零代表 |
|---|---|---|
| `heading_error_deg` | 目標**頂端偏向畫面右側** | 已對正長軸 |
| `lateral_error_px` / `_ratio` / `_m` | 目標在**瞄準軸右側** | 已對中 |

角度誤差範圍為 [-90, 90)，因為軸線沒有正反之分。`--robot-angle` 可改變機器人前進方向的定義，預設 90 度。

### 控制端要注意的四件事

1. **`valid: false` 不等於誤差為 0。** 沒有偵測到目標時應維持前一個指令或停止；把缺值當成 0 會讓機器人以為已經對準。
2. **可信度要設門檻。** `IMG_4215.MOV` 全片 641 格中有 121 格（19%）可信度低於 0.3，多半是抓到被畫面切掉的碎片。`--robot` 預設以 0.5 過濾。
3. **像素不是距離。** 同樣的像素偏移，目標越遠代表的實際偏移越大。有深度時直接用 `lateral_error_m`；沒有深度而只做閉迴路對中，用 `lateral_error_ratio` 即可，不需要標定。
4. **相機沒裝在中軸線上就要校正。** 否則把誤差收斂到 0 只是讓相機對準稻草捆，機器人本身仍然偏著。見〈相機不在中軸線上〉。

---

## 可信度

三項獨立證據各自正規化到 [0, 1]，取**最小值**（而非相乘）—— 這樣能直接看出是哪一項在拖後腿。

| 項目 | 量測什麼 | 依據 |
|---|---|---|
| **證據** | 擬到的是真側邊還是圓弧 | 兩側較差者的直線段佔比 |
| **精度** | 這個角度釘得多緊 | 角度標準誤 σ = 殘差·√12 / (線段長·√點數) |
| **種子** | 有沒有 90 度翻轉的疑慮 | 與最佳解相差 20 度以上、**且側邊長度相當**的候選的分數差距 |

實測 12 個候選（6 樣本 × 2 種子）：正確解 0.64~0.98，錯誤解全為 0.00。

種子項的競爭者要**指向不同軸線**（兩個種子收斂到同一方向是彼此印證，不是模稜兩可）且**側邊長度相當**（至少為最佳解的 `DEFAULT_SEED_RIVAL_MIN_LENGTH`；圓柱側視時短軸的直線段佔比也很高，不看長度會一直被誤判成勢均力敵的對手）。刻意**不使用**側邊夾角當懲罰項：兩側邊在透視下本來就會收斂，實測正確偵測的夾角反而比錯誤偵測更大。

---

## 深度：把像素換成公尺

`--realsense`（預設開啟深度）與 ROS2 節點都會把目標中心反投影成相機座標系的公尺座標：

| 欄位 | 意義 |
|---|---|
| `has_depth` | 這次有沒有取到有效深度 |
| `distance_m` | 目標中心到相機的距離 |
| `lateral_error_m` | 相對**機器人瞄準軸**的水平位移，正值為右 |
| `camera_lateral_m` | 同上，但未套用安裝偏差補償，供校正與除錯比對 |
| `position_m` | `[x, y, z]`，相機座標系，x 向右、y 向下、z 向前 |

有了 `lateral_error_m`，控制端就不必自己做相機標定，也不必處理「同樣的像素偏移在不同距離代表不同實際偏移」。

- **深度必須與彩色對齊且時間同步**，否則會拿這一格的目標中心去查上一格的深度。`--realsense` 用 SDK 的 `rs.align`，ROS2 節點用 `ApproximateTimeSynchronizer`（容許誤差 `sync_slop`）。
- **深度取鄰域中位數而非單一像素。** 反光、物體邊緣、超出量程都會讓單一像素的深度變成 0；取鄰域（半徑 `depth_patch_radius`）剔除 0 後取中位數，整塊都無效時回傳 `has_depth: false`。

反投影用的是相機**實際內參**而非假設光心在畫面正中央 —— 實測 D435 的 `cx = 651.9`，畫面中心是 640。深度不可用時只會讓 `has_depth` 為 `false`，角度與像素誤差仍照常輸出，不構成單點故障。

---

## 相機不在中軸線上

預設假設相機裝在機器人中軸線上正對前方，於是畫面中線就是瞄準目標。相機裝在別處時，把誤差收斂到 0 只會讓**相機**對準稻草捆，機器人本身仍然偏著。

參數共九個，列在〈命令列參數〉的「安裝偏差」表。正常流程只需要跑一次校正，不必手動填。

### 怎麼校正

把機器人擺在真正該停的位置，讓程式自己讀出參數：

```bash
# 直接對著相機校正，收集 30 格可用影格
python straw.py --realsense --calibrate axis_calibration.json

# 或錄一段以正確姿態收尾的影片，取最後 1 秒
python straw.py --video data/09081.mp4 --calibrate axis_calibration.json

# 或用一張擺好位置的參考圖
python straw.py --image data/aligned.jpg --calibrate axis_calibration.json

# 之後直接跑就好，會自動載入 axis_calibration.json
python straw.py --realsense --robot
```

**校正檔預設會自動載入**。正本在 `ros2/straw_detector/config/axis_calibration.json`，跟著 package 走；執行目錄下若另有一份 `axis_calibration.json` 會優先採用，現場要臨時換校正直接丟在旁邊就好。在機器上忘記帶參數會讓機器人瞄偏一個相機側偏的距離，而畫面看起來完全正常，所以預設載入才是安全的那一邊；載入時一律在 stderr 印出實際採用的偏差。`--calibration` 換檔案，`--no-calibration` 關掉。ROS2 節點同樣預設載入 package 內的正本，用 `calibration_file` 換檔案、`use_calibration:=false` 關掉。

這比量測相機的安裝位置好，不只是省事：它一次吸收 roll、偏航、鏡頭畸變、光心偏移，以及**夾爪相對機器人中心的偏移**。真正要問的問題不是「機器人中軸線在哪」，而是「稻草捆要出現在哪，夾爪才夾得到」—— 後者拿尺量不出來。量出來的參數仍是人看得懂的純文字，可以拿捲尺粗略核對，也可以手動編輯。

校正只收可信度過門檻的影格（同 `--min-confidence`），取中位數而非平均，**離散度過大就拒絕寫檔** —— 一個看似合理的壞校正會讓之後每一格都偏，而且不會有任何徵兆。RealSense 現場實測（1280x720，30 格全部可用）軸線角度散佈 0.21 度、目標中心散佈 0.3 px；套用後 `--robot` 連續跑 119 格，角度誤差 **±0.09 度**、`lateral_error_m` **±0.0004 m**，公尺與 ratio 兩條路徑同時歸零。參數的絕對值每次重新安裝或重新定義正確姿態都會變，重跑校正即可。

### 在容器裡校正

節點本身沒有校正模式，校正一律走 CLI。相機同時只能被一個程式開啟，所以要先把 launch 停掉（終端 1 按 Ctrl-C 即可，容器可以留著），再進容器把結果**直接寫進 package 的正本**：

```bash
docker compose exec straw /entrypoint.sh bash

cd /ws/src/TDK-straw
python3 straw.py --realsense --no-display \
    --calibrate ros2/straw_detector/config/axis_calibration.json
```

**寫完不必重 build。** `--symlink-install` 讓 `install/straw_detector/share/straw_detector/config/axis_calibration.json` 一路連回 `/ws/src/TDK-straw/ros2/straw_detector/config/`，而那裡就是主機上的 repo（bind mount，且容器以主機相同 UID 執行），校正完在主機 `git diff` 就看得到新舊差異。重起 launch 後 log 會印出 `讀入校正檔: ...`，在校正姿態下 `/straw/target` 的 `heading_error_deg` 與 `lateral_error_m` 都該貼著 0。

兩件別做的事：**不要寫到 `/ws/install/` 底下**，也不要在 `config/` 新增檔名不同的校正檔 —— `setup.py` 用的是 build 時展開的 `glob("config/*")`，新檔案要重 build 才會被安裝，覆寫既有的 `axis_calibration.json` 才是免 build 的路徑。**不要改 `--realsense-size`**，預設的 1280x720 正好對上 `straw_with_camera.launch.py` 的 `rgb_profile`。

拿影片或參考圖校正沒有相機佔用的問題，launch 可以繼續跑。

### 兩個會靜靜出錯的地方

**`axis_offset_ratio` 綁在校正時的視角上。** 它以半畫面寬為單位，純粹縮放（1280x720 → 640x360）不受影響，但改變長寬比會換掉感測器的裁切範圍，水平視角跟著變 —— 同一台 D435 實測 640x480 量到 +0.478、1280x720 量到 +0.359。而 USB 2 模式正好需要退到 `--realsense-size 640 480`。因此校正檔記下 `image_width`，執行時寬度不符就在 stderr 警告；實測把 1280 的校正套到 640 的畫面，橫向誤差會從 −0.1 px 變成 38.2 px。`axis_offset_m` 不受影響。

**瞄準軸在畫面上是斜的，不是垂直線。** 它是一條與機器人前進方向平行的 3D 直線，投影後朝消失點收斂，影像角度就是 `robot_angle`。這不只是畫面問題：`lateral_error_px` 量的是目標中心到瞄準軸的水平距離，拿垂直線去量會有 `Δy × 斜率` 的偏差（84.4 度時斜率 0.097，上下差 200 px 就差 19 px）。因此校正檔存下瞄準軸通過的點 `aim_center_px`，執行時在目標所在的高度上量。`lateral_error_m` 在 3D 座標裡算，不受影響。

同一件事的另一面是**「應有的角度」取決於目標在畫面上的位置**：所有與前進方向平行的 3D 線都交於同一個消失點，應有角度就是「從目標中心指向消失點」的方向。用固定 `robot_angle` 時，沿瞄準軸前後移動的誤差是 −0.03 度（任何距離都不變），橫向每偏離 150 px 則多 5.1 度。也就是**橫向對準之後，固定的 `robot_angle` 在任何距離都精確**；誤差只在還沒對準時出現（每 100 px 約 3.4 度），因此**先收橫向、再修角度**的順序天生避開這個問題。

### 為什麼參數存的是公尺

相機往右偏 `d` 公尺時，瞄準軸投影到畫面上的位置是 `cx - fx·d/Z` —— **隨目標距離變動**。D435 彩色在 1280 寬下 `fx` 約 900，取 `d` = 10 cm：1 m 處偏 90 px、2 m 處剩 45 px。在 1 m 校正好的固定像素補償，拉到 2 m 就錯 45 px（半畫面寬的 7%）。

因此有深度時走 `axis_offset_m`，直接在公尺座標裡補償，與距離無關。側偏是**加**不是減：相機在軸線右側 10 cm 時，稻草捆落在畫面正中央代表它其實也在軸線右側 10 cm。`axis_offset_m` 與 `axis_offset_ratio` 由同一次校正一起產生，因此在校正距離上必定一致 —— 深度掉格時控制端在兩個欄位之間切換不會看到跳變。

沒有深度、或忽略偏航時仍有殘留偏差，兩者都在校正距離歸零：

| 近似 | 控制迴圈收斂後的實際橫向偏差 |
|---|---|
| 沒有深度、只靠 `axis_offset_ratio` | `d · (1 − Z/Z_ref)` |
| 有深度、但忽略偏航 | `(Z − Z_ref) · tan(yaw)` |

所以**校正要在機器人真正要停的位置做**。偏航預設不校正：單一距離上它與側偏完全簡併，硬要從兩個姿態擬合反而會把擺放誤差放大成隨距離增長的偏差。要補的話不必寫程式 —— 在兩個距離各擺正一次、各讀一個 `lateral_error_m`，`tan(yaw) = (e₂ − e₁)/(Z₂ − Z₁)`，手算後填進 `--axis-yaw-deg`。

---

## ROS2 package

`ros2/` 底下是兩個 ROS2 package（Humble）：

- `straw_detector`（ament_python）— `straw.py` 的正本住在這裡，加上 ROS2 節點 `detector_node.py`、launch 檔、參數 YAML 與校正檔。
- `straw_interfaces`（ament_cmake）— `StrawTarget.msg`。

repo 根目錄的 `straw.py` 只是轉呼叫 package 內的正本，讓沒裝 ROS2 的電腦仍能 `python straw.py --image ...`。

### 建置

把整個 repo clone 進 workspace 的 `src/`。repo 根沒有 `package.xml`，colcon 會自己往下找到 `ros2/` 裡的兩個 package：

```bash
cd ~/ros2_ws/src && git clone <repo> TDK-straw
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### 用 Docker 跑（主機不必裝 ROS2）

主機是 Ubuntu 24.04 也沒關係，Humble 只活在容器裡。`Dockerfile` 會把 `ros2/` 下的兩個 package build 好，`docker-compose.yml` 已設定 `network_mode: host`（主機的 `ros2 topic echo` 看得到容器內的話題）、`privileged` 加掛整個 `/dev`（RealSense 走 USB）、以及 X11 socket（預覽視窗）。

```bash
docker compose build              # 第一次，或 package.xml / Dockerfile 改過之後
xhost +local:docker               # 要開預覽視窗才需要，每次開機做一次
docker compose up -d straw        # 容器常駐
docker compose exec straw /entrypoint.sh bash   # 每開一個終端就下一次
```

**進容器一定要經過 `/entrypoint.sh`。** `docker compose exec` 不會走 ENTRYPOINT，直接 `exec straw bash` 進去 `ros2` 會是 `command not found`；`entrypoint.sh` 負責 source `/opt/ros/humble/setup.bash` 與 `/ws/install/setup.bash`。已經進去的 shell 手動 source 這兩個檔也行。

repo 以 bind mount 掛在 `/ws/src/TDK-straw`，跟 build 時同一路徑，`--symlink-install` 之後改 python 檔不必重 build。容器以主機相同的 UID 執行，`output/` 等產生的檔案不會變成 root 擁有。

### 啟動

平常只需要兩個終端。每個終端都先 `docker compose exec straw /entrypoint.sh bash` 進容器（容器沒在跑就先 `docker compose up -d straw`，見上一節）。

```bash
# 終端 1：一行起完整 pipeline：realsense2_camera + straw_node
ros2 launch straw_detector straw_with_camera.launch.py
# 要看預覽視窗（主機先 xhost +local:docker）
ros2 launch straw_detector straw_with_camera.launch.py publish_annotated:=true

# 終端 2：看結果（network_mode: host，主機端有裝 ROS2 的話也能直接下）
ros2 topic echo /straw/target
```

`straw_with_camera.launch.py` 會帶 `align_depth.enable:=true`（節點吃的是 `aligned_depth_to_color`，沒開的話那個話題不存在，會靜默地一則都收不到）與 `rgb_profile:=1280x720x30`（對上校正檔的 `image_width`），再 include `straw_detector.launch.py` 起偵測節點。

**`straw_detector.launch.py` 是給相機已經另外起了的情況**，例如排查問題時手動跑了 `rs_launch.py`，這時只補偵測節點：

```bash
ros2 launch straw_detector straw_detector.launch.py
```

它和 `straw_with_camera.launch.py` 是二選一。若前者已在跑又下這條，會多出第二個 `straw_detector` 節點，兩個一起往 `/straw/target` 發，每格出現兩次。

| 情況 | 做法 |
|---|---|
| 舊版 realsense-ros，`ros2 topic list` 只有一層 `/camera/...` | 任一條 launch 加 `camera_namespace:=/camera` |
| 單獨起相機排查 | `ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true rgb_camera.color_profile:=1280x720x30` |
| 不經 ROS 直接測相機 | `python3 realsense_test.py`，看到 `USB 3.2`、`30.0 fps` 就正常，按 ESC 離開 |

起不來時先看：相機是否插在 USB 3 埠（log 出現 `Couldn't resolve requests` 就是 USB 2）、有沒有別的程式還佔著相機（`realsense_test.py`、`straw.py --realsense`、另一個 launch）。再深入走〈測試 RealSense 節點〉的四關。

相機硬體只能被一個程式開啟，那個程式就是 `realsense2_camera`；它發出的話題要幾個節點訂閱都行。要再加訂閱影像的節點，QoS 一樣得用 `qos_profile_sensor_data`（見〈兩個容易踩的坑〉），且每個節點各自解碼一份 1280x720@30，CPU 是加總的 —— 只需要偵測結果的節點請訂閱 `/straw/target`，不要各自重跑偵測。

參數集中在 `config/straw_detector.yaml`，launch 檔的 `params_file` 可換一份。話題名由 `camera_namespace` 組出，不必三個分別覆寫。

### 訊息：`straw_interfaces/msg/StrawTarget`

欄位與 `--emit-json` 的 JSON 一一對應，`header` 沿用來源影像的 header（stamp 與 frame_id）供控制端對時。

| 欄位 | 說明 |
|---|---|
| `valid` | `false` 時以下欄位皆不可信，控制端應維持前一指令或停止 |
| `reason` | `valid=false` 的原因：`no_detection` 或 `low_confidence` |
| `heading_error_deg` | 角度誤差，正值代表目標頂端偏向畫面右側 |
| `lateral_error_px` / `lateral_error_ratio` | 橫向誤差，像素與半畫面寬單位 |
| `angle_deg` / `confidence` / `angle_sigma_deg` | 目標長軸角度、可信度、角度標準誤 |
| `center_px` / `image_size` | 目標中心與來源影像大小 |
| `has_depth` | `false` 時以下深度欄位皆為 0 |
| `distance_m` / `lateral_error_m` | 目標距離、相對機器人瞄準軸的橫向位移（已補償安裝偏差） |
| `camera_lateral_m` / `position_m` | 未補償的原始相機座標，校正與除錯用 |

`low_confidence` 時 `confidence` 仍會填，其餘為 0。

### 節點參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `image_topic` / `depth_topic` / `camera_info_topic` | 由 launch 依 `camera_namespace` 組出 | 來源話題 |
| `use_depth` | `true` | 是否訂閱深度並換算公尺座標 |
| `depth_scale` | `0.001` | 深度單位換算（RealSense 為公釐） |
| `depth_patch_radius` | `6` | 深度取樣的鄰域半徑（像素） |
| `sync_slop` | `0.05` | 彩色與深度的時間同步容許誤差（秒） |
| `target_topic` | `straw/target` | 結果話題 |
| `annotated_topic` | `straw/annotated` | 標註影像話題，供 rviz 除錯 |
| `publish_annotated` | `false` | 是否發佈標註影像（關閉時會跳過繪圖） |
| `min_confidence` | `0.5` | 低於此值視同沒有偵測到 |
| `filter_alpha` | `0.25` | 跨影格軸線平滑係數 |
| `use_calibration` | `true` | 關掉就不載入任何校正檔 |
| `calibration_file` | `""` | 空字串代表 package 內 `config/axis_calibration.json` |
| `lower_hsv` / `upper_hsv` / `kernel_size` / `min_area` | 同命令列 | 偵測參數 |
| `robot_angle` / `axis_offset_m` / `axis_offset_ratio` / `axis_yaw_deg` | 未設 | 安裝偏差；有設才蓋過校正檔，沒設一律以校正檔為準 |

### 兩個容易踩的坑

**QoS 必須用 sensor data。** RealSense 以 `SensorDataQoS`（best effort）發佈影像；訂閱端若用 rclpy 預設的 reliable QoS，兩邊不相容，會**一則訊息都收不到，而且不會報錯**。本節點已使用 `qos_profile_sensor_data`。

**話題名稱依 realsense-ros 版本而異。** 較新版本是 `/camera/camera/color/image_raw`（兩層 camera 命名空間），舊版是 `/camera/color/image_raw`。先用 `ros2 topic list` 確認，再用 `camera_namespace` 覆寫。

### 測試 RealSense 節點

由淺到深分四關，每一關過了再往下，出問題才知道卡在哪一層。以下指令都在容器內（或裝好 ROS2 的主機）執行。**相機同時只能被一個程式佔用**，每一關開始前要先把上一關的程式關掉。

**第 1 關：不經 ROS，確認硬體出得了格。**

```bash
python3 realsense_test.py
```

要看到 `以 USB 3.2 連線`、`1280x720@30 彩色+深度：可用`，視窗左上 `30.0 fps  dropped 0`。印出 `USB 2.x` 就換埠或換線。看完**按 ESC 離開**，不要 Ctrl-C，讓它正常走 `pipeline.stop()`。

**第 2 關：單獨起 realsense node。**

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true rgb_camera.color_profile:=1280x720x30
```

log 要有 `Device USB type: 3.2`、`Open profile: ... Color ... 1280x720 ... 30`、`RealSense Node Is Up!`。

- `Couldn't resolve requests`：USB 2 模式撐不起 1280x720。
- `No RealSense devices were found`：第 1 關的程式還開著，或容器沒掛 `/dev`。
- 一次性的 `Hardware Notification: Depth stream start failure`：深度模組卡在上一次沒乾淨收掉的狀態，SDK 通常會自己重試成功，用第 3 關的 `hz` 確認深度有在出即可；沒有的話加 `initial_reset:=true` 重起，再不行就實體重插相機。

**第 3 關：確認話題真的有資料在流。** 另開一個終端：

```bash
ros2 topic list | grep camera        # 確認命名空間是 /camera/camera 還是 /camera
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/camera/color/camera_info --once
```

兩個 `hz` 都要穩在約 30 Hz（剛啟動的前幾秒偏低是正常的）。`camera_info` 要是 `width: 1280`、`height: 720`，`k` 的 fx/fy 約 910。

- `aligned_depth_to_color` 不存在：`align_depth.enable` 沒帶到。節點訂的就是這個話題，沒有它會靜默收不到任何東西。
- 話題列得出來但 `hz` 沒數字：QoS 不合，加 `--qos-reliability best_effort`。

**第 4 關：接上 straw_node 做端到端。** 把第 2 關的 launch 關掉，改用專案的一行 launch：

```bash
ros2 launch straw_detector straw_with_camera.launch.py publish_annotated:=true
ros2 topic echo /straw/target          # 另一個終端
```

鏡頭前放目標物，要看到 `valid: true`、`has_depth: true`、`distance_m` 接近實際距離；拿開後變 `valid: false`、`reason: no_detection`。

- `/straw/target` 一則都沒有：`aligned_depth_to_color` 沒出來（回第 3 關），或 `camera_namespace` 跟實際話題名不符。
- `valid: true` 但 `has_depth: false`：訊息有發代表彩色與深度都有同步收到，剩下兩種可能 —— log 沒有 `取得內參 fx=...`（`camera_info` 話題名不對），或目標中心鄰域的深度全是 0。後者最常見的原因是**目標離相機不到 0.3 m**（D435 深度的最短量程），其次是螢幕、照片、光滑反光面或無紋理平面。

深度為什麼取不到，用 CLI 看最快（它與節點共用同一套 `build_depth_fields`，判斷完全一致）：關掉 ROS 端的 launch，跑 `python3 straw.py --realsense`，終端每格印一行，有 `距離: 0.812 m` 就是深度取到了。一邊移動目標一邊看距離什麼時候出現，就能分辨是太近還是表面問題。

---

## 命令列參數

**輸入**

| 參數 | 預設 | 說明 |
|---|---|---|
| `--image` | `IMG_3231.JPEG`（不存在） | 輸入圖片路徑 |
| `--mask` | 無 | 外部二值 mask，指定後略過 HSV 建立（僅圖片模式） |
| `--video` | 無 | 輸入影片路徑 |
| `--camera` | 無 | 相機裝置編號，不帶數字為 0 |
| `--realsense` | 關 | 以 pyrealsense2 讀取 RealSense |
| `--realsense-size` | `1280 720` | RealSense 串流解析度 |
| `--realsense-fps` | `30` | RealSense 串流影格率 |
| `--no-realsense-depth` | 關 | 不啟用深度串流 |
| `--no-frame-drop` | 關 | 相機來源不丟格（延遲會累積，僅供除錯） |

**輸出**

| 參數 | 預設 | 說明 |
|---|---|---|
| `--robot` | 關 | 機器運行模式 |
| `--emit-json` | 關 | 完整 JSON Lines 輸出 |
| `--min-confidence` | 機器模式 0.5，其餘不過濾 | 低於此可信度視同沒有偵測到 |
| `--output` | `output/straw_detection.png` | 標註輸出路徑；影片/相機模式自動改為 `.mp4` |
| `--no-display` | 關 | 不開預覽視窗 |
| `--no-save` | 關 | 不儲存輸出 |
| `--save-masks` | 關 | 額外輸出目標 mask 與完整 HSV mask（僅圖片模式） |

**偵測**

| 參數 | 預設 | 說明 |
|---|---|---|
| `--lower-hsv` | `16 84 42` | HSV 下界 |
| `--upper-hsv` | `33 255 255` | HSV 上界 |
| `--kernel-size` | `11` | 形態學核心大小 |
| `--min-area` | `5000` | 有效連通區的最小像素面積 |

**安裝偏差**（見〈相機不在中軸線上〉）

| 參數 | 預設 | 說明 |
|---|---|---|
| `--robot-angle` | `90.0` | 機器人前進方向在影像座標中的角度 |
| `--axis-offset-m` | `0.0` | 相機相對瞄準軸的側偏（公尺，正為右），有深度時使用 |
| `--axis-offset-ratio` | `0.0` | 同上，以半畫面寬為單位，沒有深度時使用 |
| `--axis-yaw-deg` | `0.0` | 相機偏航（度，正為朝右） |
| `--calibration` | 自動找 `axis_calibration.json` | 讀入校正檔；個別參數若明確指定則優先 |
| `--no-calibration` | 關 | 不要自動載入預設校正檔 |
| `--calibrate` | 無 | 進入校正模式並把結果寫到這個路徑 |
| `--calibrate-seconds` | `1.0` | 影片來源取最後幾秒 |
| `--calibrate-frames` | `30` | 即時來源要收集幾格 |

---

## 演算法

```
輸入影格
  └─ HSV 色彩門檻 (--lower-hsv / --upper-hsv)
  └─ 形態學開閉運算去雜訊、補洞 (--kernel-size)
  └─ 取最大連通區當作目標 (--min-area)
  └─ 取輪廓，剔除貼齊畫面邊界的點
  └─ 全前景像素做 PCA → 主軸、次軸兩個候選種子方向
  └─ 各自擬合左右側邊，取直線段佔比高者
  └─ 左右側邊方向向量相加 → 長軸角度
  └─ 跨影格軸線平滑 (AxisAngleFilter)
  └─ 角度誤差 / 橫向誤差 / 可信度
```

### 關鍵設計

**側邊向量相加，而非單邊或外接矩形。** 透視會讓稻草捆的矩形投影成梯形，兩側邊各自傾斜的方向相反，相加可抵消大部分透視偏移。

**剔除貼齊畫面邊界的輪廓點。** 目標超出畫面時，輪廓會沿影像邊界走一整段完美直線 —— 那是裁切痕跡不是稻草邊緣，而 RANSAC 天生偏好這種零殘差的直線，不剔除就會擬出一條沿著畫面邊緣的假側邊。剔除範圍由 `DEFAULT_BORDER_MARGIN`（2 px）控制。

**取最長連續內點區段，而非全部內點。** 圓柱體兩端是圓弧，位於邊界點序列的頭尾。圓弧會持續偏離直線，不可能落在連續內點區段內，因此自然被截掉；只取內點則可能讓兩端圓弧「擦過」直線，形成中間夾著圓弧的假直邊。

**側邊夾角過大就否決。** 目標只露出一小截時，兩條線可能都擬在同一段端點圓弧上，成為該弧的兩條切線 —— 此時夾角會遠大於透視造成的收斂。實測正常情形不超過 30 度，因此超過 `DEFAULT_MAX_SIDE_ANGLE_DIFFERENCE`（45 度）就判定該種子方向失敗，改用另一個候選。這與「可信度不看夾角」不衝突：守門是物理合理性的硬性否決，可信度則是正常範圍內的漸進評分。

**種子方向會反覆修正。** 種子只用來決定分箱的縱向座標。目標只露出一小截時 mask 接近方形，PCA 主軸可能偏離真實長軸數十度，導致左右邊界都落在同一側；用上一輪算出的方向重新分箱可以逐步修正（`data/0908.png`：44.7 → 74.5 度，可信度 0.07 → 0.64）。直線段佔比已達 `DEFAULT_STRAIGHT_TARGET` 就提早停止。

**主軸與次軸都試一次。** 目標接近方形時 PCA 分不出長短軸，主軸可能剛好指向側邊的垂直方向，導致側邊擬成上下兩端（90 度翻轉）。兩個方向各擬合一次取較佳者，並用兩者的分數差距當作可信度的一項證據。

### 標註圖層

| 顏色 | 意義 |
|---|---|
| 灰階底圖 | 形態學清理後的完整 HSV mask |
| 橘色區塊 | 被選為目標的連通區 |
| 紅色線段 | 擬合出的左右兩條側邊 |
| 紅色圓點 | 估計的目標中心 |
| **橘色**雙向箭頭 | 長軸方向，且**可信度通過門檻** —— 這一格會被控制端採用 |
| **灰色**雙向箭頭 | 長軸方向，但可信度未達門檻 —— 這一格會被丟棄 |
| 綠色斜線 | 機器人瞄準軸；橫線標出目標中心離它的位移。斜率來自 `robot_angle`，未校正時為垂直 |
| 細灰色垂直線 | 畫面中線，只在瞄準軸被安裝偏差移開或傾斜時才畫，用來看補償了多少 |

軸線兩端無正反之分，故兩端皆有箭頭。判定用的門檻取自 `--min-confidence`；該值為 0（非機器運行模式的預設）時標註圖改用 `DEFAULT_MIN_CONFIDENCE`（0.5），否則每一格都會顯示通過而失去意義。

單張圖片模式另外會印出診斷用的中間量：目標面積、兩側邊估計長度、主軸細長比、輪廓貼齊畫面邊界比例、側邊直線段佔比、角度標準誤、可信度三細項。

---

## 效能

1920x1080、`IMG_4215.MOV` 60 格實測（CPU，無 GPU）：

| 階段 | 時間 |
|---|---|
| HSV mask + 形態學 | 9.5 ms |
| 連通區選取 | 9.2 ms |
| `analyze_target` | 18.4 ms |
| 繪圖 | 11.5 ms |
| **整體** | **46~49 ms（約 21 fps）** |
| **`--robot`（不繪圖）** | **34.0 ms（29.4 fps）** |

**降低輸入解析度幾乎沒有幫助。** 只有前兩個階段隨像素數縮放；側邊擬合處理的是 40 個切片點，與解析度無關，640x360 只快約 1.4 倍。要省時間就關繪圖（`--robot`）。

---

## 已知限制

目標貼齊畫面邊界時仍未處理完的部分（`IMG_4215.MOV` 全片 641 格實測）：

- **目標選取仍以純面積排序。** 有 70 格的最大連通區是被畫面切掉的碎片（例如 230k px 的角落三角形），而畫面中另有完整可見的稻草捆（67k px）被忽略。改進方向是把排序改成 `面積 × 完整度權重`。
- **沒有單邊退化模式。** 目前只要有一側擬合失敗就整格放棄。實際上單側仍可提供方向（可信度打折），只是中心不可求。
- **種子方向未利用時間資訊。** 截斷時 PCA 會亂跳，可改用上一格的平滑角度當種子（並在與 PCA 差距過大時回退，避免濾波器鎖死在錯誤角度）。
- **低可信度的影格仍會進入軸線濾波器。** `AxisAngleFilter` 不會拒絕錯誤的偵測，只會把它慢慢混進輸出。可信度門檻目前只擋輸出，不擋濾波器更新。

安裝偏差校正的部分（見〈相機不在中軸線上〉）：

- **橫向未對準時角度誤差有系統性偏差。** `robot_angle` 是單一常數，但透視下「應有的角度」取決於目標在畫面上的位置，橫向每偏 100 px 約差 3.4 度。橫向對準後歸零，所以先收橫向就避得開；要根治得改存消失點、每格即時算應有角度。
- **偏航未校正。** 單一距離上偏航與側偏完全簡併，`--axis-yaw-deg` 預設 0，殘留偏差 `(Z − Z_ref)·tan(yaw)`，同樣在校正距離歸零。目前要靠手算填入。
- **`axis_offset_ratio` 綁在校正時的視角上。** 換長寬比就失準，只能靠 `image_width` 不符時警告，不會自動換算。`axis_offset_m` 不受影響。

## 檔案

- `straw.py` — 轉呼叫 `ros2/straw_detector/straw_detector/straw.py`，讓沒裝 ROS2 也能直接跑
- `ros2/straw_detector/` — ROS2 package：`straw_detector/straw.py`（偵測邏輯與命令列介面的正本）、`straw_detector/detector_node.py`（ROS2 節點）、`launch/`、`config/`（參數 YAML 與校正檔）
- `ros2/straw_interfaces/` — ROS2 package：`StrawTarget.msg`
- `realsense_test.py` — 以 pyrealsense2 直連相機的串流測試，用來確認硬體正常；`--record` 可錄下素材
- `Dockerfile`、`docker-compose.yml`、`docker/entrypoint.sh` — ROS2 Humble 執行環境，主機不必裝 ROS2，見〈用 Docker 跑〉
- `data/` — 測試素材（影片檔不進版控）
- `output/` — 程式產生的標註結果，不進版控
