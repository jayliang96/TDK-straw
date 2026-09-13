# 稻草捆相對機器人的對準誤差。內容與 straw.py --emit-json 的 JSON 一一對應。

std_msgs/Header header        # 沿用來源影像的 stamp 與 frame_id，供控制端對時

bool valid                    # false 時以下欄位皆不可信，控制端應維持前一指令或停止
string reason                 # valid=false 的原因："no_detection" / "low_confidence"

float32 heading_error_deg     # 角度誤差，0 為已對正，正值代表目標頂端偏向畫面右側
float32 lateral_error_px      # 橫向誤差（像素），0 為已對中，正值代表目標在瞄準軸右側
float32 lateral_error_ratio   # 橫向誤差，以半畫面寬為單位，約 [-1, 1]，與解析度無關
float32 angle_deg             # 目標長軸在畫面上的角度
float32 confidence            # 方向可信度 [0, 1]；low_confidence 時仍會填
float32 angle_sigma_deg       # 角度標準誤
float32[2] center_px          # 目標中心像素座標 [x, y]
uint16[2] image_size          # 來源影像大小 [width, height]

bool has_depth                # false 時以下深度欄位皆為 0
float32 distance_m            # 目標中心的深度
float32 lateral_error_m       # 相對機器人瞄準軸的橫向位移（已補償相機安裝偏差）
float32 camera_lateral_m      # 未補償的原始相機座標 x，校正與除錯用
geometry_msgs/Point position_m  # 目標中心在相機座標系的位置：x 右、y 下、z 前，公尺
