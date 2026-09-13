"""轉呼叫 ros2/straw_detector/straw_detector/straw.py。

真正的程式碼住在 ROS2 package 裡（節點要能 import 它），這支只是讓
`python straw.py --image ...` 在 repo 根目錄照舊可用，不必裝 ROS2。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "ros2" / "straw_detector"))

from straw_detector.straw import main  # noqa: E402

if __name__ == "__main__":
	main()
