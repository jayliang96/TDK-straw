from glob import glob

from setuptools import setup

package_name = "straw_detector"

setup(
	name=package_name,
	version="0.1.0",
	packages=[package_name],
	data_files=[
		("share/ament_index/resource_index/packages",
		 ["resource/" + package_name]),
		("share/" + package_name, ["package.xml"]),
		("share/" + package_name + "/launch", glob("launch/*.launch.py")),
		# 校正檔一起安裝，節點才有穩定的預設路徑可以自動載入。
		("share/" + package_name + "/config", glob("config/*")),
	],
	install_requires=["setuptools"],
	zip_safe=True,
	maintainer="yusheng",
	maintainer_email="qqja3710859@gmail.com",
	description="從 RealSense 影像偵測稻草捆，發佈機器人對準所需的角度與橫向誤差",
	license="MIT",
	entry_points={
		"console_scripts": [
			"straw_node = straw_detector.detector_node:main",
			"straw_detect = straw_detector.straw:main",
		],
	},
)
