"""RealSense 串流測試：確認硬體正常，並量出實際的影格率與掉格數。

畫面會卡的原因通常不在這支程式的處理速度（上色 + 併圖只要 1~2 ms），
而在相機端根本沒把影格送上來。實際量到的元凶是 USB 只跑在 2.x：D435
的 640x480@30 彩色與深度各約 18 MB/s，兩條加起來超過 USB 2.0 的實用
頻寬，結果不是變慢而是一格都收不到 —— pipeline.start() 會成功，
wait_for_frames() 永遠逾時，嚴重時裝置直接從匯流排上掉線
（HRESULT 0x8007001F）。換到 USB 3.2 之後同一組設定實測穩定 30.0 fps。

反過來，有兩件事量過但沒有影響，別再往那邊找：sensor 的
frames_queue_size（rs.pipeline 本身只保留最新的 frameset，改成 1 對
延遲與掉格都沒有差別），以及各種後處理濾波。

RGB 的 auto_exposure_priority 預設開啟，光線一暗驅動會自己把彩色降到
6~15 fps 來換曝光時間，看起來也像卡頓，這裡關掉；此次未在暗處實測。

--record 會把這次串流錄下來當測試素材。副檔名決定錄什麼：.mp4/.avi 只
錄彩色（可直接餵給 straw.py --video），.bag 是 SDK 原生格式，彩色與深度
都留得住，但一分鐘就好幾百 MB。
"""

import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs

# 依頻寬由高到低排列。USB 2.x 撐不住第一組，往下退到能實際出格的組合。
# 格式為 (說明, 彩色 (寬, 高, fps), 深度 (寬, 高, fps) 或 None)。
STREAM_CANDIDATES = [
    ("640x480@30 彩色+深度", (640, 480, 30), (640, 480, 30)),
    ("640x480@15 彩色+深度", (640, 480, 15), (640, 480, 15)),
    ("424x240@30 彩色 + 480x270@30 深度", (424, 240, 30), (480, 270, 30)),
    ("640x480@30 僅彩色", (640, 480, 30), None),
]

# 錄影支援的副檔名。.bag 交給 SDK 錄，其餘走 cv2.VideoWriter。
RECORD_SUFFIXES = (".mp4", ".avi", ".bag")


def describe_device():
    """回報連線型態；USB 2.x 是畫面卡頓最常見的根因。"""
    try:
        devices = list(rs.context().query_devices())
    except RuntimeError as error:
        return None, "無法列舉裝置(%s)。請重新插拔相機。" % error
    if not devices:
        return None, "找不到 RealSense 裝置。"

    usb = devices[0].get_info(rs.camera_info.usb_type_descriptor)
    name = devices[0].get_info(rs.camera_info.name)
    if usb.startswith("2"):
        return usb, (
            "%s 目前是 USB %s 連線。此頻寬吃不下 640x480@30 的彩色+深度，"
            "會掉格甚至讓裝置掉線；請改插主機板上的 USB 3 埠(藍色/SS)，"
            "並確認用的是相機原廠的 USB 3 線、中間沒有接 USB 2 集線器。"
            % (name, usb)
        )
    return usb, "%s 以 USB %s 連線。" % (name, usb)


def colorize_depth(depth_image, near_mm, far_mm):
    """把深度攤在實際工作距離上，無資料塗黑。

    常見的 alpha=0.03 等於把 0~8.5m 攤在整條色階上，但實測場景的深度
    99.9% 落在 3.5m 內，七成的顏色預算浪費在根本量不到的距離上，近處
    全擠在藍色端而分不出層次。改成只攤工作距離，對比才夠。

    另外 0 代表立體匹配失敗(沒資料)，不是「很近」。JET 會把它畫成深藍，
    跟真的很近的像素混在一起；塗黑之後破洞與近處一眼就分得開。
    """
    span = max(far_mm - near_mm, 1)
    scaled = (depth_image.astype(np.float32) - near_mm) * (255.0 / span)
    colored = cv2.applyColorMap(
        np.clip(scaled, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    colored[depth_image == 0] = 0
    return colored


def tune_sensors(profile, keep_fps):
    """不讓驅動為了曝光而偷偷降影格率。"""
    for sensor in profile.get_device().sensors:
        if keep_fps and sensor.supports(rs.option.auto_exposure_priority):
            # 關掉之後驅動就不會為了拉長曝光而偷偷降低影格率。
            sensor.set_option(rs.option.auto_exposure_priority, 0)


def start_stream(color, depth, keep_fps, record_bag=None, probe_ms=2000):
    """啟動串流，並確認真的收得到影格。

    在 USB 2.x 上 pipeline.start() 會成功、wait_for_frames() 卻永遠等不到
    東西，所以「能不能開起來」不算數，要實際拿到一格才算數。
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, color[0], color[1],
                         rs.format.bgr8, color[2])
    if depth:
        config.enable_stream(rs.stream.depth, depth[0], depth[1],
                             rs.format.z16, depth[2])
    if record_bag is not None:
        # 錄 bag 由 SDK 在 pipeline 內部完成，必須在 start() 之前掛上；
        # 寫進檔案的是原始影格，不受下面的上色與時間濾波影響。
        config.enable_record_to_file(str(record_bag))

    profile = pipeline.start(config)
    tune_sensors(profile, keep_fps)
    try:
        pipeline.wait_for_frames(probe_ms)
    except RuntimeError:
        pipeline.stop()
        raise
    return pipeline


def open_camera(args, record_bag=None):
    """依序嘗試候選組合，回傳第一個真的出得了格的串流。

    一併回傳選中的彩色規格：錄影要用實際談成的解析度與影格率開檔，退到
    後備組合時若還照 640x480@30 開，寫出來的檔案是壞的。
    """
    candidates = STREAM_CANDIDATES
    if args.no_depth:
        candidates = [(label, color, None) for label, color, _ in candidates]

    for label, color, depth in candidates:
        try:
            pipeline = start_stream(
                color, depth, not args.keep_auto_exposure_priority, record_bag)
        except RuntimeError as error:
            print("  %s：不可用(%s)" % (label, error))
            continue
        print("  %s：可用" % label)
        return pipeline, label, color
    raise RuntimeError(
        "所有組合都拿不到影格。請重新插拔相機，並確認沒有其他程式"
        "(straw.py 或另一個 realsense_test.py)正佔用它。"
    )


def open_recorder(record_path, color):
    """開好彩色影片的寫檔器；.bag 由 SDK 自己錄，不走這裡。

    影格率用相機談成的值，不用畫面上那個實測值 —— 實測值要跑滿一秒才有，
    開檔的當下還沒有。代價是掉格時錄出來的片長會比實際短，掉了多少畫面
    上的 dropped 有寫。
    """
    record_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(record_path), fourcc, float(color[2]), (color[0], color[1])
    )
    if not writer.isOpened():
        raise RuntimeError("無法開啟 %s 寫檔，請確認路徑存在且副檔名支援。"
                           % record_path)
    return writer


def main():
    parser = argparse.ArgumentParser(description="RealSense 串流測試")
    parser.add_argument("--no-depth", action="store_true",
                        help="只看彩色，頻寬減半")
    parser.add_argument("--keep-auto-exposure-priority", action="store_true",
                        help="保留自動曝光優先，暗處畫質較好但影格率會掉")
    parser.add_argument("--depth-range", type=float, nargs=2,
                        metavar=("NEAR", "FAR"), default=(0.2, 2.0),
                        help="深度上色的工作距離，單位公尺，預設 0.2 2.0")
    parser.add_argument("--no-temporal-filter", action="store_true",
                        help="關掉時間濾波，可以看到未經平滑的原始雜點")
    parser.add_argument("--record", metavar="PATH",
                        help="把這次串流錄下來；.mp4/.avi 只錄彩色，"
                             "可直接餵給 straw.py --video，.bag 連深度一起"
                             "錄但檔案大得多")
    args = parser.parse_args()

    near_mm, far_mm = (round(v * 1000.0) for v in args.depth_range)

    record_path = Path(args.record) if args.record else None
    record_bag = None
    if record_path is not None:
        suffix = record_path.suffix.lower()
        if suffix not in RECORD_SUFFIXES:
            parser.error("--record 的副檔名只能是 %s，收到 %r。"
                         % ("/".join(RECORD_SUFFIXES), record_path.suffix))
        if suffix == ".bag":
            # bag 要在 pipeline 起來之前就掛上，不能等拿到影格才決定。
            record_bag = record_path
            record_path.parent.mkdir(parents=True, exist_ok=True)

    usb, message = describe_device()
    print(message)
    if usb is None:
        return

    print("尋找可用的串流組合：")
    pipeline, label, color = open_camera(args, record_bag)
    print("使用 %s，ESC 離開。" % label)
    print("深度上色範圍 %d~%dmm，黑色代表沒有深度資料。" % (near_mm, far_mm))

    writer = None
    if record_path is not None and record_bag is None:
        writer = open_recorder(record_path, color)
    if record_path is not None:
        print("錄影中：%s（%s）。按 ESC 停止並收檔。"
              % (record_path, "彩色+深度 bag" if record_bag else "彩色影片"))

    # 時間濾波拿前幾格做加權，實測可以把破洞從 22.4% 降到 21.0%、有效值的
    # 逐格抖動壓掉約 1.6mm。它擋不掉邊緣的假距離(紅點)：預設 delta 是 20mm，
    # 假值跟歷史值差好幾公尺，會被當成真實變化而放行。spatial 與 median 也
    # 一樣擋不住 —— 那些假值是 3~6px 的連通小塊，濾波器分不出真假。
    temporal = None if args.no_temporal_filter else rs.temporal_filter()

    # 顯示最近一秒的實測值，卡頓與否用數字判斷，不靠感覺。
    #
    # 這裡看掉格而不是延遲：影格編號的斷號是掉格的直接證據，實測消費端
    # 放慢到 100ms/格時會從 0 跳到 78。而 now - frame.get_timestamp() 就算
    # 在那麼卡的情況下也只有 8ms（時間戳的 domain 是 system_time，記的是
    # 抵達主機的時刻，不含感光到傳輸的那一段），拿來當延遲指標會騙人。
    shown = 0
    window_start = time.perf_counter()
    fps = 0.0
    dropped = 0
    previous_number = None
    recorded = 0
    record_start = time.perf_counter()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            number = color_frame.get_frame_number()
            if previous_number is not None:
                dropped += number - previous_number - 1
            previous_number = number
            images = np.asanyarray(color_frame.get_data())

            if record_path is not None:
                recorded += 1
            if writer is not None:
                # 存進去的是原始彩色影格：右邊的深度併圖與左上的疊字都是
                # 給人看的，錄進檔案會讓 straw.py 讀到根本不存在的東西。
                writer.write(images)

            depth_frame = frames.get_depth_frame()
            if depth_frame:
                if temporal is not None:
                    depth_frame = temporal.process(depth_frame).as_depth_frame()
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = colorize_depth(depth_image, near_mm, far_mm)
                if depth_colormap.shape[:2] != images.shape[:2]:
                    depth_colormap = cv2.resize(
                        depth_colormap, (images.shape[1], images.shape[0])
                    )
                images = np.hstack((images, depth_colormap))

            shown += 1
            elapsed = time.perf_counter() - window_start
            if elapsed >= 1.0:
                fps = shown / elapsed
                shown = 0
                window_start = time.perf_counter()

            cv2.putText(
                images, "%.1f fps  dropped %d  USB %s" % (fps, dropped, usb),
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            if record_path is not None:
                cv2.putText(
                    images,
                    "REC %.1fs  %d frames"
                    % (time.perf_counter() - record_start, recorded),
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
            cv2.imshow('RealSense D435', images)
            if cv2.waitKey(1) == 27:  # ESC 離開
                break
    finally:
        # bag 由 pipeline.stop() 收尾，mp4 由 writer.release() 補完索引；
        # 少了任何一邊檔案都會是壞的，所以放在 finally 裡。
        pipeline.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if record_path is not None:
        size_mb = (record_path.stat().st_size / (1024.0 * 1024.0)
                   if record_path.exists() else 0.0)
        print("錄影已儲存：%s（%d 格，%.1f MB，期間掉格 %d）"
              % (record_path, recorded, size_mb, dropped))


if __name__ == "__main__":
    main()
