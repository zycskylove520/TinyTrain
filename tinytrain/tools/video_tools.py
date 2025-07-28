import os
import cv2

from datetime import datetime

from tinytrain.utils.any_utils import generate_unique_id

def video_to_images(video_file, image_folder, start_time, end_time, fps):
    """
    将视频裁剪为图片。
    :param video_file: 输入视频文件路径
    :param image_folder: 输出图片保存的文件夹路径
    :param start_time: 起始时间（小时:分钟:秒）
    :param end_time: 结束时间（小时:分钟:秒）
    :param fps: 每秒保存图片的帧率（整数表示每秒切n张图，小于1表示n秒切1张图）

    Examples:
    # 每秒切2张图
    video_to_images("input.mp4", "output_images", "00:00:10", "00:00:30", 2)

    # 每5秒切1张图
    video_to_images("input.mp4", "output_images", "00:00:10", "00:00:30", 0.2)
    """

    # 将时间字符串转换为秒
    def time_to_seconds(time_str):
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s

    start_time_seconds = time_to_seconds(start_time)
    end_time_seconds = time_to_seconds(end_time)

    # 检查输入文件是否存在
    if not os.path.exists(video_file):
        print(f"输入文件 {video_file} 不存在！")
        return

    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_file)

    # 获取视频的帧率和总帧数
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 检查 fps 是否超出视频帧率限制
    if fps > video_fps:
        print(f"指定的 fps ({fps}) 超出视频帧率 ({video_fps})，已调整为视频的最大帧率。")
        fps = video_fps

    # 计算起始帧和结束帧
    start_frame = int(start_time_seconds * video_fps)
    end_frame = int(end_time_seconds * video_fps)

    # 检查起始帧和结束帧是否在视频范围内
    if start_frame < 0 or end_frame > total_frames:
        print("起始时间或结束时间超出视频范围！")
        cap.release()
        return

    # 计算每秒保存图片的间隔帧数
    if fps < 1:
        interval_frames = int(1 / fps * video_fps)
    else:
        interval_frames = math.ceil(video_fps / fps)

    # 计算总共需要切多少张图片
    total_images = (end_frame - start_frame) // interval_frames + 1
    print(f"总共需要切 {total_images} 张图片。")

    # 初始化帧计数器和图片计数器
    frame_count = 0
    image_count = 0

    # 跳转到起始时间
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 逐帧读取视频
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 如果当前帧在裁剪范围内
        if frame_count + start_frame <= end_frame:
            # 每隔 interval_frames 帧保存一次图片
            if frame_count % interval_frames == 0:
                # 获取当前时间戳
                timestamp = datetime.now()
                # 生成原始的图片文件名
                original_output_path = os.path.join(image_folder, f"frame_{frame_count + start_frame:06d}")
                # 使用原始文件名和时间戳生成唯一的文件名
                unique_id = generate_unique_id(original_output_path, timestamp)
                # 最终的输出路径
                output_path = os.path.join(image_folder, f"{unique_id}.jpg")
                cv2.imwrite(output_path, frame)
                image_count += 1
                # 每次保存图片时打印总图片数和当前图片数
                print(f"正在处理第 {image_count} 张图片，共 {total_images} 张，保存到 {output_path}")
        frame_count += 1

        # 如果已经处理到结束帧，退出循环
        if frame_count + start_frame > end_frame:
            break

    # 释放视频文件
    cap.release()
    print(f"视频裁剪为图片完成，已保存到 {image_folder}")


def images_to_video(image_folder, output_video_path, fps=30):
    """
    将图片文件夹中的图片合成成一个视频。

    :param image_folder: 包含图片的文件夹路径
    :param output_video_path: 输出视频的路径
    :param fps: 输出视频的帧率，默认为30fps
    """
    # 获取文件夹中的所有图片文件
    images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png", ".jpeg"))]
    images.sort()  # 确保图片按顺序排序

    if not images:
        print("图片文件夹为空或没有找到图片文件！")
        return

    # 获取第一张图片的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 也可以选择其他编码格式，如"XVID"
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print("无法创建视频文件")
        return

    # 遍历图片文件夹，将图片写入视频
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法读取图片: {image_path}")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"视频合成完成！已保存到: {output_video_path}")

def extract_frames(video_path, output_folder, save_every_nth_frame=1):
    """
    将视频按帧提取为图片并保存到指定文件夹。

    参数:
        video_path (str): 输入视频文件路径
        output_folder (str): 输出图片保存目录（自动创建）
        save_every_nth_frame (int): 每多少帧保存一次，默认1（全部保存）

    使用示例:
        >>> extract_frames(
        ...     video_path='./video.mp4',
        ...     output_folder='./frames',
        ...     save_every_nth_frame=10
        ... )
    """
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频的帧率和总帧数
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频帧率: {video_fps} fps, 总帧数: {total_frames}")

    # 初始化变量
    frame_count = 0
    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 如果读取失败，退出循环

        # 每隔save_every_nth_frame帧保存一次
        if frame_count % save_every_nth_frame == 0:
            output_path = os.path.join(output_folder, f"frame_{save_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"保存图片到: {output_path}")
            save_count += 1

        frame_count += 1

    cap.release()
    print(f"完成！共保存了 {save_count} 张图片到 {output_folder}")

# 示例用法
if __name__ == "__main__":
    input_video_path = r"E:\DownLoad\yanhuachang\视频\开关门\20250717\00000006589000000.mp4"  # 输入视频文件路径
    output_folder = r"E:\DownLoad\yanhuachang\img"  # 输出图片文件夹路径
    start_time = "00:00:00"  # 起始时间（秒）
    end_time = "01:00:00"  # 结束时间（秒）
    fps = 0.02  # 每秒保存图片的帧率

    video_to_images(input_video_path, output_folder, start_time, end_time, fps)
