"""
Copyright (c) 2025 zycskylove520

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import random
import shutil
import cv2
import chardet

from collections import Counter
from datetime import datetime

from tinytrain.utils.any_utils import generate_unique_id


def rename_and_move_files(source_images_dir, source_labels_dir, target_images_dir, target_labels_dir):
    """
    将图片与对应标签成对重命名为基于 UUID 的新文件名，并移动到新目录。
    仅处理能一一匹配的文件，忽略多余部分。

    Args:
        source_images_dir (str): 原始图片目录
        source_labels_dir (str): 原始标签目录
        target_images_dir (str): 目标图片目录（自动创建）
        target_labels_dir (str): 目标标签目录（自动创建）

    Example:
        >>> rename_and_move_files(
        ...     source_images_dir='./raw/images',
        ...     source_labels_dir='./raw/labels',
        ...     target_images_dir='./renamed/images',
        ...     target_labels_dir='./renamed/labels'
        ... )
    """
    # 创建目标目录
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_labels_dir, exist_ok=True)

    # 获取所有图片和标签文件
    image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    label_files = [f for f in os.listdir(source_labels_dir) if f.endswith('.txt')]

    # 按文件名排序，确保一一对应
    image_files.sort()
    label_files.sort()

    # 找到匹配的部分
    min_length = min(len(image_files), len(label_files))
    print(f"Found {min_length} matching pairs of images and labels.")

    # 获取当前时间
    timestamp = datetime.now()

    # 遍历文件并重命名
    for index in range(min_length):
        image_file = image_files[index]
        label_file = label_files[index]

        # 生成基于文件名和时间戳的UUID
        unique_id = generate_unique_id(image_file, timestamp)

        # 构造新的文件路径
        new_image_path = os.path.join(target_images_dir, f"{unique_id}{os.path.splitext(image_file)[1]}")
        new_label_path = os.path.join(target_labels_dir, f"{unique_id}.txt")

        # 构造原始文件路径
        original_image_path = os.path.join(source_images_dir, image_file)
        original_label_path = os.path.join(source_labels_dir, label_file)

        # 移动并重命名文件
        shutil.copy(original_image_path, new_image_path)
        shutil.copy(original_label_path, new_label_path)

        print(f"Processed: {image_file} -> {new_image_path}")
        print(f"Processed: {label_file} -> {new_label_path}")

    print("Processing complete. Files have been renamed and moved.")


def draw_rect_from_image(img_path, label_path, out_img_path):
    """
    读取单张图片及其 YOLO 格式标签，在图上绘制所有矩形框后保存。

    Args:
        img_path (str): 原始图片路径
        label_path (str): YOLO 标签路径（每行: class cx cy w h，均为归一化值）
        out_img_path (str): 绘制后的图片保存路径

    Example:
        >>> draw_rect_from_image(
        ...     img_path='./images/0001.jpg',
        ...     label_path='./labels/0001.txt',
        ...     out_img_path='./vis/0001_vis.jpg'
        ... )
    """
    # 读取图片
    img = cv2.imread(img_path)
    height = img.shape[0]
    width = img.shape[1]

    # 读取标签
    box_list = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            results = line.strip().split(" ")
            box = []
            box.extend([float(results[1]), float(results[2]), float(results[3]), float(results[4])])
            box_list.append(box)
            print(box)

    # 绘制矩形框
    for box_i in box_list:
        cx = box_i[0] * width
        cy = box_i[1] * height
        b_width = box_i[2] * width
        b_height = box_i[3] * height
        print(f"{cx}, {cy}, {b_width}, {b_height}")

        lx = int(cx - 0.5 * b_width)
        ly = int(cy - 0.5 * b_height)
        rx = int(cx + 0.5 * b_width)
        ry = int(cy + 0.5 * b_height)
        cv2.rectangle(img, [lx, ly], [rx, ry], (0, 255, 0))

    cv2.imwrite(out_img_path, img)


def select_assign_label_img(image_folder, label_folder, new_image_folder, new_label_folder, class_ids):
    """
    筛选出包含指定类别 ID 的图片与标签，复制到新目录，并将类别 ID 重映射为 0 开始的连续索引。

    Args:
        image_folder (str): 原始图片目录
        label_folder (str): 原始标签目录
        new_image_folder (str): 筛选后图片输出目录
        new_label_folder (str): 筛选后标签输出目录
        class_ids (list[int]): 需要保留的原始类别 ID 列表，如 [2,5,7]

    Example:
        >>> select_assign_label_img(
        ...     image_folder='./all/images',
        ...     label_folder='./all/labels',
        ...     new_image_folder='./subset/images',
        ...     new_label_folder='./subset/labels',
        ...     class_ids=[2, 5, 7]
        ... )
    """
    # 创建类别ID到新索引的映射
    class_id_to_new_index = {class_id: new_index for new_index, class_id in enumerate(class_ids)}

    # 确保目标文件夹存在
    os.makedirs(new_image_folder, exist_ok=True)
    os.makedirs(new_label_folder, exist_ok=True)

    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)

    matching_files = {}

    # 遍历图片文件夹，提取文件前缀
    for image_file in image_files:
        prefix = os.path.splitext(image_file)[0]
        matching_files[prefix] = {"image": image_file}

    # 遍历标签文件夹，匹配前缀
    for label_file in label_files:
        prefix = os.path.splitext(label_file)[0]
        if prefix in matching_files:
            matching_files[prefix]["label"] = label_file

    num = 1
    total = len(matching_files)
    for prefix, files in matching_files.items():
        flag = False
        if "image" in files and "label" in files:
            file = os.path.join(label_folder, files["label"])
            new_lines = []
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(" ")
                    if len(parts) == 0:
                        continue  # 跳过空行
                    classes = int(parts[0])
                    if classes in class_id_to_new_index:  # 检查类别ID是否在映射字典的键中
                        flag = True
                        new_index = class_id_to_new_index[classes]  # 获取新的索引
                        new_line = f"{new_index} " + " ".join(parts[1:]) + "\n"
                        new_lines.append(new_line)

            if flag:
                old_image_path = os.path.join(image_folder, files['image'])
                old_label_path = os.path.join(label_folder, files['label'])
                new_image_path = os.path.join(new_image_folder, files['image'])
                new_label_path = os.path.join(new_label_folder, files['label'])

                # 复制图片文件
                shutil.copy(old_image_path, new_image_path)

                # 写入新的标签文件
                with open(new_label_path, 'w') as f:
                    f.writelines(new_lines)

        print(f"一共{total}张，当前完成{num}张")
        num += 1


def remove_extra(img_dir, label_dir):
    """
    删除图片目录中没有对应 txt 标签文件的多余图片。

    Args:
        img_dir (str): 图片目录路径
        label_dir (str): 标签文件目录路径

    Example:
        >>> remove_extra(
        ...     img_dir='./images',
        ...     label_dir='./labels'
        ... )
    """
    # 获取label目录中所有txt文件的文件名（不包含扩展名）
    label_files = set(os.path.splitext(file)[0] for file in os.listdir(label_dir) if file.endswith('.txt'))

    # 遍历img目录中的所有图片文件
    for img_file in os.listdir(img_dir):
        # 获取图片文件的文件名（不包含扩展名）
        img_name = os.path.splitext(img_file)[0]

        # 如果图片文件名不在label文件名集合中，则删除该图片文件
        if img_name not in label_files:
            img_path = os.path.join(img_dir, img_file)
            print(f"删除多余的图片文件: {img_path}")
            os.remove(img_path)

    print("多余的图片文件已处理完毕。")


def split_dataset(image_folder, label_folder, output_folder, split_ratios=(0.70, 0.20, 0.10)):
    """
    将匹配的图片与标签按给定比例随机切分为训练/验证/测试集，并复制到输出目录。

    Args:
        image_folder (str): 图片目录
        label_folder (str): 标签目录
        output_folder (str): 输出根目录（自动创建 img/train、label/train 等子目录）
        split_ratios (tuple[float]): (train, val, test) 比例，默认 (0.7,0.2,0.1)

    Example:
        >>> split_dataset(
        ...     image_folder='./images',
        ...     label_folder='./labels',
        ...     output_folder='./split',
        ...     split_ratios=(0.8, 0.1, 0.1)
        ... )
    """
    # 确保输出目录存在
    os.makedirs(os.path.join(output_folder, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels", "test"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels", "val"), exist_ok=True)

    # 获取所有图片和标签文件
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.txt')])

    # 如果图片数量多余标签数量，将多出来的图片划分给训练集
    if len(image_files) > len(label_files):
        print(f"图片数量多余标签数量，将多出来的 {len(image_files) - len(label_files)} 张图片划分给训练集")
        # 找出没有标签的图片
        unmatched_images = [img for img in image_files if os.path.splitext(img)[0] + ".txt" not in label_files]
        # 将这些图片添加到训练集的图片列表中
        train_images = unmatched_images
        # 剩下的图片和标签是一一匹配的
        matched_images = [img for img in image_files if img not in unmatched_images]
        matched_labels = [os.path.splitext(img)[0] + ".txt" for img in matched_images]
    else:
        # 如果图片数量不多于标签数量，直接使用所有图片和标签
        train_images = []
        matched_images = image_files
        matched_labels = label_files

    # 确保匹配的图片和标签文件数量一致
    if len(matched_images) != len(matched_labels):
        raise ValueError("匹配的图片文件和标签文件数量不一致！")

    # 打乱文件顺序
    combined_files = list(zip(matched_images, matched_labels))
    random.shuffle(combined_files)
    matched_images, matched_labels = zip(*combined_files)

    # 计算切分数量
    total_files = len(matched_images)
    train_end = int(total_files * split_ratios[0])
    val_end = train_end + int(total_files * split_ratios[1])

    # 切分数据集
    train_images += matched_images[:train_end]
    val_images = matched_images[train_end:val_end]
    test_images = matched_images[val_end:]
    train_labels = matched_labels[:train_end]
    val_labels = matched_labels[train_end:val_end]
    test_labels = matched_labels[val_end:]

    # 定义复制函数
    def copy_files(file_list, src_folder, dst_folder):
        for file in file_list:
            src_path = os.path.join(src_folder, file)
            dst_path = os.path.join(dst_folder, file)
            shutil.copy(src_path, dst_path)

    # 复制文件到对应的目录
    print("复制训练集...")
    copy_files(train_images, image_folder, os.path.join(output_folder, "images", "train"))
    copy_files(train_labels, label_folder, os.path.join(output_folder, "labels", "train"))

    print("复制验证集...")
    copy_files(val_images, image_folder, os.path.join(output_folder, "images", "val"))
    copy_files(val_labels, label_folder, os.path.join(output_folder, "labels", "val"))

    print("复制测试集...")
    copy_files(test_images, image_folder, os.path.join(output_folder, "images", "test"))
    copy_files(test_labels, label_folder, os.path.join(output_folder, "labels", "test"))

    print("数据集切分完成！")


def count_label_types(directory):
    """
    统计指定目录下所有 YOLO 标签文件中出现的类别 ID 及其出现次数。

    Args:
        directory (str): 标签文件所在目录

    Example:
        >>> count_label_types('./labels')
    """
    label_counter = Counter()  # 记录每个类别 ID 的出现次数

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {file_path}")

            # 读取文件内容（兼容多种编码）
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='gbk') as file:
                        lines = file.readlines()
                except UnicodeDecodeError:
                    print(f"Warning: Unable to decode {file_path}. Trying to detect encoding...")
                    with open(file_path, 'rb') as file:
                        raw_data = file.read()
                    detected_encoding = chardet.detect(raw_data)['encoding']
                    print(f"Detected encoding: {detected_encoding}")
                    try:
                        with open(file_path, 'r', encoding=detected_encoding) as file:
                            lines = file.readlines()
                    except UnicodeDecodeError:
                        print(f"Error: Still unable to decode {file_path}. Skipping...")
                        continue

            # 提取类别 ID 并计数
            for line in lines:
                parts = line.strip().split()
                if parts:
                    try:
                        label_id = int(parts[0])
                        label_counter[label_id] += 1
                    except ValueError:
                        print(f"Warning: Invalid label in {file_path}: {parts[0]}")
                        continue

    # 输出统计结果
    print("统计结果：")
    print(f"数据集中共有 {len(label_counter)} 个标签类别")
    print("类别 ID | 出现次数")
    for label_id in sorted(label_counter):
        print(f"{label_id:>7} | {label_counter[label_id]}")


def convert_txt_files_to_utf8(directory):
    """
    批量将目录下所有 txt 文件转码为 UTF-8 编码（自动尝试 utf-8、gbk 及 chardet 检测）。

    Args:
        directory (str): 需要处理的目录路径

    Example:
        >>> convert_txt_files_to_utf8('./labels')
    """
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否为 .txt 文件
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {file_path}")

            # 尝试读取文件内容（尝试多种编码）
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='gbk') as file:
                        content = file.read()
                except UnicodeDecodeError:
                    print(f"Warning: Unable to decode {file_path}. Skipping...")
                    continue

            # 将内容以 UTF-8 格式重新保存
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)

            print(f"Converted {file_path} to UTF-8 encoding.")

    print("All .txt files have been converted to UTF-8 encoding.")


def delete_empty_txt_files(directory):
    """
    删除指定目录下所有大小为 0 字节的 txt 文件。

    Args:
        directory (str): 需要清理的目录路径

    Example:
        >>> delete_empty_txt_files('./labels')
    """
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否以 .txt 结尾
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(directory, filename)
            # 检查文件大小是否为0字节
            if os.path.getsize(file_path) == 0:
                # 删除文件
                os.remove(file_path)
                print(f"Deleted empty file: {filename}")
            else:
                print(f"File has content: {filename}")

    print("Processing complete. All empty .txt files have been deleted.")


def count_boxes(label_folder):
    """
    统计 YOLO 标签目录下所有 txt 文件中的目标框总数（非空行数）。

    Args:
        label_folder (str): 标签目录路径

    Returns:
        int: 目标框总数

    Example:
        >>> total = count_boxes('./labels')
        >>> print(total)
        12345
    """
    total_boxes = 0

    # 遍历文件夹中的所有文件
    for filename in os.listdir(label_folder):
        # 检查文件是否为txt文件
        if filename.endswith(".txt"):
            file_path = os.path.join(label_folder, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    # 跳过空行
                    if line.strip():
                        total_boxes += 1

    print(f"total boxes count: {total_boxes}")
    return total_boxes


def modify_category(source_label_folder, target_label_folder, category_mapping):
    """
    批量修改 YOLO 标签文件中类别索引，并保存到新目录。

    Args:
        source_label_folder (str): 原始标签目录
        target_label_folder (str): 修改后标签输出目录（自动创建）
        category_mapping (dict[str, str]): 旧类别到新类别的映射字典，如 {"0":"2", "3":"1"}

    Example:
        >>> modify_category(
        ...     source_label_folder='./labels',
        ...     target_label_folder='./labels_mapped',
        ...     category_mapping={"0": "2", "3": "1"}
        ... )
    """
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_label_folder):
        os.makedirs(target_label_folder)

    # 遍历原始标签文件夹中的所有txt文件
    for filename in os.listdir(source_label_folder):
        if filename.endswith('.txt'):
            source_label_file = os.path.join(source_label_folder, filename)
            target_label_file = os.path.join(target_label_folder, filename)

            modified_lines = []
            with open(source_label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts[0] in category_mapping:
                        parts[0] = category_mapping[parts[0]]
                    modified_lines.append(' '.join(parts) + '\n')

            with open(target_label_file, 'w') as f:
                f.writelines(modified_lines)

    print("类别修改完成，并已保存到新文件夹！")
