"""
通用数据/设备校验与图像处理工具模块
提供文件搜索、图像格式/尺寸检查、AMP 验证、YOLO 标签校验等一站式工具函数。
"""

import os
import torch
import numpy as np

from pathlib import Path
from PIL import ImageOps, Image
from torch import autocast
from torch.utils.data import DataLoader

from tinytrain.global_var import ROOT, IMG_FORMATS, VID_FORMATS
from tinytrain.utils import LOGGER


def exif_size(img: Image.Image) -> tuple[int, int]:
    """
    根据 EXIF 修正 PIL 图像的宽高（处理旋转 90/270°）。

    Args:
        img: PIL.Image 对象

    Returns:
        (width, height) 修正后的尺寸
    """
    # 获取图像的原始尺寸
    width, height = img.size

    # 只处理 JPEG 格式的图像
    if img.format != "JPEG":
        return width, height

    try:
        # 获取 EXIF 信息
        exif = img.getexif()
        if exif:
            # 获取旋转信息（EXIF 标签 274 表示方向）
            rotation = exif.get(274, None)
            # 如果旋转角度为 90 或 270 度，交换宽高
            if rotation in {6, 8}:
                width, height = height, width
    except (AttributeError, KeyError, ValueError) as e:
        # 如果获取 EXIF 信息失败，忽略异常
        pass

    return width, height


def check_file(file: str | Path) -> Path:
    """
    根据相对或绝对路径，在 ROOT 及父目录中递归搜索文件，返回唯一绝对路径。

    Raises:
        FileNotFoundError: 文件不存在或存在多个匹配
    """
    # 将 file 转换为 Path 对象
    file_path = Path(file).resolve()

    # 检查是否是绝对路径且文件存在
    if file_path.is_absolute() and file_path.exists():
        return file_path.resolve()

    # 构造可能的搜索路径
    search_paths = [
        ROOT / file_path,  # 在 ROOT 下直接查找
        ROOT / "**" / file_path,  # 在 ROOT 下递归查找
        ROOT.parent / file_path,  # 在 ROOT 的父目录下查找
        ROOT.parent / "**" / file_path  # 在 ROOT 的父目录下递归查找
    ]

    # 进行递归搜索
    found_files = []
    for path in search_paths:
        if path.is_absolute():
            if path.exists():
                found_files.append(path)
        else:
            found_files.extend(ROOT.rglob(str(path)))

    # 检查搜索结果
    if not found_files:
        raise FileNotFoundError(f"'{file}' does not exist")
    elif len(found_files) > 1:
        raise FileNotFoundError(f"Multiple files match '{file}'. Specify exact path: {found_files}")

    return found_files[0].resolve()


def check_img_size(img_size: int | list[int] | tuple[int, int], divisor=32, mode="train") -> tuple[int, int]:
    """
    确保输入尺寸是 divisor 的倍数；不足时自动向上取整。

    Args:
        img_size: 单个 int 或 (w, h)
        divisor:  对齐基数
        mode:     用于日志提示，如 "train"/"val"

    Returns:
        (width, height) 已对齐的尺寸
    """
    # 确保 img_size 是一个元组或列表，并且包含两个整数
    if isinstance(img_size, int):
        img_size = (img_size, img_size)  # 如果是单个整数，将其转换为宽高相等的元组
    elif not isinstance(img_size, (list, tuple)) or len(img_size) != 2:
        raise ValueError("Image size must be an integer, a list of two integers, or a tuple of two integers.")
    elif not all(isinstance(dim, int) for dim in img_size):
        raise ValueError("Image size dimensions must be integers.")

    original_w, original_h = img_size

    # 检查是否已经是 divisor 的倍数
    if original_w % divisor == 0 and original_h % divisor == 0:
        LOGGER.info(f"{mode}: input image size (w, h) is: {img_size}")
        return img_size

    # 调整图像尺寸
    new_w = divisor if original_w < divisor else round(original_w / divisor) * divisor
    new_h = divisor if original_h < divisor else round(original_h / divisor) * divisor

    # 输出警告信息
    LOGGER.info(f"{mode}: The image size (w, h) specified during training [{original_w}, {original_h}] must be a multiple of {divisor}! "
                f"The training size is automatically adjusted to: [{new_w}, {new_h}]")
    return new_w, new_h


def check_device_mini(device):
    """
    根据输入字符串或整数返回 torch.device。

    Args:
        device: 'cpu' | 'cuda' | 'mps' | int | list[int]

    Returns:
        torch.device
    """
    # Check if the specified device is available
    if device is None or device == "cpu":
        return torch.device("cpu")
    elif device == "mps":
        if not torch.mps.is_available():
            raise ValueError("MPS device requested, but MPS is not available. Defaulting to CPU.")
        return torch.device("mps")
    elif device == "cuda" or isinstance(device, int) or isinstance(device, list):
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested, but CUDA is not available. Defaulting to CPU.")
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        elif isinstance(device, list):
            if len(device) == 0:
                raise ValueError("List of CUDA devices is empty. Defaulting to CPU.")
            return torch.device(f"cuda:{device[0]}")
        else:
            return torch.device("cuda:0")
    else:
        raise ValueError("Device must be 'cpu', 'cuda', 'mps', an integer (GPU index), or a list of integers (GPU indices).")


def check_amp(trainer):
    """
    检测模型在 AMP 模式下是否数值稳定（不 NaN、不崩溃）。

    原理：对比 FP32 与 AMP 输出误差 < atol=0.5。
    若失败，返回 False，提示关闭 AMP。

    Args:
        trainer:  训练器实例，用于数据预处理

    Returns:
        bool: 是否支持 AMP
    """
    device = trainer.device

    def amp_allclose(model: torch.nn.Module, batch: torch.Tensor):
        """
        Checks if the model's output is close between FP32 and AMP modes.

        Args:
            model (torch.nn.Module): The model to test.
            batch (torch.Tensor): The input batch to test.

        Returns:
            bool: True if the outputs are close, False otherwise.
        """
        model_ = model.to(device)
        batch = batch.to(device)
        with torch.no_grad():  # Disable gradient computation to save memory
            a = model_(batch)[0]  # FP32 inference
            with autocast(device_type=device.type, enabled=True):
                b = model_(batch)[0]  # AMP inference
        if isinstance(a, (list, tuple, set)):
            return a[0].shape == b[0].shape and torch.allclose(a[0], b[0].float(), atol=0.5)
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)

    # Create a temporary DataLoader to get a batch of data
    dataset = trainer.train_dataloader.dataset
    temp_dataloader = DataLoader(dataset=dataset, batch_size=2, collate_fn=dataset.collate_fn)
    batch_samples = trainer.preprocess_data(next(iter(temp_dataloader)))

    try:
        # Perform the AMP check
        assert amp_allclose(trainer.model, batch_samples.data)
    except (AttributeError, ModuleNotFoundError) as e:
        LOGGER.warning(
            f"AMP: Checks skipped due to unsupported functionality. {e}. "
            f"AMP will be disabled. If you experience issues, set amp=False."
        )
        return False
    except AssertionError:
        LOGGER.warning(
            "Disabling AMP due to detected anomalies that may cause NaN losses or zero-mAP results."
        )
        return False
    return True


def check_image(img_file: Path, *args):
    """
    检测图像是否完整，必要时修复损坏 JPEG。

    Returns:
        tuple: (img_file, good, bad, message, *args)
               good=1 表示有效，bad=1 表示异常
    """
    good, bad, message = 0, 0, ""
    try:
        # 打开图像并验证
        with Image.open(img_file) as image:
            image.verify()  # 验证图像完整性
            shape = exif_size(image)  # 获取图像尺寸
            height, width = shape  # 转换为高宽格式

            # 检查图像尺寸
            assert height > 9 and width > 9, f"Image size {shape} < 10 pixels"

            # 检查图像格式
            image_format = image.format.lower()
            assert image_format in IMG_FORMATS, f"Invalid image format {image.format}. Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"

            # 修复损坏的 JPEG 图像
            if image_format in {"jpg", "jpeg"}:
                with open(img_file, "rb+") as f:
                    f.seek(-2, 2)
                    if f.read(2) != b"\xff\xd9":  # 检查 JPEG 文件是否损坏
                        LOGGER.warning(f"Dataset WARNING ⚠️ {img_file}: Corrupt JPEG detected. Attempting to repair...")
                        ImageOps.exif_transpose(image).save(img_file, "JPEG", subsampling=0, quality=100)
                        message = f"Dataset WARNING ⚠️ {img_file}: Corrupt JPEG restored and saved"

        # 如果没有异常，标记为有效图像
        good = 1
    except Exception as e:
        bad = 1
        message = f"Dataset WARNING ⚠️ {img_file}: Ignoring corrupt image: {e}"
        LOGGER.warning(message)

    return img_file, good, bad, message, *args


def check_image_and_label(img_file: Path | str, label_paths: Path | list[Path]):
    """
    检查图片及对应标签是否存在；若标签不存在/空，则标记为背景图。

    Args:
        img_file: 图片路径
        label_paths: 标签目录或目录列表

    Returns:
        (bg_img_file, img_file, label_file, is_bg, good, bad, message)
    """
    # 初始化返回值
    bg_img_file, is_bg, good, bad, message, label_file = None, False, 0, 0, "", None

    # 验证图片
    img_file, good, bad, message, *_ = check_image(img_file)

    # 获取图像文件的名称（不带扩展名）
    img_stem = Path(img_file).stem
    label_file_name = f"{img_stem}.txt"

    # 遍历所有可能的标签路径
    if isinstance(label_paths, list):
        label_paths = [Path(path) for path in label_paths]
    else:
        label_paths = [Path(label_paths)]

    for label_path in label_paths:
        label_file = label_path / label_file_name
        if label_file.exists():
            if os.path.getsize(label_file) == 0:
                is_bg = True
                bg_img_file = img_file
                message = f"Dataset WARNING ⚠️ {img_file}: {label_file} has zero size and has been set as the background image."
            break
        else:
            is_bg = True
            bg_img_file = img_file
            message = f"Dataset WARNING ⚠️ {img_file}: {label_file} does not exist and has been set as the background image."
            break

    return bg_img_file, img_file, label_file, is_bg, good, bad, message


def check_detect_yolo_label(img_file, npy_file=None, label_file=None):
    """
    验证 YOLO 目标检测标签文件合法性：每行 5 个元素，cls ≥ 0，xywh ∈ [0,1]。

    Args:
        img_file:   图像路径
        npy_file:   NumPy 文件路径（可选）
        label_file: txt 标签路径（可选）

    Returns:
        (img_file, npy_file, message, cls, boxes)
    """
    message = None
    labels_arr = np.zeros((0, 5), dtype=np.float32)  # 初始化为空数组

    if label_file is not None:
        try:
            # 读取标签文件
            with open(label_file, "r") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if x]

            # 检查标签文件是否为空
            if len(lb) == 0:
                message = f"Label file: {label_file} is an empty txt file!"
            else:
                # 检查每行是否包含 5 个元素
                if any(len(x) != 5 for x in lb):
                    raise ValueError(f"Label file: {label_file} has a line that does not contain 5 elements: cls, xywh. Please correct the label file.")

                # 将标签数据转换为 NumPy 数组
                labels_arr = np.array(lb, dtype=np.float32)

                # 检查类别索引是否为非负整数
                if not np.all(labels_arr[:, 0] >= 0):
                    raise ValueError(f"Label file: {label_file} contains invalid class index. Class index must be a non-negative integer.")

                # 检查边界框坐标是否在有效范围内 [0, 1]
                if not np.all((labels_arr[:, 1:] >= 0) & (labels_arr[:, 1:] <= 1)):
                    raise ValueError(f"Label file: {label_file} contains invalid bounding box coordinates. Coordinates must be within the range [0, 1].")
        except ValueError as e:
            raise ValueError(f"Error in label file: {label_file}. {e}")

    # 提取类别索引和边界框坐标
    cls = labels_arr[:, 0]
    boxes = labels_arr[:, 1:]

    return img_file, npy_file, message, cls, boxes


def check_detect_voc_label(img_file: str, npy_file: str = None, label_file: str = None):
    """
    验证 PASCAL VOC 目标检测标签文件合法性：
        每框包含：cls≥0，xyxy 有效且位于图像内。
    返回的 boxes 为 [xmin, ymin, xmax, ymax]（绝对像素）。

    Args / Returns 同 check_detect_yolo_label
    """
    import xml.etree.ElementTree as ET
    message = None
    labels_arr = np.zeros((0, 5), dtype=np.float32)

    if label_file is None:
        return img_file, npy_file, message, labels_arr[:, 0], labels_arr[:, 1:]

    if not os.path.exists(label_file):
        raise ValueError(f"VOC label file not found: {label_file}")

    try:
        tree = ET.parse(label_file)
        root = tree.getroot()

        # 拿图像宽高
        size = root.find("size")
        img_w = int(size.findtext("width", 0)) if size is not None else 0
        img_h = int(size.findtext("height", 0)) if size is not None else 0
        if img_w == 0 or img_h == 0:
            with Image.open(img_file) as im:
                img_w, img_h = im.size

        objs = root.findall("object")
        if not objs:
            message = f"VOC label file: {label_file} has zero objects."
        else:
            records = []
            for obj in objs:
                name = obj.findtext("name")
                if name is None:
                    raise ValueError("Missing <name> in VOC object")
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    raise ValueError("Missing <bndbox> in VOC object")

                xmin = float(bndbox.findtext("xmin", 0))
                ymin = float(bndbox.findtext("ymin", 0))
                xmax = float(bndbox.findtext("xmax", 0))
                ymax = float(bndbox.findtext("ymax", 0))

                if not (0 <= xmin <= xmax <= img_w and 0 <= ymin <= ymax <= img_h):
                    raise ValueError(
                        f"Invalid bbox in VOC label: ({xmin},{ymin},{xmax},{ymax}) "
                        f"with image size ({img_w},{img_h})"
                    )

                cls_id = int(name)  # 若 name 是字符串请先做映射
                if cls_id < 0:
                    raise ValueError(f"Negative class id: {cls_id}")

                records.append([cls_id, xmin, ymin, xmax, ymax])

            labels_arr = np.array(records, dtype=np.float32)

    except ET.ParseError as e:
        raise ValueError(f"VOC label file: {label_file} XML parse error: {e}")

    cls = labels_arr[:, 0]
    boxes = labels_arr[:, 1:]  # (N,4) [xmin,ymin,xmax,ymax]
    return img_file, npy_file, message, cls, boxes


def check_detect_coco_label(img_file: str, npy_file: str = None, label_file: str = None):
    """
    验证 COCO 目标检测标签文件合法性：
        每框包含：cls≥0，bbox[x,y,w,h] 绝对像素坐标。
    返回的 boxes 为 [x, y, w, h]（绝对像素）。

    Args / Returns 同 check_detect_yolo_label
    """
    import json
    message = None
    labels_arr = np.zeros((0, 5), dtype=np.float32)

    if label_file is None:
        return img_file, npy_file, message, labels_arr[:, 0], labels_arr[:, 1:]

    if not os.path.exists(label_file):
        raise ValueError(f"COCO label file not found: {label_file}")

    try:
        with open(label_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # 建立 img_id → (w,h) 映射
        id2size = {img["id"]: (img["width"], img["height"])
                   for img in coco.get("images", [])}

        # 从图片路径反推文件名，再匹配 image_id
        img_name = os.path.basename(img_file)
        img_entry = next((img for img in coco.get("images", [])
                          if img["file_name"] == img_name), None)
        if img_entry is None:
            raise ValueError(f"Image {img_name} not found in COCO json")
        img_id = img_entry["id"]
        img_w, img_h = img_entry["width"], img_entry["height"]

        anns = [ann for ann in coco.get("annotations", []) if ann["image_id"] == img_id]
        if not anns:
            message = f"COCO label file: {label_file} has zero objects for image {img_name}."
        else:
            records = []
            for ann in anns:
                cls_id = int(ann["category_id"])
                if cls_id < 0:
                    raise ValueError(f"Negative class id: {cls_id}")

                x, y, w, h = ann["bbox"]
                x, y, w, h = float(x), float(y), float(w), float(h)
                if w <= 0 or h <= 0:
                    raise ValueError(f"Invalid bbox size (w,h)=({w},{h})")
                if not (0 <= x <= img_w and 0 <= y <= img_h and
                        0 <= x + w <= img_w and 0 <= y + h <= img_h):
                    raise ValueError(
                        f"Invalid bbox in COCO label: ({x},{y},{w},{h}) "
                        f"with image size ({img_w},{img_h})"
                    )

                records.append([cls_id, x, y, w, h])

            labels_arr = np.array(records, dtype=np.float32)

    except json.JSONDecodeError as e:
        raise ValueError(f"COCO label file: {label_file} JSON decode error: {e}")

    cls = labels_arr[:, 0]
    boxes = labels_arr[:, 1:]  # (N,4) [x,y,w,h] 绝对像素
    return img_file, npy_file, message, cls, boxes


def check_pose_yolo_label(img_file, keypoint_shape, npy_file=None, label_file=None):
    """
    验证 YOLO 姿态估计标签文件合法性：每行 5 个元素，cls ≥ 0，xywh ∈ [0,1]。

    Args:
        img_file:   图像路径
        npy_file:   NumPy 文件路径（可选）
        label_file: txt 标签路径（可选）

    Returns:
        (img_file, npy_file, message, cls, boxes, key_points)
    """
    keypoint_num = keypoint_shape[0]
    keypoint_dim = keypoint_shape[1]

    message = None
    labels_arr = np.zeros((0, 5), dtype=np.float32)  # 初始化为空数组
    keypoints = np.zeros((0, keypoint_num, keypoint_dim), dtype=np.float32)

    if label_file is not None:
        try:
            # 读取标签文件
            with open(label_file, "r") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if x]

            # 检查标签文件是否为空
            if len(lb) == 0:
                message = f"Label file: {label_file} is an empty txt file!"
            else:
                # 检查每行是否包含足够的元素
                if any(len(x) != (5 + keypoint_num * keypoint_dim) for x in lb):
                    raise ValueError(f"Label file: {label_file} has a line that does not contain 5+{keypoint_num}*{keypoint_dim} elements: cls, xywh, keypoint. Please correct the label file.")

                # 将标签数据转换为 NumPy 数组
                lb = np.array(lb, dtype=np.float32)
                labels_arr = lb[:, :5]
                keypoints = lb[:, 5:].reshape(-1, keypoint_num, keypoint_dim)
                if keypoint_dim == 2:
                    # 如果关键点是2维的，没有visibility值，就补上这个值
                    kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                    keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)

                # 检查类别索引是否为非负整数
                if not np.all(labels_arr[:, 0] >= 0):
                    raise ValueError(f"Label file: {label_file} contains invalid class index. Class index must be a non-negative integer.")

                # 检查边界框坐标是否在有效范围内 [0, 1]
                if not np.all((labels_arr[:, 1:] >= 0) & (labels_arr[:, 1:] <= 1)):
                    raise ValueError(f"Label file: {label_file} contains invalid bounding box coordinates. Coordinates must be within the range [0, 1].")
        except ValueError as e:
            raise ValueError(f"Error in label file: {label_file}. {e}")

    # 提取类别索引和边界框坐标
    cls = labels_arr[:, 0]
    boxes = labels_arr[:, 1:]

    return img_file, npy_file, message, cls, boxes, keypoints
