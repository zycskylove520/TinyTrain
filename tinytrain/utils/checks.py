import os

import torch
import numpy as np

from pathlib import Path
from PIL import ImageOps, Image
from torch import autocast
from torch.utils.data import DataLoader

from tinytrain.global_var import ROOT
from tinytrain.utils import LOGGER

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"


def exif_size(img: Image.Image) -> tuple[int, int]:
    """
    Returns the EXIF-corrected size of a PIL Image.

    Args:
        img (Image.Image): The input PIL Image object.

    Returns:
        tuple[int, int]: The corrected size (width, height).
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
    """Search file and return path."""
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
    检查图像尺寸是否为 divisor 的倍数。
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


def check_amp(trainer, model, dataset):
    """
    Checks if the model can use PyTorch Automatic Mixed Precision (AMP).
    This function tests whether the model produces similar results in FP32 and AMP modes.
    It returns False if AMP is not supported or if there are significant discrepancies.
    """

    device = next(model.parameters()).device  # Get the model's device

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
        del model_
        if isinstance(a, (list, tuple, set)):
            return a[0].shape == b[0].shape and torch.allclose(a[0], b[0].float(), atol=0.5)
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)

    # Create a temporary DataLoader to get a batch of data
    temp_dataloader = DataLoader(dataset=dataset, batch_size=2, collate_fn=dataset.collate_fn)
    batch_samples = trainer.preprocess_data(next(iter(temp_dataloader)))

    try:
        # Perform the AMP check
        assert amp_allclose(model, batch_samples.data)
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
    Checks if an image file is valid and repairs corrupt JPEGs if necessary.

    Args:
        img_file (Path): The path to the image file.
        *args: Additional arguments to be returned unchanged.

    Returns:
        tuple: (img_file, good, bad, message, *args)
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
            assert image_format in IMG_FORMATS, f"Invalid image format {image.format}. {FORMATS_HELP_MSG}"

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
    Checks if an image file and its corresponding label file exist.

    Args:
        img_file (Path | str): The path to the image file.
        label_paths (Path | list[Path]): The path(s) to the label directory or directories.

    Returns:
        tuple: (bg_img_file, img_file, label_file, is_bg, good, bad, message)
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


def check_device_mini(device):
    """
    Checks the availability of the specified device and returns a torch.device object.

    Args:
        device (str | int | list): The device to check. Can be 'cpu', 'cuda', 'mps', an integer (GPU index), or a list of integers (GPU indices).

    Returns:
        torch.device: The selected device.

    Raises:
        ValueError: If the specified device is not supported.
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


def check_detect_yolo_label(img_file, npy_file=None, label_file=None):
    """
    检查目标检测数据集的标签文件是否正确。

    Args:
        img_file (str | Path): 图像文件路径。
        npy_file (str | Path, optional): 可选的 NumPy 文件路径。默认为 None。
        label_file (str | Path, optional): 标签文件路径。如果图片没有对应的标签文件，则为 None。

    Returns:
        tuple: (img_file, npy_file, message, cls, boxes)

    Raises:
        ValueError: 如果标签文件格式不正确或包含无效值。
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
