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

import io
import os
import shutil
import warnings
import cv2
import matplotlib

from pathlib import Path
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

# path
ROOT = Path(__file__).resolve().parent.parent  # TinyTrain/
ASSETS_PATH = ROOT / "assets"

# PyTorch Multi-GPU DDP
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))

# profiler
TIMER_ENABLED = int(os.getenv("USE_TINYTRAIN_TIMER_ENABLED", 0))  # 是否进行耗时统计，0表示不统计，1表示统计

# log
LOGGING_NAME = os.getenv("TINYTRAIN_LOGGING_NAME", "TinyTrain")

# format
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes

# albumentations
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # disable automatic new versions checks
warnings.filterwarnings("ignore", module="albumentations.*")

# opencv
cv2.setNumThreads(NUM_THREADS)  # 在模块初始化时设置一次

# font
DEFAULT_FONT = os.getenv("TINYTRAIN_FONT", "LXGWWenKai-Regular.ttf")


def localization(font: str = None):
    if font is None:
        # warnings.filterwarnings("ignore", message="Glyph.*missing from font")
        return

    font_dir = Path(matplotlib.matplotlib_fname()).parent / "fonts/ttf/"

    # 判断字体路径类型
    font_path = Path(font)

    if font_path.is_absolute() or (len(font_path.parts) > 1 and font_path.parts[0] not in {".", ".."}):
        # 绝对路径或相对路径（包含目录分隔符）
        if font_path.exists():
            remote_font = font_path
            local_font = font_dir / font_path.name
        else:
            print(f"Specified font file does not exist: {font_path}")
            return
    else:
        # 只有字体文件名，从 assets/fonts 目录获取
        remote_font = ASSETS_PATH / "fonts" / font
        local_font = font_dir / font

    if local_font.exists():
        font_prop = FontProperties(fname=local_font)
    else:
        print(f"Localizing font Loading: {font_path.name}")

        # 拷贝新字体
        shutil.copy(remote_font, font_dir)

        # 方法一：直接删除字体缓存目录
        # shutil.rmtree(matplotlib.get_cachedir())
        # 方法二：移除缓存json文件
        cache_dir = Path(matplotlib.get_cachedir())
        for json_file in cache_dir.glob("*.json"):
            json_file.unlink()

        # 立即加入字体并扫描
        fm.fontManager.addfont(remote_font)
        font_prop = FontProperties(fname=remote_font)

    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # 强制生成缓存
    buf = io.BytesIO()
    plt.figure().text(0.5, 0.5, "test", ha='center')
    plt.savefig(buf, format='png')  # 直接写内存
    plt.close()
    buf.close()  # 立即释放内存


localization(DEFAULT_FONT)  # 在这里修改自己喜欢的字体,比如xxx.otf,xxx.ttf
