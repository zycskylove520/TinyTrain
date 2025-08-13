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
DEFAULT_CFG_PATH = ROOT / "cfg"
ASSETS_PATH = ROOT / "assets"

# PyTorch Multi-GPU DDP
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))

# log
LOGGING_NAME = "TinyTrain"

# albumentations
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # disable automatic new versions checks
warnings.filterwarnings("ignore", module="albumentations.*")

# opencv
cv2.setNumThreads(NUM_THREADS)  # 在模块初始化时设置一次

# font
def localization(font):
    font_dir = Path(matplotlib.matplotlib_fname()).parent / "fonts/ttf/"
    local_font = font_dir / font
    if local_font.exists():
        font_prop = FontProperties(fname=local_font)
    else:
        print(f"Localizing font Loading: {font}")
        import importlib
        importlib.reload(matplotlib)

        # 方法一：直接删除字体缓存目录
        # shutil.rmtree(matplotlib.get_cachedir())
        # 方法二：移除缓存json文件
        cache_dir = matplotlib.get_cachedir()
        cache_files = os.listdir(cache_dir)
        for cache_file in cache_files:
            if cache_file.split(".")[-1] == "json":
                os.remove(os.path.join(cache_dir, cache_file))

        # 拷贝新字体
        remote_font = ASSETS_PATH / f"fonts/{font}"
        shutil.copy(remote_font, font_dir)
        fm.fontManager.addfont(remote_font)
        font_prop = FontProperties(fname=remote_font)

        # 重新导入 matplotlib，刷新缓存加载新字体
        import importlib
        importlib.reload(matplotlib)

    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


localization("SourceHanSansSC-Normal.otf")  # 在这里修改自己喜欢的字体
