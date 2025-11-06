import math
import subprocess
import numpy as np
import torch

from pathlib import Path
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from tinytrain.utils import LOGGER
from tinytrain.utils.dist import find_available_port


class TrainResult:
    """
    训练/验证结果记录与可视化工具。

    功能
    ----
    1. 逐字段增量收集指标（loss、lr、mAP 等）。
    2. 实时写入 TensorBoard。
    3. 自动拉起 TensorBoard 进程（可选）。
    4. 训练完成后一键绘制折线图并保存 PNG。
    5. 追加写入 CSV，兼容中断续写。
    """

    def __init__(self, config_manager, save_dir: Path, row_name='epoch'):
        """
        Args:
            save_dir (Path): 结果保存目录，内部创建子目录 `log/` 和文件。
            row_name (str): 横轴名称，默认 "epoch"，可改为 "step"。
            launch_tb (bool): 是否自动启动 TensorBoard。
        """
        self.config_manager = config_manager
        self.save_dir = save_dir
        self.data = {}
        self.row_name = row_name
        self.launch_tb = launch_tb

        # tensorboard visualize
        self.writer = None
        self.tb_proc = None
        if launch_tb:
            # 创建 SummaryWriter
            tb_log_dir = save_dir / 'log'
            self.writer = SummaryWriter(tb_log_dir.as_posix())
            if self.writer:
                LOGGER.info(f"TensorBoard logging to: {tb_log_dir.resolve()}")

            # 自动拉起 TensorBoard
            self._launch_tensorboard(tb_log_dir)

    def add(self, key, value):
        """
        记录单个指标值，并同步写入 TensorBoard。

        Args:
            key (str): 指标名称，如 "loss_cls"。
            value (int | float | Tensor): 数值或 0-D Tensor。
        """
        self.data.setdefault(key, []).append(value)

        if self.launch_tb:
            val = self._to_float(value)
            if val is not None and self.writer is not None:
                step = len(self.data[key])
                self.writer.add_scalar(key, val, global_step=step)

    def plot(self, start=1):
        """
        一次性绘制所有数值型指标的折线图，并保存为 `result.png`。
        子图行列自适应，保证紧凑美观。

        Args:
            start (int): 横轴起始值，默认 1。
        """
        LOGGER.info(f"plotting result.png...")
        # 过滤非数字键值对
        filtered_dic = {key: value for key, value in self.data.items() if all(isinstance(x, (int, float)) for x in value)}

        # 获取子图数量
        num_subplots = len(filtered_dic)

        # 如果没有子图，直接返回
        if num_subplots == 0:
            return

        # 定义背景颜色和折线图颜色
        background_color = '#fff5e6'  # 米色
        line_color = '#003366'  # 深蓝色
        grid_color = '#666666'  # 深灰色

        # 计算子图的行列数，使得子图尽可能均匀地分布在矩形区域内
        cols = math.ceil(math.sqrt(num_subplots))  # 列数
        rows = math.ceil(num_subplots / cols)  # 行数

        # 创建大图和子图
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))  # 添加子图间距

        # 如果只有一个子图，axes不是数组，需要将其转换为数组
        if num_subplots == 1:
            axes = [[axes]]  # 转换为二维数组

        # 将所有轴对象展平为一个列表
        axes = axes.flatten()

        # 绘制每个子图
        for i, (key, value) in enumerate(filtered_dic.items()):
            # x轴数据为从start到start + len(value) - 1的整数列表
            x_data = list(range(start, start + len(value)))
            axes[i].plot(x_data, value, marker='o', linestyle='-', color=line_color)  # 使用对比度最强的颜色绘制折线图
            axes[i].set_xlabel(self.row_name, fontweight='bold')  # 设置x轴标签
            axes[i].set_ylabel(key, fontweight='bold')  # 设置y轴标签为键名
            axes[i].set_title(f"result/{key}", fontweight='bold')  # 设置子图标题

            # 设置子图背景颜色
            axes[i].set_facecolor(background_color)  # 设置背景颜色为浅蓝色

            # 确保x轴刻度为整数
            axes[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # 添加网格线
            axes[i].grid(True, which='both', linestyle='--', linewidth=0.5, color=grid_color)  # 使用黑色网格线

        # 隐藏剩余的空白子图
        for j in range(num_subplots, rows * cols):
            fig.delaxes(axes[j])

        # 调整子图间距
        plt.tight_layout(pad=2.0)

        # 保存图像
        plt.savefig(self.save_dir / f"result.png", dpi=300)  # 保存为PNG文件，分辨率为300dpi

    def save_csv(self):
        """
        将当前所有指标追加写入 `result.csv`。
        - 首次写入自动生成表头。
        - 后续追加行，支持断点续训。
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        csv_file_path = self.save_dir / "result.csv"

        # 计算当前 epoch/step
        row_num = max(len(v) for v in self.data.values())  # 当前行号
        epoch = row_num                     # 与旧版本保持一致，从 1 开始
        row_data = {self.row_name: epoch}

        # 组装这一行的所有列
        for key, values in self.data.items():
            # 如果某指标还没到当前行，则填 NaN
            val = values[row_num - 1] if len(values) >= row_num else float('nan')
            row_data[key] = val

        # 写文件
        import csv
        write_header = not csv_file_path.exists()  # 文件不存在时才写表头
        with csv_file_path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[self.row_name] + list(self.data.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row_data)

    def _launch_tensorboard(self, log_dir: Path):
        """在可用端口上启动 TensorBoard 子进程。"""
        port = find_available_port()
        cmd = ["tensorboard", "--logdir", str(log_dir), "--port", str(port), "--host", "0.0.0.0"]
        self.tb_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        LOGGER.info(f"TensorBoard launched at http://localhost:{port}  (PID={self.tb_proc.pid})")

    def close(self):
        """优雅关闭 SummaryWriter 与 TensorBoard 子进程。"""
        if self.writer:
            self.writer.close()
        if self.tb_proc:
            self.tb_proc.terminate()
            self.tb_proc.wait()
        LOGGER.info("TensorBoard process terminated.")

    @staticmethod
    def _to_float(x):
        """
        将 torch.Tensor / np.ndarray / Python 数值 → float。

        高维张量需先 `.detach().cpu().numpy().item()`。
        """
        if isinstance(x, torch.Tensor):
            # .item() 只对 0-D 张量有效；若需兼容高维请先 x.detach().cpu().numpy().item()
            return x.detach().cpu().item()
        if isinstance(x, np.ndarray):
            return x.item()  # 0-D ndarray
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return None  # 其它类型直接丢弃
