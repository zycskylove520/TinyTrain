from pathlib import Path
from typing import Literal, List

from pydantic import BaseModel



class CoreConfig(BaseModel):
    """
    核心配置类，包含训练的基本设置、训练参数、验证参数、超参数、分布式训练设置等。
    """
    # Base Settings
    project_name: str  # 项目名称。如果为空字符串，则训练结果保存在默认目录 `default_project` 下。
    task: Literal["classify", "detect"]  # 训练任务类型。用于加载对应任务的模型。
    save_dir: str | Path  # 训练结果保存路径。可以是字符串或 `Path` 对象。

    # Train Settings
    epochs: int  # 训练总轮数。必须大于等于 1。
    batch_size: int  # 每个批次的样本数量。必须大于等于 2。
    device: list[int] | int | Literal["mps", "cpu", "cuda"]  # 训练设备。可以是 GPU 设备编号（整数或整数列表）、"mps"（苹果 Metal）、"cpu" 或 "cuda"。
    workers: int  # 数据加载器的多线程数量。值为 0 表示不使用多线程。必须大于等于 0。
    resume: bool  # 是否从上次中断的训练进度继续训练。
    amp: bool  # 是否启用自动混合精度（AMP）。用于节省内存并加速训练，但可能影响精度。
    accumulate: int  # 梯度累积步数。每累积 `accumulate` 步后更新一次参数。必须大于等于 1。
    patience: int  # 早停机制的耐心值。如果训练结果在 `patience` 个 epoch 内没有改善，则提前终止训练。值为 0 表示不启用。
    save_period: int  # 每隔多少个 epoch 保存一次模型权重文件。值为 0 表示不保存中间权重。
    time: int  # 训练时间限制（秒）。训练超过该时间后自动终止。值为 0 表示不限制时间。
    seed: int  # 随机种子，用于复现训练结果。
    deterministic: bool  # 是否启用确定性模式。启用后，确保在相同的输入、硬件和软件环境下，每次运行的结果完全一致。
    ema: bool  # 是否启用指数移动平均（EMA）。EMA 可以生成参数更平滑的模型，但会增加训练时间。适用于中小型模型。
    shuffle_val_dataloader: bool  # 是否在验证集 DataLoader 中打乱数据。在分类任务中启用后，可以在验证阶段看到更多样的图片。

    # Validation Settings
    conf_threshold: float  # 验证时的置信度阈值。用于非极大值抑制（NMS）。该值越大，在验证集数据大的情况下，效率越高，对召回率的要求越严格，这会导致召回率更低。
    nms_threshold: float  # 验证时的 NMS 阈值。

    # Hyperparameters
    optimizer: Literal["SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"]  # 优化器类型。
    momentum: float  # 优化器的动量值。
    weight_decay: float  # 权重衰减（L2 正则化）值。用于防止过拟合。
    scheduler: Literal["LinearLR", "CosineLR", "auto"]  # 学习率调度器类型。`auto` 模式下，模型会根据训练效果自动调整学习率。
    lr0: float  # 初始学习率。
    lr1: float  # 结束学习率（`lr0 * lr1`）。在 `scheduler` 为 `auto` 时无效。
    warmup_lr: float  # 预热阶段的学习率。
    warmup_epochs: int  # 预热阶段的 epoch 数量。
    l1_norm: float  # L1 正则化值。用于防止过拟合。

    # DDP Settings
    master_addr: str  # 分布式训练时的主机 IP 地址。
    master_port: int  # 分布式训练时的主机端口。
    nodes: int  # 分布式训练的节点数量。
    node_rank: int  # 当前节点的排名（RANK）。


class ModelConfig(BaseModel):
    """
    模型配置类，包含模型的尺寸和网络结构。
    """
    scale: str  # 默认启用的模型尺寸。
    scales: dict  # 模型的多种尺寸配置。
    network: List  # 模型的网络结构定义。


class DatasetConfig(BaseModel):
    """
    数据集配置类，包含数据集的基本路径、训练集、验证集、测试集等信息。
    """
    # common
    img_size: int | list[int, int] | tuple[int, int]  # 训练图片的尺寸。可以是单个整数（宽=高）或一个列表 `[宽, 高]`。
    cache: bool  # 是否缓存数据集。如果启用，会将数据集缓存为 `.npy` 文件，以加速读取。
    path: str | Path | list[Path] | list[str]  # 数据集的根目录路径。可以是字符串、`Path` 对象或路径列表。
    train: str  # 训练集的相对目录。完整的训练集路径为 `path/train`。
    val: str  # 验证集的相对目录。完整的验证集路径为 `path/val`。
    test: str  # 测试集的相对目录。完整的测试集路径为 `path/test`。可选。
    nc: int  # 数据集的类别数量。
    names: dict  # 类别索引映射字典。键为类别索引，值为类别名称。

    # pose
    kpt_shape: list | None  # 关键点的形状。仅在处理关键点检测任务时使用。
    flip_idx: list | None  # 对称镜像索引。仅在处理关键点检测任务时使用。
