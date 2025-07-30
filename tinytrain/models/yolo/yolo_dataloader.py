from torch.utils.data import dataloader


class InfiniteDataLoader(dataloader.DataLoader):
    """
    无限循环的 DataLoader，可在训练过程中持续复用 worker 进程而无需重启。

    特性
    ----
    1. 对原有 DataLoader 使用方式 100 % 兼容，即插即用。
    2. 通过 `_RepeatSampler` 将 BatchSampler 无限展开，避免 epoch 边界导致
       worker 重启，显著减少 CPU 进程创建/销毁开销。
    3. 提供 `reset()` 接口，允许在运行时动态替换 / 修改底层 Dataset
       （例如切换数据增强策略、更新采样权重等）后立刻生效，
       无需重新构造 DataLoader 实例。

    示例
    ----
    >>> dataloader = InfiniteDataLoader(dataset, batch_size=32, num_workers=8)
    >>> for step, batch in enumerate(dataloader):
    ...     # 无限循环，step 会一直递增
    ...     if step == 1000:
    ...         dataloader.reset()   # 训练中途重置采样器
    """

    def __init__(self, *args, **kwargs):
        """
        初始化 InfiniteDataLoader。

        参数
        ----
        *args, **kwargs : 与原生 `torch.utils.data.DataLoader` 完全一致。
        """
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """
        返回「原始采样器」的长度，即每个 epoch 的迭代次数。
        注意：由于数据流无限循环，实际迭代次数不受此限制。
        """
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """无限循环产生 mini-batch，直到外部主动 break 或抛出异常。"""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        重置内部迭代器，常用于：
        - 切换 Dataset 内部状态（transform、采样策略等）。
        - 手动触发 worker 重启，以加载最新数据。

        调用后，下一次 `__iter__` 会重新生成迭代器。
        """
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """
    无限重复给定采样器的「胶水」类。

    说明
    ----
    任何 `torch.utils.data.Sampler` 经过 `_RepeatSampler` 包装后，
    都会变成「永不停止」的迭代器，供 `InfiniteDataLoader` 内部使用。
    """

    def __init__(self, sampler):
        """
        参数
        ----
        sampler : torch.utils.data.Sampler
            需要被无限重复的采样器实例。
        """
        self.sampler = sampler

    def __iter__(self):
        """无限循环地产生采样器中的索引序列。"""
        while True:
            yield from iter(self.sampler)