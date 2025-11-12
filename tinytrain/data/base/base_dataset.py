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

from torch.utils.data import Dataset, IterableDataset

from tinytrain.data.data_format import BaseDataInfo, BaseBatchDataInfo


class TTBaseMapDataset(Dataset):
    """
    所有map-style数据集的 **抽象基类**。

    职责
    ----
    1. 定义必须实现的接口：`__getitem__`、`__len__`、`collate_fn`。
    2. 作为类型标记，便于 `DataLoader` 自动识别。

    子类要求
    --------
    - 必须实现 `__getitem__`、`__len__`、`collate_fn`。
    - 必须返回 **继承自 `BaseDataInfo` 的对象**。
    """

    def __init__(self, config_manager):
        """
        Args:
            config_manager (TTConfigManager): 全局配置管理器。
        """
        super().__init__()
        self.config_manager = config_manager

    def __getitem__(self, index) -> BaseDataInfo:
        """子类必须实现：返回单个样本（`BaseDataInfo` 子类）。"""
        raise NotImplementedError

    def __len__(self) -> int:
        """子类必须实现：返回数据集大小。"""
        raise NotImplementedError

    def collate_fn(self, batch: list[BaseDataInfo]) -> BaseBatchDataInfo:
        """
        子类必须实现：将 `list[BaseDataInfo]` 整理为 `BaseBatchDataInfo`。

        Args:
            batch (list): 一批样本。

        Returns:
            BaseBatchDataInfo: 批数据容器。
        """
        return batch  # type: ignore[arg-type]


class TTBaseIterableDataset(IterableDataset):
    """
    所有 iterable-style 数据集的 **抽象基类**。

    职责
    ----
    1. 定义必须实现的接口：`__iter__`、`collate_fn`。
    2. 作为类型标记，便于 `DataLoader` 自动识别。

    子类要求
    --------
    - 必须实现 `__iter__`，按需返回 **继承自 `BaseDataInfo` 的对象**。
    - 必须实现 `collate_fn`，用于整理批数据。
    """

    def __init__(self, config_manager):
        """
        Args:
            config_manager (TTConfigManager): 全局配置管理器。
        """
        super().__init__()
        self.config_manager = config_manager

    def __iter__(self) -> BaseDataInfo:
        """子类必须实现：返回数据迭代器，每次产出单个样本（`BaseDataInfo` 子类）。"""
        raise NotImplementedError

    def collate_fn(self, batch: list[BaseDataInfo]) -> BaseBatchDataInfo:
        """
        子类必须实现：将 `list[BaseDataInfo]` 整理为 `BaseBatchDataInfo`。

        Args:
            batch (list): 一批样本。

        Returns:
            BaseBatchDataInfo: 批数据容器。
        """
        return batch  # type: ignore[arg-type]
