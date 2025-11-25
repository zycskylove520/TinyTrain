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
import torch

from typing import Iterator, Sequence
from torch.utils.data import Dataset, IterableDataset, ChainDataset, ConcatDataset, TensorDataset, StackDataset, Subset

from tinytrain.data.data_format import BaseDataInfo, BaseBatchDataInfo


#  tinytrain/data.dataset  ‑ 统一数据集规范
#  所有 Map/Iterable 风格数据集的 TT 包装与抽象基类。
#  约定：单个样本必须是 BaseDataInfo 子类；批数据必须是 BaseBatchDataInfo 子类。
#  ---------------------------------------------------------------------------

class TTBaseDataset:
    """
    全局配置入口 + 批整理协议。

    子类必须实现
    ------------
    collate_fn(batch: list[BaseDataInfo]) -> BaseBatchDataInfo
    """

    def __init__(self, config_manager):
        """
        Args:
            config_manager (TTConfigManager): 全局配置管理器。
        """
        self.config_manager = config_manager

    def collate_fn(self, batch: list[BaseDataInfo]) -> BaseBatchDataInfo:
        """
        子类必须实现：将 `list[BaseDataInfo]` 整理为 `BaseBatchDataInfo`。

        Args:
            batch (list): 一批样本。

        Returns:
            BaseBatchDataInfo: 批数据容器。
        """
        return batch  # type: ignore[arg-type]


class TTMapDataset(TTBaseDataset, Dataset):
    """
    Map-style 数据集的 **TinyTrain 统一抽象基类**。

    职责
    ----
    1. 强制约定单样本类型：``__getitem__`` 必须返回 ``BaseDataInfo`` 子类实例。
    2. 提供 **全局配置入口**（通过 ``config_manager``）。
    3. 内置 **拼接语法糖**：``ds = ds1 + ds2`` 自动返回 ``TTConcatDataset``。
    4. 子类只需实现 ``__getitem__`` / ``__len__`` / ``collate_fn`` 即可直接接入
       TinyTrain 训练管线（Sampler、DataLoader、Trainer 等均按此协议交互）。

    子类必须实现
    ------------
    __getitem__(index: int) -> BaseDataInfo
        根据下标返回单个样本，样本必须是 ``BaseDataInfo`` 子类。
    __len__() -> int
        返回数据集总样本数。
    collate_fn(batch: list[BaseDataInfo]) -> BaseBatchDataInfo
        将一批样本整理成批数据对象，供 DataLoader 自动调用。
    """

    def __init__(self, config_manager):
        """
        Args:
            config_manager (TTConfigManager): 全局配置管理器。
        """
        TTBaseDataset.__init__(self, config_manager)

    def __getitem__(self, index) -> BaseDataInfo:
        """
        根据下标返回单个样本。

        Args:
            index (int): 样本下标，取值范围 ``0 <= index < len(self)``。

        Returns:
            BaseDataInfo: 单样本对象，必须是 ``BaseDataInfo`` 的子类。

        Raises:
            IndexError: 下标越界时抛出。

        Note:
            子类必须实现此方法；基类不提供默认实现。
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        返回数据集总样本数。

        Returns:
            int: 数据集长度，必须 >= 0。

        Note:
            子类必须实现此方法；基类不提供默认实现。
        """
        raise NotImplementedError

    def __add__(self, other: "TTMapDataset") -> "TTConcatDataset":
        """
        语法糖：支持 ``ds = ds1 + ds2``，等价于 ``TTConcatDataset([ds1, ds2])``。

        Args:
            other (TTMapDataset): 另一个 Map-style 数据集，需与 ``self`` 拥有相同协议。

        Returns:
            TTConcatDataset: 逻辑拼接后的新数据集，不复制底层数据。
        """
        return TTConcatDataset(self.config_manager, [self, other])


class TTIterableDataset(TTBaseDataset, IterableDataset):
    """
    Iterable-style 数据集的 **TinyTrain 统一抽象基类**。

    职责
    ----
    1. 强制约定迭代产出类型：``__iter__`` 必须每次 yield 一个 ``BaseDataInfo`` 子类实例。
    2. 提供 **全局配置入口**（通过 ``config_manager``）。
    3. 内置 **链式语法糖**：``it = it1 + it2`` 自动返回 ``TTChainDataset``，
       迭代时先耗尽 ``it1``，再无缝切换到 ``it2``。
    4. 子类只需实现 ``__iter__`` / ``collate_fn`` 即可接入 TinyTrain 管线，
       支持流式读取、动态数据增强等场景。

    子类必须实现
    ------------
    __iter__() -> Iterator[BaseDataInfo]
        返回一个迭代器，每次产出单个 ``BaseDataInfo`` 子类实例。
    collate_fn(batch: list[BaseDataInfo]) -> BaseBatchDataInfo
        将一批样本整理成批数据对象，供 DataLoader 自动调用。
    """

    def __init__(self, config_manager):
        """
        Args:
            config_manager (TTConfigManager): 全局配置管理器。
        """
        TTBaseDataset.__init__(self, config_manager)

    def __iter__(self) -> Iterator[BaseDataInfo]:
        """
        返回数据迭代器，逐条产出样本。

        Yields:
            BaseDataInfo: 单样本对象，必须是 ``BaseDataInfo`` 的子类。

        Note:
            子类必须实现此方法；基类不提供默认实现。
        """
        raise NotImplementedError

    def __add__(self, other: "TTIterableDataset") -> "TTChainDataset":
        """
        语法糖：支持 ``it = it1 + it2``，等价于 ``TTChainDataset([it1, it2])``。

        Args:
            other (TTIterableDataset): 另一个 Iterable-style 数据集，需与 ``self`` 拥有相同协议。

        Returns:
            TTChainDataset: 链式串联后的新数据集，迭代时按顺序无缝切换。
        """
        return TTChainDataset(self.config_manager, [self, other])


class TTConcatDataset(ConcatDataset):
    """
    把多个 TTMapDataset 按样本维度「首尾拼接」。

    行为
    ----
    len(concat) = sum(len(d) for d in datasets)
    concat[i]   = datasets[k][j]           # 自动定位到对应子集
    """

    def __init__(self, config_manager, datasets: list[TTMapDataset]):
        """
        将多个 ``TTMapDataset`` 首尾拼接成一个新的 Map-style 数据集。

        Args:
            config_manager: 全局配置管理器，供下游取用超参、环境信息等。
            datasets: 待拼接的 ``TTMapDataset`` 列表，按顺序拼接。
        """
        ConcatDataset.__init__(self, datasets)
        self.config_manager = config_manager

    def __getitem__(self, index) -> BaseDataInfo:
        """
        根据下标返回对应的单个样本（BaseDataInfo）。

        实际索引逻辑由 PyTorch ``ConcatDataset`` 完成，这里只做类型保持。
        """
        return ConcatDataset.__getitem__(self, index)

    def __len__(self) -> int:
        """
        返回拼接后数据集的总样本数，等于所有子数据集长度之和。
        """
        return super().__len__()

    def collate_fn(self, batch: list[BaseDataInfo]) -> BaseBatchDataInfo:
        # 所有子集协议相同，用谁都可以
        if not self.datasets:
            raise RuntimeError("TTConcatDataset 中没有任何子数据集")
        return self.datasets[0].collate_fn(batch)  # type: ignore[arg-type]


class TTChainDataset(ChainDataset):
    """
    把多个 TTIterableDataset 按迭代顺序「链式串联」。

    行为
    ----
    for x in chain:  # 先产出 datasets[0] 所有元素，再 datasets[1] ...
    """

    def __init__(self, config_manager, datasets: list[TTIterableDataset]):
        """
        将多个 ``TTIterableDataset`` 按迭代顺序链式串联。

        Args:
            config_manager: 全局配置管理器。
            datasets: 待串联的 ``TTIterableDataset`` 列表，迭代时按序产出。
        """
        ChainDataset.__init__(self, datasets)
        self.config_manager = config_manager

    def __iter__(self) -> Iterator[BaseDataInfo]:
        """
        返回一个迭代器，按顺序依次产出所有子数据集中的 BaseDataInfo 样本。
        """
        return ChainDataset.__iter__(self)

    def __len__(self) -> int:
        """
        返回链式数据集的总长度（若所有子集都可求长度），否则可能抛出 TypeError。
        """
        return super().__len__()

    def collate_fn(self, batch: list[BaseDataInfo]) -> BaseBatchDataInfo:
        if not self.datasets:
            raise RuntimeError("TTChainDataset 中没有任何子数据集")
        return self.datasets[0].collate_fn(batch)  # type: ignore[arg-type]


class TTStackDataset(StackDataset):
    """
    把多个 TTMapDataset 在「样本维度堆叠」成 tuple。

    行为
    ----
    stack[i] = (ds0[i], ds1[i], ...)  # 每个位置都是 BaseDataInfo
    """

    def __init__(self, config_manager, *args: TTMapDataset, **kwargs: TTMapDataset):
        """
        将多个 ``TTMapDataset`` 在样本维度按位置堆叠成元组。

        行为:
            ``stack[i] = (ds0[i], ds1[i], ...)``，每个元素都是 BaseDataInfo。

        Args:
            config_manager: 全局配置管理器。
            *args, **kwargs: 待堆叠的 ``TTMapDataset`` 实例。
        """
        StackDataset.__init__(self, *args, **kwargs)
        self.config_manager = config_manager

    def __getitem__(self, index) -> tuple[BaseDataInfo, ...]:
        """
        返回下标对应位置所有子数据集样本组成的元组。

        Args:
            index: 样本下标。

        Returns:
            每个位置都是 BaseDataInfo 的元组。
        """
        return StackDataset.__getitem__(self, index)

    def __len__(self) -> int:
        """
        返回数据集长度，等于所有子数据集长度（要求长度一致）。
        """
        return super().__len__()

    def collate_fn(self, batch):
        """
        参数
        ----
        batch: list[tuple[BaseDataInfo, ...]]
            外层 list 长度 = batch_size；
            内层 tuple 长度 = 子数据集个数；
            每个元素是对应子数据集产出的 BaseDataInfo。

        返回
        ----
        tuple
            长度 == 子数据集个数；
            第 i 个元素是 self.datasets[i].collate_fn 返回的批对象。
        """
        # 1. 转置：把 [ (s0,s1,...), (s0,s1,...), ... ] 变成 [ [s0列表], [s1列表], ... ]
        transposed = list(zip(*batch))  # tuple of lists

        # 2. 每个子数据集各自整理自己的一批样本
        collated = []
        for ds, samples in zip(self.datasets, transposed):
            # 确保子数据集实现了 collate_fn
            if not hasattr(ds, "collate_fn"):
                raise TypeError(f"{type(ds).__name__} 未实现 collate_fn，无法参与 StackDataset 打包")
            collated.append(ds.collate_fn(list(samples)))

        return tuple(collated)


class TTSubset(Subset):
    """
    按索引取子集，仍保持 BaseDataInfo 协议。

    行为
    ----
    subset[i] = dataset[indices[i]]  # 单样本类型不变
    """

    def __init__(self, config_manager, dataset: TTMapDataset, indices: Sequence[int]):
        """
        根据给定下标序列，从原数据集中抽取子集，仍保持 BaseDataInfo 协议。

        Args:
            config_manager: 全局配置管理器。
            dataset: 原 ``TTMapDataset``。
            indices: 用于抽样的下标序列，可以是 list、tuple、ndarray 等。
        """
        Subset.__init__(self, dataset, indices)
        self.config_manager = config_manager

    def __getitem__(self, index) -> BaseDataInfo:
        """
        返回子集中第 ``index`` 个样本（BaseDataInfo），
        实际对应原数据集的 ``indices[index]`` 位置。
        """
        return Subset.__getitem__(self, index)

    def __len__(self) -> int:
        """
        返回子集大小，等于 ``len(indices)``。
        """
        return super().__len__()

    def collate_fn(self, batch: list[BaseDataInfo]) -> BaseBatchDataInfo:
        return self.dataset.collate_fn(batch)  # type: ignore[arg-type]


class TTTensorDataset(TTBaseDataset, TensorDataset):
    """
    TensorDataset 的 TT 包装。

    注意
    ----
    本类**不**返回 BaseDataInfo，仅作裸张量容器；如需协议兼容，请在外层再包一层
    MapDataset 把 Tensor -> BaseDataInfo。
    """

    def __init__(self, config_manager, *tensors: torch.Tensor):
        """
        对 PyTorch ``TensorDataset`` 的 TT 封装，仅作为裸张量容器使用。

        注意:
            本类 ``__getitem__`` 返回的是张量元组，**不**是 BaseDataInfo；
            如需协议兼容，请在外层再包一层 ``TTMapDataset`` 做转换。

        Args:
            config_manager: 全局配置管理器。
            *tensors: 若干 ``torch.Tensor``，要求第一维长度相同。
        """
        TTBaseDataset.__init__(self, config_manager)
        TensorDataset.__init__(self, *tensors)

    def __getitem__(self, index) -> tuple[torch.Tensor, ...]:
        """
        根据下标返回对应的张量元组。

        Args:
            index: 样本下标。

        Returns:
            一个包含各输入张量第 ``index`` 个元素的元组。
        """
        return super().__getitem__(index)

    def __len__(self) -> int:
        """
        返回数据集样本数，等于第一维张量的长度。
        """
        return super().__len__()

    def collate_fn(self, batch:list[tuple[torch.Tensor, ...]])->tuple[torch.Tensor, ...]:
        return torch.utils.data.default_collate(batch)
