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

from typing import Iterator, List

from tinytrain.data.data_format import TextDataInfo, BaseBatchDataInfo
from tinytrain.data.base.base_dataset import (
    TTMapDataset, TTIterableDataset, TTConcatDataset, TTChainDataset,
    TTStackDataset, TTSubset, TTTensorDataset
)


# ------------------------------------------------------------------
# 一个最小 ConfigManager 占位
# ------------------------------------------------------------------
class DummyConfigManager:
    pass


# ------------------------------------------------------------------
# 最小 Map-style 数据集
# ------------------------------------------------------------------
class DummyMapDataset(TTMapDataset):
    def __init__(self, length: int = 10):
        super().__init__(DummyConfigManager())
        self.length = length

    def __getitem__(self, index: int) -> TextDataInfo:
        return TextDataInfo(text=f"map_{index}")

    def __len__(self) -> int:
        return self.length

    def collate_fn(self, batch: List[TextDataInfo]) -> BaseBatchDataInfo:
        # 把 text 拼成列表即可
        texts = [item.text for item in batch]
        return BaseBatchDataInfo(data=texts)


# ------------------------------------------------------------------
# 最小 Iterable-style 数据集
# ------------------------------------------------------------------
class DummyIterableDataset(TTIterableDataset):
    def __init__(self, length: int = 10):
        super().__init__(DummyConfigManager())
        self.length = length

    def __iter__(self) -> Iterator[TextDataInfo]:
        for i in range(self.length):
            yield TextDataInfo(text=f"iter_{i}")

    def collate_fn(self, batch: List[TextDataInfo]) -> BaseBatchDataInfo:
        texts = [item.text for item in batch]
        return BaseBatchDataInfo(data=texts)


# ==================================================================
# 下面开始 pytest 用例
# ==================================================================
try:
    import pytest
except ImportError as e:
    raise ImportError(
        "TinyTrain 的测试依赖 pytest，请先安装：\n"
        "  pip install -U pytest\n"
    )


# ------------------------------------------------------------------
# 1. Map-style 基础能力
# ------------------------------------------------------------------
def test_map_dataset():
    ds = DummyMapDataset(5)
    assert len(ds) == 5
    assert ds[2].text == "map_2"
    batch = ds.collate_fn([ds[0], ds[1]])
    assert batch.data == ["map_0", "map_1"]


# ------------------------------------------------------------------
# 2. Iterable-style 基础能力
# ------------------------------------------------------------------
def test_iterable_dataset():
    ds = DummyIterableDataset(3)
    items = list(ds)
    assert len(items) == 3
    assert items[0].text == "iter_0"
    batch = ds.collate_fn(items[:2])
    assert batch.data == ["iter_0", "iter_1"]


# ------------------------------------------------------------------
# 3. ConcatDataset
# ------------------------------------------------------------------
def test_concat():
    ds1 = DummyMapDataset(3)
    ds2 = DummyMapDataset(2)
    concat = TTConcatDataset(DummyConfigManager(), [ds1, ds2])
    assert len(concat) == 5
    assert concat[0].text == "map_0"
    assert concat[3].text == "map_0"  # ds2 的第一条
    batch = concat.collate_fn([concat[0], concat[3]])
    assert batch.data == ["map_0", "map_0"]


# ------------------------------------------------------------------
# 4. ChainDataset
# ------------------------------------------------------------------
def test_chain():
    it1 = DummyIterableDataset(2)
    it2 = DummyIterableDataset(2)
    chain = TTChainDataset(DummyConfigManager(), [it1, it2])
    items = list(chain)
    assert len(items) == 4
    assert [x.text for x in items] == ["iter_0", "iter_1", "iter_0", "iter_1"]


# ------------------------------------------------------------------
# 5. StackDataset
# ------------------------------------------------------------------
def test_stack():
    ds1 = DummyMapDataset(3)
    ds2 = DummyMapDataset(3)
    stack = TTStackDataset(DummyConfigManager(), ds1, ds2)
    assert len(stack) == 3
    tup = stack[1]
    assert isinstance(tup, tuple)
    assert tup[0].text == "map_1"
    assert tup[1].text == "map_1"
    # 测试 collate
    batch = stack.collate_fn([stack[0], stack[1]])
    assert isinstance(batch, tuple)
    assert len(batch) == 2
    assert batch[0].data == ["map_0", "map_1"]
    assert batch[1].data == ["map_0", "map_1"]


# ------------------------------------------------------------------
# 6. Subset
# ------------------------------------------------------------------
def test_subset():
    ds = DummyMapDataset(10)
    subset = TTSubset(DummyConfigManager(), ds, indices=[2, 4, 6])
    assert len(subset) == 3
    assert subset[0].text == "map_2"
    assert subset[2].text == "map_6"
    batch = subset.collate_fn([subset[0], subset[1]])
    assert batch.data == ["map_2", "map_4"]


# ------------------------------------------------------------------
# 7. TTTensorDataset
# ------------------------------------------------------------------
def test_tensor_dataset():
    x = torch.randn(4, 3)
    y = torch.randint(0, 10, (4,))
    tensor_ds = TTTensorDataset(DummyConfigManager(), x, y)
    assert len(tensor_ds) == 4
    a, b = tensor_ds[2]
    assert a.shape == (3,)
    assert b.shape == ()
    # collate 用默认的
    batch = tensor_ds.collate_fn([tensor_ds[0], tensor_ds[1]])
    assert batch[0].shape == (2, 3)
    assert batch[1].shape == (2,)


# ------------------------------------------------------------------
# 8. 语法糖 __add__
# ------------------------------------------------------------------
def test_add_sugar():
    ds1 = DummyMapDataset(2)
    ds2 = DummyMapDataset(2)
    merged = ds1 + ds2
    assert isinstance(merged, TTConcatDataset)
    assert len(merged) == 4

    it1 = DummyIterableDataset(1)
    it2 = DummyIterableDataset(1)
    chained = it1 + it2
    assert isinstance(chained, TTChainDataset)
    assert len(list(chained)) == 2


# ==================================================================
# pytest 入口
# ==================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
