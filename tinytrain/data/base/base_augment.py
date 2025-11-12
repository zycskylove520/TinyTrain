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

from __future__ import annotations

from tinytrain.data.data_format import BaseDataInfo


class TTBaseAugmentation:
    """
    所有数据增强/变换策略的抽象基类，定义统一接口：
    - augment : 通常返回一个 **离线增强流水线**（一次性生成多图，训练用）。
    - transform : 返回一个 **在线变换流水线**（每次读取样本时实时执行，验证/预测用）。
    子类按需实现其中之一或两者。
    """

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.augment = None
        self.transform = None

    def set_augment(self, *args, **kwargs):
        """离线增强接口，子类自定义流水线，默认空实现。"""
        pass

    def do_augment(self, sample: BaseDataInfo):
        """执行离线增强，默认空实现。"""
        return sample

    def set_transform(self, *args, **kwargs):
        """在线变换接口，子类自定义流水线，默认空实现。"""
        pass

    def do_transform(self, sample: BaseDataInfo):
        """执行在线变换，默认空实现。"""
        return sample
