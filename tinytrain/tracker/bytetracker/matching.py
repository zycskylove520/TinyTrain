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

"""
匹配工具模块：基于 IoU 的线性分配实现
提供两条轨迹集合之间的最优匹配及未匹配项拆分功能，零外部依赖（除 lap 求解器）。
"""

import lap
import numpy as np

from tinytrain.utils.box_utils import box_iou_numpy


def linear_assignment(cost_matrix, thresh):
    """
    使用 Jonker-Volgenant 算法求解带阈值约束的线性分配问题。

    Args:
        cost_matrix: 代价矩阵，形状 (M, N)
        thresh: 最大允许代价，超过该阈值的边会被视为不可匹配

    Returns:
        matches:   ndarray, 形状 (K, 2)，成功匹配的 (行索引, 列索引) 对
        unmatched_a: ndarray，未匹配的行索引
        unmatched_b: ndarray，未匹配的列索引
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def iou_distance(atracks, btracks):
    """
    计算两条轨迹集合之间的 IoU 距离矩阵（1 - IoU）。

    Args:
        atracks: list[STrack]，轨迹 A
        btracks: list[STrack]，轨迹 B

    Returns:
        cost_matrix: ndarray，形状 (len(atracks), len(btracks))
                     值为 1 - IoU，范围 [0, 1]
    """
    ious = np.zeros((len(atracks), len(btracks)), dtype=np.float64)
    if ious.size != 0:
        atlbrs = [track.lxlyrxry for track in atracks]
        btlbrs = [track.lxlyrxry for track in btracks]
        ious = box_iou_numpy(
            np.ascontiguousarray(atlbrs, dtype=np.float32),
            np.ascontiguousarray(btlbrs, dtype=np.float32)
        )
    cost_matrix = 1 - ious

    return cost_matrix
