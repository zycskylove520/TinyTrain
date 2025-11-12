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

from torch import nn, Tensor

from tinytrain.cfg import TTModuleRegistry


@TTModuleRegistry.register
class Concat(nn.Module):
    """
    Concatenate a list of tensors along a specified dimension.

    Args:
        dimension (int): The dimension along which the tensors will be concatenated. Default is 1.

    Raises:
        TypeError: If ``x`` is not an iterable of ``torch.Tensor``.
        ValueError: If ``x`` is empty or tensor shapes mismatch
                     (non-concat dimensions must be identical).

    Example:
        >>> concat_module = Concat(dimension=1)
        >>> tensor1 = torch.randn(2, 3)
        >>> tensor2 = torch.randn(2, 4)
        >>> result = concat_module([tensor1, tensor2])
        >>> print(result.shape)
        torch.Size([2, 7])
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.d = dim

    def forward(self, x: list[Tensor] | tuple[Tensor, ...]) -> Tensor:
        """Perform concatenation.

        Args:
            x: Sequence (list or tuple) of tensors to concatenate.

        Returns:
            Concatenated tensor.
        """
        if not x:  # 空序列
            raise ValueError("Concat needs at least one tensor.")
        if not all(isinstance(t, Tensor) for t in x):
            raise TypeError("All elements must be torch.Tensor.")

        return torch.cat(x, dim=self.d)


@TTModuleRegistry.register
class Add(nn.Module):
    """Element-wise addition of an arbitrary number of tensors.

    Raises:
        TypeError: If ``x`` is not an iterable of ``torch.Tensor``.
        ValueError: If ``x`` is empty or tensor shapes do not match.

    Example::
        >>> add = Add()
        >>> a = torch.randn(2, 3)
        >>> b = torch.randn(2, 3)
        >>> out = add([a, b])
        >>> out.shape
        torch.Size([2, 3])
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: list[Tensor] | tuple[Tensor, ...]) -> Tensor:
        """Perform element-wise addition.

        Args:
            x: Sequence (list or tuple) of tensors to add.

        Returns:
            Sum tensor.
        """
        if not x:
            raise ValueError("Add needs at least one tensor.")
        if not all(isinstance(t, Tensor) for t in x):
            raise TypeError("All elements must be torch.Tensor.")

        return torch.sum(torch.stack(x, dim=0), dim=0)


@TTModuleRegistry.register
class Combine(nn.Module):
    """Identity module that returns the input tensor list unchanged.

    Typically used as a placeholder to keep the graph structure when
    multiple tensors need to be passed downstream as a list.

    Raises:
        TypeError: If any element of ``x`` is not a ``torch.Tensor``.

    Example::
        >>> combine = Combine()
        >>> tensors = [torch.randn(2, 3), torch.randn(2, 4)]
        >>> out = combine(tensors)
        >>> len(out), out[0].shape, out[1].shape
        (2, torch.Size([2, 3]), torch.Size([2, 4]))
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: list[Tensor] | tuple[Tensor, ...]) -> list[Tensor]:
        """Return the input sequence of tensors as a list.

        Args:
            x: Sequence (list or tuple) of tensors.

        Returns:
            The same sequence converted to a list.
        """
        if not all(isinstance(t, Tensor) for t in x):
            raise TypeError("All elements must be torch.Tensor.")
        return list(x)


@TTModuleRegistry.register
class Flatten(nn.Module):
    """Flatten all dimensions except the batch (first) dimension.

    Equivalent to ``x.view(x.size(0), -1)``.

    Example::
        >>> flat = Flatten()
        >>> x = torch.randn(3, 4, 5, 6)
        >>> out = flat(x)
        >>> out.shape
        torch.Size([3, 120])
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Flatten the tensor.

        Args:
            x: Input tensor of any shape.

        Returns:
            2-D tensor with shape ``(x.size(0), -1)``.
        """
        return x.flatten(start_dim=1)
