import torch

from torch import nn, Tensor

from tinytrain.cfg.TT_register import TTModuleRegistry


@TTModuleRegistry.register
class Concat(nn.Module):
    """
    Concatenate a list of tensors along a specified dimension.

    Args:
        dimension (int): The dimension along which the tensors will be concatenated. Default is 1.

    Example:
        >>> concat_module = Concat(dimension=1)
        >>> tensor1 = torch.randn(2, 3)
        >>> tensor2 = torch.randn(2, 4)
        >>> result = concat_module([tensor1, tensor2])
        >>> print(result.shape)
        torch.Size([2, 7])
    """

    def __init__(self, dimension: int = 1):
        """
        Initializes the Concat module.

        Args:
            dimension (int): The dimension along which the tensors will be concatenated.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[Tensor]) -> Tensor:
        """
        Concatenates a list of tensors along the specified dimension.

        Args:
            x (List[Tensor]): A list of tensors to be concatenated.

        Returns:
            Tensor: The concatenated tensor.

        Raises:
            TypeError: If the input is not a list of tensors.
            ValueError: If the input list is empty.
        """
        # Check if the input is a list of tensors
        if not isinstance(x, list) or not all(isinstance(item, torch.Tensor) for item in x):
            raise TypeError("Input must be a list of tensors.")

        # Check if the input list is empty
        if len(x) == 0:
            raise ValueError("Input list cannot be empty.")

        return torch.cat(x, dim=self.d)


@TTModuleRegistry.register
class Add(nn.Module):
    """
    将任意多个输入tensor进行element-wise加法操作
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: list[Tensor]) -> Tensor:
        return torch.sum(torch.stack(x, dim=0), dim=0)


@TTModuleRegistry.register
class Combine(nn.Module):
    """
    将任意多个输入tensor作为列表返回
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: list[Tensor]) -> list[Tensor]:
        # 检查输入是否为tensor列表
        if not all(isinstance(item, torch.Tensor) for item in x):
            raise TypeError("All elements in the input list must be torch.Tensor.")
        return x
