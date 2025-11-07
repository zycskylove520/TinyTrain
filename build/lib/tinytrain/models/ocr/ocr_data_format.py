import torch

from tinytrain.data.data_format import ImgDataInfo, ImgBatchDataInfo


class LPRDataInfo(ImgDataInfo):
    """
    分类任务专用单张图像数据容器。
    """

    def __init__(self,
                 label: list[int] | None = None,
                 length: int | None = None,
                 **kwargs
                 ) -> None:
        """
        Args:
            label (list[int] | None): 车牌字符索引列表。
            length (int | None): 车牌字符的长度
            **kwargs: 透传给父类。
        """
        super().__init__(**kwargs)
        self.label = label
        self.length = length


class LPRBatchDataInfo(ImgBatchDataInfo):
    def __init__(self,
                 target: torch.Tensor | None = None,
                 lengths: list[int] | None = None,
                 **kwargs
                 ) -> None:
        """
        初始化LPR批量数据信息。

        Args:
           target: LPR目标（PyTorch张量）
           lengths: 批次内车牌字符长度形成的列表
           **kwargs: 透传给父类。
        """
        super().__init__(**kwargs)
        self.target = target
        self.lengths = lengths
