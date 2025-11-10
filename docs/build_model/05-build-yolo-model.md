# 学习如何构建YOLO风格的模型

该章节将引导你构建一个符合YOLO风格要求的模型，你可以学会：

1. 从零开始搭建一个符合YOLO风格的模型
2. 对已有YOLO模型进行改进

请确保在学习该章节前已学习前置章节：[学习更高级的模型构建技巧](04-advanced-config-tips)

## 入门

YOLO风格的模型使用YOLOModel类，该类在继承自TTConfigModel的基础上，为模型配置文件的scales字段新增了一个width字段，允许调整特征图的通道数。

使用YOLO风格来搭建一个基于配置文件构建的模型，如果需要使用width字段来控制通道大小，则需要满足以下要求：

1. network层，需要使用width字段来控制通道大小的entry层的args字段必须有out_channels参数。
2. network层，需要使用width字段来控制通道大小的flow层的args字段必须有in_channels和out_channels两个参数。
3. network层，需要使用width字段来控制通道大小的head层的args字段必须有in_channels参数。
4. 如果head层有args.nc参数，则会自动寻找self.config_manager.dataset["nc"]的值赋给args.nc。

其他方面与普通的基于配置文件的构建方式无差异。

要学习如何使用YOLOModel类构建模型，请参考tinytrain框架已实现好的yolo模型的模型文件来学习，这并不难。