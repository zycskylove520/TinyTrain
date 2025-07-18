import cv2
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from matplotlib import pyplot as plt


class LabelInfo:
    """
    该类负责绘制跟标签相关的统计图表。
    """

    def __init__(self, num_classes: int, class_names: list[str], labels: np.ndarray, bboxes: np.ndarray):
        """

        @param labels: 标签类别，一维的ndarray，shape为：[box_num]。如：np.array([1,2,3])
        @param bboxes: 边界框，二维的ndarray，shape为：[box_num, 4],要求box_format为cxcywh
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.labels = labels
        self.bboxes = bboxes

    def label_histogram(self, save_dir: Path):
        """
        绘制标签直方图。
        """
        instances = np.arange(0, self.num_classes, dtype=int)
        for label in self.labels:
            instances[int(label)] += 1
        data = {
            'Classes': self.class_names,
            'Instances': instances
        }
        df = pd.DataFrame(data)

        # 根据标签数量自适应图表宽度
        width = self.num_classes
        height = 6  # 固定高度
        plt.figure(figsize=(width, height))  # 设置图表大小
        ax = sns.barplot(x='Classes', y='Instances', data=df, hue='Classes', palette=sns.color_palette("hsv", self.num_classes), legend=False)

        # 在每个条形上添加文本标签
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')

        # 设置x轴刻度
        plt.xticks(ticks=self.class_names, labels=self.class_names, rotation=90)  # 设置x轴刻度和标签
        # 设置y轴刻度
        plt.yticks([])  # 关闭y轴刻度
        # plt.yticks(ticks=instances, labels=instances)  # 设置y轴刻度和标签

        # 设置图表标题和标签
        plt.title('label distribution histogram')

        # 保存图像
        plt.savefig(save_dir / 'label_histogram.png', bbox_inches='tight', dpi=300)

        # 显示图形
        # plt.show()

    def boxes_statistics(self, axes):
        # 默认绘制在640x640的大小的图上
        width, height = 640, 640
        image = np.full((height, width, 3), fill_value=255, dtype=np.uint8)

        # 图像中心点
        center_x, center_y = width // 2, height // 2
        for bbox in self.bboxes:
            cx, cy, w, h = bbox
            # 将框的中心点移动到图像中心
            x_min = int(center_x - (w / 2) * width)
            y_min = int(center_y - (h / 2) * height)
            x_max = int(center_x + (w / 2) * width)
            y_max = int(center_y + (h / 2) * height)

            # 生成随机颜色
            color = (np.random.randint(0, 201), np.random.randint(0, 201), np.random.randint(0, 201))
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=1)

        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Boxes Statistics')
        axes[0, 0].axis('off')

    def cxcy_and_wh(self, axes):
        data1 = {
            "cx": [],
            "cy": [],
            "label": []
        }
        data2 = {
            "width": [],
            "height": [],
            "label": []
        }
        for bbox, label in zip(self.bboxes, self.labels):
            cx, cy, w, h = bbox
            data1["cx"].append(cx)
            data1["cy"].append(cy)
            data1["label"].append(label)
            data2["width"].append(w)
            data2["height"].append(h)
            data2["label"].append(label)

        # 转换为 Pandas DataFrame
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        # 绘制散点图
        sns.scatterplot(x='cx', y='cy', hue='label', data=df1, palette='viridis', legend=False, ax=axes[1, 0])
        sns.scatterplot(x='width', y='height', hue='label', data=df2, palette='viridis', legend=False, ax=axes[1, 1])

        # 设置 x 轴和 y 轴刻度范围为 0 到 1，间隔为 0.1
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_xticks(np.arange(0, 1.1, 0.1))
        axes[1, 0].set_yticks(np.arange(0, 1.1, 0.1))

        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_xticks(np.arange(0, 1.1, 0.1))
        axes[1, 1].set_yticks(np.arange(0, 1.1, 0.1))

        # 为每个散点图设置名称
        axes[1, 0].set_title('Center X and Y')
        axes[1, 1].set_title('Width and Height')

        # 设置 x 轴和 y 轴标签
        axes[1, 0].set_xlabel('Center X')
        axes[1, 0].set_ylabel('Center Y')

        axes[1, 1].set_xlabel('Width')
        axes[1, 1].set_ylabel('Height')

    def plot(self, save_dir: Path):
        # 标签直方图单独绘制
        self.label_histogram(save_dir)

        # 创建一个2行2列的子图网格
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        self.boxes_statistics(axes)
        self.cxcy_and_wh(axes)

        # 调整子图之间的间距
        fig.tight_layout()

        # 保存图像
        fig.savefig(save_dir / 'label_infos.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    from TinyTrain.data import YOLODetectionDataset
    from pathlib import Path

    img_path1 = Path(r"D:\project\python_code\TinyTrain-main\datasets\coco8\images\train")
    da = YOLODetectionDataset(img_path1, 640, True)
    samples = da.samples
    labels = []
    bboxes = []
    for sample in samples:
        labels.append(sample.label)
        bboxes.append(sample.bboxes)
    labels = np.concatenate(labels, axis=0)
    bboxes = np.concatenate(bboxes, axis=0)
    label_info = LabelInfo(num_classes=80, labels=labels, bboxes=bboxes)
    save_dir = Path(r"D:\project\python_code\TinyTrain-main\runs\default_project\detect\train_0")
    label_info.plot(save_dir)
