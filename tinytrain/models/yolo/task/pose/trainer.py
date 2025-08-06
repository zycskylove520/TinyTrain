from tinytrain.models.yolo.task.detect import YOLODetectionTrainer
from tinytrain.models.yolo.yolo_dataset import YOLOPoseDataset

class YOLOPoseTrainer(YOLODetectionTrainer):
    def build_dataset(self, mode="train"):
        if mode == "train":
            return YOLOPoseDataset(config_manager=self.config_manager,
                                   img_path=self.train_dir,
                                   mode="train"
                                   )
        elif mode == "val":
            return YOLOPoseDataset(config_manager=self.config_manager,
                                   img_path=self.val_dir,
                                   mode="val"
                                   )
        else:
            raise NotImplementedError
