from pathlib import Path

from TinyTrain import YOLOCore


def train(model):
    # train
    model.set_config_overrides(
        link_type="core",
        task="detect",
        warmup_epochs=0,
        epochs=10,
        batch_size=16,
        lr0=0.01,
        lr1=0.01,
        scheduler="auto",
        workers=4
    )

    model.set_config_overrides(
        link_type="dataset",
        img_size=640,
        cache=False,
        path="../../datasets/firework"
    )
    # use_last_pt为True则自动搜索最新训练的pt文件进行训练
    model.train(model_scale='n', use_last_pt=True)


def predict(model):
    model.set_config_overrides(
        link_type="core",
        task="detect"
    )
    model.set_config_overrides(
        link_type="dataset"
    )
    results = model.predict(
        use_last_pt=True,
        source=r"D:\project\python_code\TinyTrain-main\datasets\firework\images\val\2f730a7b97ef51dcbba8fe15aee67092.jpg",
        img_shape=(640, 640)
    )
    for result in results:
        print(result)


def export(model):
    model.set_config_overrides(
        link_type="core",
        task="detect"
    )
    model.export(
        use_last_pt=True,
        backend="onnx",
        input_shapes=[(1, 3, 640, 640)],
        # jit_export=True
    )


if __name__ == '__main__':
    yolo = YOLOCore(link_file=r"D:\project\python_code\TinyTrain-main\projects\det\link.toml")
    train(yolo)
    # predict(yolo)
    # export(yolo)
