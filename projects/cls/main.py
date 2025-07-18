from TinyTrain import YOLOCore


def train(model):
    # train
    model.set_config_overrides(
        link_type="core",
        warmup_epochs=0,
        epochs=1,
        batch_size=16,
        lr0=0.01,
        lr1=0.01,
        scheduler="auto",
    )

    model.set_config_overrides(
        link_type="dataset",
        cache=False,
    )
    model.train(model_scale='n', use_last_pt=False)


def predict(model):
    results = model.predict(
        use_last_pt=True,
        source=r"D:\project\python_code\TinyTrain-main\datasets\firework\images\val\2f730a7b97ef51dcbba8fe15aee67092.jpg",
        img_shape=(32,32)
    )
    for result in results:
        print(result)


def export(model):
    model.export(
        use_last_pt=True,
        backend="onnx",
        input_shapes=[(1, 3, 32, 32)],
        # jit_export=True
    )


if __name__ == '__main__':
    yolo = YOLOCore(link_file=r"link.toml")
    train(yolo)
    # predict(yolo)
    # export(yolo)
