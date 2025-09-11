from tinytrain import YOLOCore


def train(model):
    # train
    model.set_config_overrides(
        link_type="core",
        warmup_epochs=0,
        epochs=2,
        batch_size=16,
        lr0=0.01,
        lr1=0.01,
        scheduler="auto"
    )

    model.set_config_overrides(
        link_type="dataset",
        cache=False,
    )
    model.train(model_scale='n')


def predict(model):
    results = model.predict(
        backend="onnx",
        model=r"model.onnx",
        source=r"mnist\val\0\3.png",
        img_shape=(32, 32)
    )
    for result in results:
        print(result)


def export(model):
    model.export(
        use_best_pt=True,
        backend="onnx",
        input_shapes=[(1, 3, 32, 32)],
        # jit_export=True
    )


def tune(model):
    model.set_config_overrides(
        link_type="core",
        task="detect",
        epochs=10,
        warmup_epochs=0
    )
    results = model.tune(model_scale='n', pop_size=40, generations=20)
    print(results)


if __name__ == '__main__':
    yolo = YOLOCore(link_file=r"link.toml")
    train(yolo)
    # predict(yolo)
    # export(yolo)
