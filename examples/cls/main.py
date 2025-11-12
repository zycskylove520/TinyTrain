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

from tinytrain.models.yolo import YOLOCore


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
    yolo = YOLOCore(link_file=r"D:\project\python_code\TinyTrain-main\example_projects\cls_copy\link.toml")
    # model = yolo.get_model()
    # train(yolo)
    # predict(yolo)
    # export(yolo)
