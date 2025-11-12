name = "YOLOv11-cls"
scale = "n"

scales = {
    "n": {"depth": 0.25, "summary": "151 layers, 1543914 parameters, 1543914 gradients, 0.4 GFLOPs, for a 224×224 input."},
    "s": {"depth": 0.50, "summary": "xxxx"}
}

network = {
    0: {"type": "entry",
        "module": "CBA",
        "repeat": 1,
        "from": -1,
        "args": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": 2}
        },
    1: {"type": "entry",}
}
