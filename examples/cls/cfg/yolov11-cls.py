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
