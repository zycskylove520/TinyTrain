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

"""
分布式训练命令生成工具
"""
from __future__ import annotations

import socket
import subprocess
import sys

from pathlib import Path
from typing import List

from tinytrain.utils import LOGGER


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def find_available_port(host: str = "127.0.0.1") -> int:
    """
    在本地环回地址上随机申请一个空闲 TCP 端口并立即释放，返回端口号。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]  # port


class DDPLauncher:
    """
    通用分布式启动器，支持：
      - torchrun（推荐）
      - python -m torch.distributed.launch（旧）
      - Slurm（未来扩展）
    """

    def __init__(
            self,
            main_script: Path | str,
            nproc_per_node: int,
            nnodes: int = 1,
            node_rank: int = 0,
            master_addr: str = "127.0.0.1",
            master_port: int = 29500,
    ):
        self.main_script = Path(main_script).resolve()
        self.nproc_per_node = nproc_per_node
        self.nnodes = nnodes
        self.node_rank = node_rank
        self.master_addr = master_addr
        self.master_port = master_port

    # ---------------------------------------------------------------------
    # 生成命令
    # ---------------------------------------------------------------------
    def build_torchrun_cmd(self) -> List[str]:
        """返回 torchrun 命令列表"""
        return [
            "torchrun",
            "--nproc_per_node", str(self.nproc_per_node),
            "--nnodes", str(self.nnodes),
            "--node_rank", str(self.node_rank),
            "--master_addr", self.master_addr,
            "--master_port", str(self.master_port),
            str(self.main_script),
        ]

    def build_launch_cmd(self) -> List[str]:
        """返回旧的 launch 命令（兼容 PyTorch 1.x）"""
        return [
            sys.executable,
            "-m",
            "torch.distributed.launch",
            "--nproc_per_node", str(self.nproc_per_node),
            "--nnodes", str(self.nnodes),
            "--node_rank", str(self.node_rank),
            "--master_addr", self.master_addr,
            "--master_port", str(self.master_port),
            "--use_env",
            str(self.main_script)
        ]

    # ---------------------------------------------------------------------
    # 启动入口
    # ---------------------------------------------------------------------
    def run(self, use_torchrun: bool = True) -> None:
        """执行启动"""

        # 先确保端口可用
        self.resolve_port()

        cmd = self.build_torchrun_cmd() if use_torchrun else self.build_launch_cmd()
        LOGGER.info("Launching DDP with command:" + " ".join(cmd))
        subprocess.run(cmd, check=True)

    def resolve_port(self):
        """
        返回可用端口；单机时自动换一个，多机时直接抛错。
        """
        if not is_port_in_use(self.master_port, self.master_addr):
            return

        if self.nnodes > 1:
            raise RuntimeError(
                f"Port {self.master_port} already in use and nnodes={self.nnodes}>1, "
                f"cannot auto-switch. Please set a free port via MASTER_PORT."
            )

        new_port = find_available_port(self.master_addr)
        LOGGER.warning(f"Port {self.master_port} occupied, switch to {new_port}")
        self.master_port = new_port
