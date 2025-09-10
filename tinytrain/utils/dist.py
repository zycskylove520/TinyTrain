"""
分布式训练命令生成工具
"""
from __future__ import annotations

import socket
import subprocess
import sys
import torch
import torch.distributed as dist

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
        LOGGER.info("Launching DDP with command:", " ".join(cmd))
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


class AllGatherWithGradFunc(torch.autograd.Function):
    """AllGather op with gradient backward"""

    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        dist.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        rank = dist.get_rank()
        grad_out = grad_list[rank]

        dist_ops = [
            dist.reduce(grad_out, rank, dist.ReduceOp.SUM, async_op=True)
            if i == rank
            else dist.reduce(
                grad_list[i], i, dist.ReduceOp.SUM, async_op=True
            )
            for i in range(dist.get_world_size())
        ]
        for _op in dist_ops:
            _op.wait()

        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return (grad_out, *[None for _ in range(len(grad_list))])


def all_gather_with_grad(tensor, *gather_list):
    """AllGather op with gradient backward"""
    return AllGatherWithGradFunc.apply(tensor, *gather_list)


class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        """ """
        batch_size = logits.size(0)
        # for numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        # local to global
        dist.all_reduce(max_logits, dist.ReduceOp.MAX)
        logits = logits - max_logits
        logits = logits.exp()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        # local to global
        dist.all_reduce(sum_logits_exp, dist.ReduceOp.SUM)
        logits = logits / sum_logits_exp
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1, device=logits.device)
        loss[index] = logits[index].gather(1, label[index])
        dist.all_reduce(loss, dist.ReduceOp.SUM)
        ctx.save_for_backward(index, logits, label)
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        batch_size = logits.size(0)
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device
        )
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        logits.div_(batch_size)
        return logits * loss_gradient.item(), None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part)
