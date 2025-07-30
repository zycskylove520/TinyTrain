"""
分布式训练命令生成工具
"""

import socket


def find_available_port():
    """
    在本地环回地址上随机申请一个空闲 TCP 端口并立即释放，返回端口号。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # port


def generate_ddp_command(trainer, nproc_per_node: int):
    """
    根据训练器配置生成 torchrun 命令行参数，用于启动分布式训练。

    Args:
        trainer:         BaseTrainer 实例，包含 main_script_path 与 config_manager
        nproc_per_node:  当前节点 GPU 数量

    Returns:
        List[str]: 可直接传递给 subprocess.run 的命令
    """

    file = trainer.main_script_path
    nodes = trainer.config_manager.core["nodes"]
    node_rank = trainer.config_manager.core["node_rank"]
    master_addr = trainer.config_manager.core["master_addr"]
    master_port = trainer.config_manager.core["master_port"]
    if master_port <= 0:
        master_port = find_available_port()
    cmd = ["torchrun", "--nnodes", f"{nodes}", "--node_rank", f"{node_rank}", "--nproc_per_node", f"{nproc_per_node}", "--master_addr", f"{master_addr}", "--master_port", f"{master_port}", file]
    return cmd
