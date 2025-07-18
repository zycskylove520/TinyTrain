import os
import pickle
import socket
import tempfile

from TinyTrain.global_var import ASSETS_PATH, ROOT


def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # port


def generate_ddp_command(trainer, nproc_per_node: int):
    """Generates and returns command for distributed training."""
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218

    file = trainer.main_script_path
    nodes = trainer.config_manager.core["nodes"]
    node_rank = trainer.config_manager.core["node_rank"]
    master_addr = trainer.config_manager.core["master_addr"]
    master_port = trainer.config_manager.core["master_port"]
    if master_port <= 0:
        master_port = find_available_port()
        # sys.executable, "-m",sys.executable, "-m", dist_cmd
    cmd = ["torchrun", "--nnodes", f"{nodes}", "--node_rank", f"{node_rank}", "--nproc_per_node", f"{nproc_per_node}", "--master_addr", f"{master_addr}", "--master_port", f"{master_port}", file]
    return cmd
