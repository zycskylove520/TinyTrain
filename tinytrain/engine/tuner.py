from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Dict, Any
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

if TYPE_CHECKING:
    from .core import Core


class TTBaseHPOEnv(gym.Env):
    def __init__(self, core: Core, model_scale=None, model=None):
        super().__init__()
        self.core, self.model_scale, self.model = core, model_scale, model
        self.prev = 0.0
        self.history = []

        # 1. 子类返回 {link_type: {param: meta}}
        self.param_tree = self.define_params()

        # 2. 扁平化以便构建 Box
        self.flat_meta = []
        self.flat_keys = []  # [(link_type, param_name), ...]
        for link_type, group in self.param_tree.items():
            for pname, meta in group.items():
                self.flat_keys.append((link_type, pname))
                self.flat_meta.append(meta)

        lows = [m["low"] for m in self.flat_meta]
        highs = [m["high"] for m in self.flat_meta]
        self.action_space = spaces.Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(0, 1, (1,), np.float32)

        self.num = 0

    # ---------- 子类实现 ----------
    def define_params(self) -> Dict[str, Dict[str, Dict]]:
        """
        返回: {link_type: {param_name: meta}}
        meta = {"type": "continuous"|"discrete"|..., "low":..., "high":..., "choices":...}
        """
        return {
            "core": {
                "batch_size": {"type": "discrete", "choices": [16, 32, 48, 64], "low": 0, "high": 3},
                "optimizer": {"type": "discrete", "choices": ["SGD", "Adam", "AdamW", "Adamax", "NAdam", "RAdam", "RMSprop", "Adadelta", "Adagrad"], "low": 0, "high": 8},
                "scheduler": {"type": "discrete", "choices": ["auto", "LinearLR", "CosineLR", "ExponentialLR", "StepLR", "MultiStepLR"], "low": 0, "high": 5},
                "lr0": {"type": "continuous", "low": 1e-5, "high": 1e-2},
                "lr1": {"type": "continuous", "low": 1e-5, "high": 1e-2},
                "l1_norm": {"type": "continuous", "low": 0.0, "high": 1e-3},
                "momentum": {"type": "continuous", "low": 0.01, "high": 0.95},
                "weight_decay": {"type": "continuous", "low": 0.0, "high": 1e-4},
            }
        }

    # ---------------------------------

    def _decode_action(self, action: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """解码 -> {link_type: {param: value}}"""
        grouped = {}
        for (link_type, pname), meta, val in zip(self.flat_keys, self.flat_meta, action):
            grouped.setdefault(link_type, {})
            if meta["type"] == "continuous":
                grouped[link_type][pname] = float(np.clip(val, meta["low"], meta["high"]))
            elif meta["type"] == "discrete":
                choices = meta["choices"]
                idx = int(np.clip(int(val), 0, len(choices) - 1))
                grouped[link_type][pname] = choices[idx]
            elif meta["type"] == "bool":
                grouped[link_type][pname] = bool(val > 0.5)
            else:
                raise ValueError(meta["type"])
        return grouped

    def step(self, action):
        self.num += 1
        print(f"action num: {self.num}")

        decoded = self._decode_action(action)

        # 按 link_type 分别调用
        for link_type, params in decoded.items():
            self.core.set_config_overrides(link_type=link_type, **params)

        # 训练 & 记录
        self.core.train(model_scale=self.model_scale, model=self.model)
        csv_path = Path(self.core.trainer.save_dir) / "result.csv"
        fitness = float(pd.read_csv(csv_path)["fitness"].max())

        # 扁平化记录
        flat = {}
        for link_type, params in decoded.items():
            for k, v in params.items():
                flat[f"{link_type}.{k}"] = v
        flat["fitness"] = fitness
        self.history.append(flat)

        reward = fitness - self.prev - 0.001
        self.prev = fitness
        return np.array([fitness], dtype=np.float32), reward, True, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.prev = 0.0
        return np.array([self.prev], dtype=np.float32), {}

    def close(self):
        config_core = self.core.config_manager.core
        save_dir = Path(config_core["save_dir"]).resolve()
        project_name = config_core["project_name"] or "default_project"
        config_core["project_name"] = project_name
        out_dir = save_dir / project_name / "tune"

        pd.DataFrame(self.history).to_csv(out_dir / "hpo_history.csv", index=False)
        if self.history:
            best = max(self.history, key=lambda x: x["fitness"])
            with open(out_dir / "best_config.json", "w") as f:
                json.dump(best, f, indent=2)
            print("✅ Best config saved to", out_dir / "best_config.json")


def make_env(core: Core, model_scale, model: str | Path = None):
    """工厂函数：返回一个无参数的 env构造器，供 SB3 的 SubprocVecEnv使用"""

    def _init():
        env = TTBaseHPOEnv(core=core, model_scale=model_scale, model=model)
        return env

    return _init


class BaseTuner:
    def __init__(self, core: Core, model_scale, model: str | Path = None):
        config_core = core.config_manager.core
        save_dir = Path(config_core["save_dir"]).resolve()
        project_name = config_core["project_name"] or "default_project"
        config_core["project_name"] = project_name
        self.save_dir = save_dir / project_name / "tune"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 创建环境
        self.env = DummyVecEnv([make_env(core, model_scale, model)])

    def tune(self, total_timesteps: int = 2048):
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

        # PPO智能体
        ppo_model = PPO(
            "MlpPolicy",
            self.env,
            device="cpu",
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=16,
            batch_size=16,
            gamma=0.99,
            verbose=1,
            tensorboard_log=str(self.save_dir / "tensorboard")
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(total_timesteps // 10, 1),
            save_path=str(self.save_dir / "checkpoints"),
            name_prefix="ppo_model"
        )
        try:
            ppo_model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
            ppo_model.save(self.save_dir / "final_model")
        finally:
            self.env.close()
