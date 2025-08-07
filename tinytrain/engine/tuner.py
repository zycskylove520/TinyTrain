from __future__ import annotations

import json
import random
import traceback
import pandas as pd
import numpy as np

from typing import Dict, Any, List, Tuple
from pathlib import Path

from tinytrain.utils import LOGGER


class BaseTuner:
    """
    BaseTuner 是基于遗传算法（Genetic Algorithm, GA）的超参数自动调优框架。
    子类仅需覆写 build_param_tree() 即可灵活定义/扩展超参数搜索空间，
    其余 GA 流程（种群初始化、选择、交叉、变异、评估、精英保留等）已在基类完整实现。

    设计要点
    ----------
    1. 统一使用实数编码的 GA：连续参数直接保留浮点，离散参数用 [0, n-1] 的连续值间接索引。
    2. 与训练框架解耦：通过 core（训练器）提供的 set_config_overrides() 与 train() 接口注入超参数并执行训练。
    3. 训练结束后自动读取 result.csv 中的 fitness 列作为适应度。
    4. 每次调优结果自动持久化：hpo_history.csv（完整历史）+ best_config.json（最优个体）。
    5. 随机种子由 core.config_manager.core.seed 控制，保证可复现。

    示例
    ----
    >>> core = Core("link.toml") # type: ignore[arg-type]
    >>> tuner = BaseTuner(core, model_scale="s")
    >>> result = tuner.tune(pop_size=30, generations=50)
    >>> best_cfg = result["best_config"]
    """

    def __init__(self, core, model_scale):
        """
        初始化调优器，完成以下工作：
        1. 设定随机种子；
        2. 解析并创建保存目录；
        3. 构建扁平化后的超参数空间（self.flat_keys, self.flat_meta）。

        Args:
            core: 训练核心对象，需实现 set_config_overrides() 与 train()。
            model_scale (str): 模型规模，如 'n', 's', 'm', 'l', 'x'。
        """
        self.core = core
        self.model_scale = model_scale

        # 设定随机种子
        seed = getattr(core.config_manager.core, "seed", 42)
        random.seed(seed)
        np.random.seed(seed)

        # 保存路径
        cfg = core.config_manager.core
        save_dir = Path(cfg["save_dir"]).resolve()
        project_name = cfg["project_name"] or "default_project"
        self.save_dir = save_dir / project_name / "tune"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 解析参数空间
        self.param_tree = self.build_param_tree()
        self.flat_keys, self.flat_meta = self._flatten_param_tree(self.param_tree)

    # ------------------------------------------------------------------
    # 子类可覆写：新增/修改超参
    # ------------------------------------------------------------------
    @staticmethod
    def build_param_tree() -> Dict[str, Dict[str, Dict]]:
        """
        定义超参数搜索空间。

        返回的树形结构：
            {"link_type": {"param_name": meta_dict, ...}, ...}

        meta_dict 支持两种形式：
            - 连续型 {"type": "continuous", "low": float, "high": float}
            - 离散型 {"type": "discrete", "choices": list, "low": 0, "high": len(choices)-1}

        子类覆写时可新增/删除任意节点，无需修改其余代码。
        """
        return {
            "core": {
                "batch_size": {"type": "discrete", "choices": [16, 32, 64], "low": 0, "high": 2},
                "optimizer": {"type": "discrete", "choices": ["SGD", "Adam", "AdamW", "Adamax", "NAdam", "RAdam", "RMSprop", "Adadelta", "Adagrad"], "low": 0, "high": 8},
                "scheduler": {"type": "discrete", "choices": ["auto", "LinearLR", "CosineLR", "ExponentialLR", "StepLR", "MultiStepLR"], "low": 0, "high": 5},
                "lr0": {"type": "continuous", "low": 1e-5, "high": 1e-1},
                "lr1": {"type": "continuous", "low": 1e-5, "high": 1e-1},
                "l1_norm": {"type": "continuous", "low": 0.0, "high": 1e-3},
                "momentum": {"type": "continuous", "low": 0.01, "high": 0.95},
                "weight_decay": {"type": "continuous", "low": 0.0, "high": 1e-4},
            }
        }

    # ------------------------------------------------------------------
    # 以下不建议子类重写的方法
    # ------------------------------------------------------------------
    def tune(
            self,
            pop_size: int = 30,
            generations: int = 50,
            elite_ratio: float = 0.1,
            crossover_rate: float = 0.9,
            mutation_rate: float = 0.1,
            mutation_sigma: float = 0.1,
    ) -> Dict[str, Any]:
        """
        启动遗传算法搜索。

        Args:
            pop_size (int, optional): 种群规模，需 ≥3。默认 30。
            generations (int, optional): 迭代代数。默认 50。
            elite_ratio (float, optional): 精英保留比例。默认 0.1。
            crossover_rate (float, optional): 交叉概率。默认 0.9。
            mutation_rate (float, optional): 变异概率。默认 0.1。
            mutation_sigma (float, optional): 高斯变异标准差。默认 0.1。

        Returns:
            Dict[str, Any]: {"history": DataFrame 格式的历史记录, "best_config": 最优配置字典}
        """
        if pop_size < 3:
            raise ValueError("pop_size must be >= 3 for GA to work")

        history, best_config = self._ga_search(
            pop_size=pop_size,
            generations=generations,
            elite_ratio=elite_ratio,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            mutation_sigma=mutation_sigma,
        )
        self._save(history, best_config)
        return {"history": history, "best_config": best_config}

    @staticmethod
    def _flatten_param_tree(tree: Dict[str, Any]) -> Tuple[List[Tuple[str, str]], List[Dict]]:
        """
        将嵌套的超参数树扁平化为两个平行列表：
        keys: [(link_type, param_name), ...]
        meta: [meta_dict, ...]

        Args:
            tree (dict): build_param_tree() 返回的字典。

        Returns:
            Tuple[List[Tuple[str, str]], List[Dict]]: (keys, meta)
        """
        keys, meta = [], []
        for link_type, group in tree.items():
            for pname, m in group.items():
                keys.append((link_type, pname))
                meta.append(m)
        return keys, meta

    def _decode_vector(self, vector: List[float]) -> Dict[str, Dict[str, Any]]:
        """
        将 GA 个体（实数向量）解码为可直接注入训练器的配置字典。

        Args:
            vector (List[float]): 一维实数向量，长度与 flat_meta 相同。

        Returns:
            Dict[str, Dict[str, Any]]: 形如 {"core": {"lr0": 0.01, ...}, ...}
        """
        cfg = {}
        for (link_type, pname), m, val in zip(self.flat_keys, self.flat_meta, vector):
            cfg.setdefault(link_type, {})
            if m["type"] == "continuous":
                cfg[link_type][pname] = float(max(m["low"], min(m["high"], val)))
            elif m["type"] == "discrete":
                idx = int(np.clip(np.round(val), 0, len(m["choices"]) - 1))
                cfg[link_type][pname] = m["choices"][idx]
            else:
                raise ValueError(m["type"])
        return cfg

    # ------------------------------------------------------------------
    # 遗传算法实现
    # ------------------------------------------------------------------
    def _ga_search(
            self,
            pop_size: int,
            generations: int,
            elite_ratio: float,
            crossover_rate: float,
            mutation_rate: float,
            mutation_sigma: float,
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        遗传算法主循环，返回 (history, best_config)。

        流程：
        1. 初始化种群；
        2. 每代评估 → 记录历史；
        3. 选择、交叉、变异；
        4. 精英保留；
        5. 循环直到达到代数。

        Args:
            ...（同 tune 方法）

        Returns:
            Tuple[List[Dict], Dict[str, Any]]: (history, best_config)
        """
        population = [self._random_genome() for _ in range(pop_size)]
        elite_n = max(1, int(pop_size * elite_ratio))
        history = []
        best_ind = None
        best_fit = -float("inf")

        for gen in range(generations):
            fits = self._evaluate(population)

            # 记录历史
            for g, f in zip(population, fits):
                row = {"gen": gen, "fitness": f, **self._decode_vector(g.tolist())}
                history.append(row)

            gen_best = max(fits)
            if gen_best > best_fit:
                best_fit = gen_best
                best_ind = population[np.argmax(fits)]
            print(f"Gen {gen:03d} | best={gen_best:.4f}")

            # 精英 + 遗传操作
            sorted_pairs = sorted(zip(population, fits), key=lambda x: x[1], reverse=True)
            new_pop = [p for p, _ in sorted_pairs][:elite_n]

            while len(new_pop) < pop_size:
                p1 = self._tournament_select(population, fits)
                p2 = self._tournament_select(population, fits)
                c1, c2 = self._crossover(p1, p2, crossover_rate)
                new_pop.append(self._mutate(c1, mutation_rate, mutation_sigma))
                if len(new_pop) < pop_size:
                    new_pop.append(self._mutate(c2, mutation_rate, mutation_sigma))
            population = new_pop

        best_config = self._decode_vector(best_ind.tolist())
        return history, best_config

    # 遗传算法内部工具
    def _random_genome(self) -> np.ndarray:
        """
        根据 flat_meta 随机生成一条染色体（实数向量）。

        Returns:
            np.ndarray: 一维实数向量。
        """
        vec = []
        for m in self.flat_meta:
            if m["type"] == "continuous":
                vec.append(np.random.uniform(m["low"], m["high"]))
            elif m["type"] == "discrete":
                vec.append(np.random.uniform(0, len(m["choices"])))
            else:
                raise ValueError(m["type"])
        return np.array(vec, dtype=np.float32)

    def _evaluate(self, pop: List[np.ndarray]) -> List[float]:
        """
        评估整个种群的适应度。

        对于每个个体：
        1. 解码为配置；
        2. 注入 core；
        3. 启动训练；
        4. 读取 result.csv 的 fitness 列作为适应度；
        5. 异常时返回 -inf。

        Args:
            pop (List[np.ndarray]): 种群，每个元素是一条染色体。

        Returns:
            List[float]: 各染色体的适应度列表。
        """
        fitnesses = []
        for genome in pop:
            try:
                cfg = self._decode_vector(genome.tolist())
                LOGGER.info(f"[GA-eval] Config to be trained: {json.dumps(cfg, indent=2)}")
                for link_type, params in cfg.items():
                    self.core.set_config_overrides(link_type=link_type, **params)
                self.core.train(model_scale=self.model_scale)

                csv_path = Path(self.core.trainer.save_dir) / "result.csv"  # type: ignore[arg-type]
                if not csv_path.exists():
                    raise FileNotFoundError(csv_path)
                df = pd.read_csv(csv_path)
                if df.empty or "fitness" not in df.columns:
                    raise ValueError("Empty or invalid result.csv")

                fit = float(df["fitness"].max())
            except Exception as e:
                print(f"[GA-eval] genome failed: {e}")
                traceback.print_exc()
                fit = -float("inf")
            fitnesses.append(fit)
        return fitnesses

    def _tournament_select(self, pop, fits, k: int = 3):
        """
        锦标赛选择算子。

        Args:
            pop: 种群列表。
            fits: 对应适应度列表。
            k (int, optional): 每次锦标赛抽取的个体数。默认 3。

        Returns:
            np.ndarray: 选中的父代染色体。
        """
        k = min(k, len(pop))
        idx = random.sample(range(len(pop)), k)
        best_i = max(idx, key=lambda i: fits[i])
        return pop[best_i]

    def _crossover(self, p1, p2, crossover_rate: float):
        """
        均匀交叉（Uniform Crossover）。

        Args:
            p1, p2 (np.ndarray): 两条父代染色体。
            crossover_rate (float): 交叉概率。

        Returns:
            Tuple[np.ndarray, np.ndarray]: (child1, child2)
        """
        if random.random() > crossover_rate:
            return p1.copy(), p2.copy()
        alpha = np.random.rand(len(p1))
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1
        return c1, c2

    def _mutate(self, ind, mutation_rate: float, mutation_sigma: float):
        """
        高斯变异，并强制将值域裁剪到合法范围内。

        Args:
            ind (np.ndarray): 单条染色体。
            mutation_rate (float): 变异概率。
            mutation_sigma (float): 高斯噪声标准差。

        Returns:
            np.ndarray: 变异后的染色体。
        """
        if random.random() > mutation_rate:
            return ind.copy()
        noise = np.random.normal(0, mutation_sigma, size=ind.shape)
        mutant = ind + noise
        for i, m in enumerate(self.flat_meta):
            if m["type"] == "continuous":
                mutant[i] = max(m["low"], min(m["high"], mutant[i]))  # type: ignore[arg-type]
            elif m["type"] == "discrete":
                # 保持连续值，仅限制在 [0, len-1]
                mutant[i] = max(0.0, min(len(m["choices"]) - 1, mutant[i]))  # type: ignore[arg-type]
        return mutant.astype(np.float32)

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    def _save(self, history: List[Dict], best_config: Dict):
        """
        持久化调优结果。

        Args:
            history (List[Dict]): 完整历史记录。
            best_config (Dict): 最优个体解码后的配置。
        """
        pd.DataFrame(history).to_csv(self.save_dir / "hpo_history.csv", index=False)
        with open(self.save_dir / "best_config.json", "w") as f:
            json.dump(best_config, f, indent=2, default=str)  # type: ignore[arg-type]
        print("✅ Best config saved to", self.save_dir / "best_config.json")
