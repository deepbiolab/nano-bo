import torch
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import json
from datetime import datetime

from surrogate import TabPFNSurrogate
from pfnbo import (
    generate_initial_samples,
    generate_candidates,
    select_batch,
    compute_gradient_subspace,
)


class PFNBOActiveLearner:
    """
    同步 Active Learning 接口，适用于实验室环境
    支持保存/恢复功能，适合长期实验
    """
    
    def __init__(
        self,
        problem,
        mode: str = "fullspace",
        q: int = 1,
        acquisition_type: str = "ei",
        parallel_strategy: str = "ts",
        n_candidates: int = 1000,
        r: int = 15,
        n_grad_samples: int = 100,
        subspace_sampling: str = "sobol",
        reg_model_path: Optional[str] = None,
        n_estimators: int = 8,
        softmax_temperature: float = 0.9,
        fit_mode: str = "fit_with_cache",
        ucb_beta: float = 2.0,
        device: str = "cpu",
        seed: Optional[int] = None,
        save_dir: str = "./experiments",
        experiment_name: Optional[str] = None,
        **kwargs
    ):
        # 配置
        self.problem = problem
        self.mode = mode.lower()
        self.q = q
        self.acquisition_type = acquisition_type
        self.parallel_strategy = parallel_strategy
        self.n_candidates = n_candidates
        self.r = min(r, problem.dim) if mode == "subspace" else None
        self.n_grad_samples = n_grad_samples
        self.subspace_sampling = subspace_sampling
        self.ucb_beta = ucb_beta
        self.device = torch.device(device)
        self.seed = seed
        
        # 保存相关
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name or f"{problem.__class__.__name__}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = self.save_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型配置
        self.model_config = {
            "n_estimators": n_estimators,
            "device": str(self.device),
            "reg_model_path": reg_model_path,
            "softmax_temperature": softmax_temperature,
            "fit_mode": fit_mode,
            **kwargs
        }
        
        # 数据存储
        self.X_all = torch.empty((0, problem.dim), device=self.device, dtype=torch.float32)
        self.Y_all = torch.empty((0, 1), device=self.device, dtype=torch.float32)
        
        # 状态
        self.model = None
        self.iteration = 0
        self.best_Y = -float('inf')
        self.best_X = None
        
        # 历史记录
        self.history = {
            "best_values": [],
            "subspace_dims": [] if mode == "subspace" else None,
            "gradient_norms": [] if mode == "subspace" else None,
        }
        
        # Subspace 相关
        self.V_r_history = []
        self.H_history = []
        
        # 保存配置
        self._save_config()
        
    def _save_config(self):
        """保存实验配置"""
        config = {
            "problem": {
                "name": self.problem.__class__.__name__,
                "dim": self.problem.dim,
                "bounds": self.problem.bounds if hasattr(self.problem, 'bounds') else None,
            },
            "mode": self.mode,
            "q": self.q,
            "acquisition_type": self.acquisition_type,
            "parallel_strategy": self.parallel_strategy,
            "n_candidates": self.n_candidates,
            "r": self.r,
            "n_grad_samples": self.n_grad_samples,
            "subspace_sampling": self.subspace_sampling,
            "ucb_beta": self.ucb_beta,
            "device": str(self.device),
            "seed": self.seed,
            "model_config": self.model_config,
            "created_at": datetime.now().isoformat(),
        }
        
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
            
    def save_checkpoint(self, tag: Optional[str] = None):
        """
        保存当前状态
        
        Args:
            tag: 可选的标签（如 'day1', 'batch5' 等）
        """
        # 准备checkpoint数据
        checkpoint = {
            # 核心数据
            "X_all": self.X_all.cpu(),
            "Y_all": self.Y_all.cpu(),
            
            # 状态
            "iteration": self.iteration,
            "best_Y": self.best_Y,
            "best_X": self.best_X.cpu() if self.best_X is not None else None,
            
            # 历史
            "history": self.history,
            
            # Subspace相关
            "V_r_history": [V.cpu() for V in self.V_r_history] if self.V_r_history else [],
            "H_history": [H.cpu() for H in self.H_history] if self.H_history else [],

            # 添加pending points
            "_pending_X_unit": self._pending_X_unit.cpu() if hasattr(self, '_pending_X_unit') else None,
            
            # 元信息
            "seed": self.seed,
            "timestamp": datetime.now().isoformat(),
            "n_evaluated": len(self.X_all),
        }
        
        # 保存checkpoint
        filename = f"checkpoint_iter{self.iteration}"
        if tag:
            filename += f"_{tag}"
        filename += ".pt"
        
        torch.save(checkpoint, self.experiment_dir / filename)
        
        # 同时保存为latest
        torch.save(checkpoint, self.experiment_dir / "checkpoint_latest.pt")
        
        # 保存数据为CSV（方便查看）
        self._save_data_csv()
        
        print(f"Checkpoint saved: {filename}")
        
    def _save_data_csv(self):
        """保存数据为CSV格式"""
        # 准备数据
        X_np = self.X_all.cpu().numpy()
        Y_np = self.Y_all.cpu().numpy().flatten()
        
        # 创建DataFrame
        data = {}
        for i in range(X_np.shape[1]):
            data[f"x{i}"] = X_np[:, i]
        data["y"] = Y_np
        data["iteration"] = [i//self.q for i in range(len(Y_np))]
        
        df = pd.DataFrame(data)
        
        # 保存
        df.to_csv(self.experiment_dir / "all_data.csv", index=False)
        
        # 如果有原始空间的bounds，也保存缩放后的数据
        if hasattr(self.problem, 'scale'):
            X_scaled = self.problem.scale(self.X_all).cpu().numpy()
            data_scaled = {}
            for i in range(X_scaled.shape[1]):
                data_scaled[f"x{i}_scaled"] = X_scaled[:, i]
            data_scaled["y"] = Y_np
            
            df_scaled = pd.DataFrame(data_scaled)
            df_scaled.to_csv(self.experiment_dir / "all_data_scaled.csv", index=False)
            
    def load_checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None):
        """
        加载checkpoint
        
        Args:
            checkpoint_path: checkpoint文件路径，如果为None则加载最新的
        """
        if checkpoint_path is None:
            checkpoint_path = self.experiment_dir / "checkpoint_latest.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
            
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 恢复数据
        self.X_all = checkpoint["X_all"].to(self.device)
        self.Y_all = checkpoint["Y_all"].to(self.device)
        
        # 恢复状态
        self.iteration = checkpoint["iteration"]
        self.best_Y = checkpoint["best_Y"]
        self.best_X = checkpoint["best_X"].to(self.device) if checkpoint["best_X"] is not None else None
        
        # 恢复历史
        self.history = checkpoint["history"]
        
        # 恢复Subspace相关
        if "V_r_history" in checkpoint:
            self.V_r_history = checkpoint["V_r_history"]
        if "H_history" in checkpoint:
            self.H_history = checkpoint["H_history"]
            
        # 恢复随机种子
        if "seed" in checkpoint and checkpoint["seed"] is not None:
            self.seed = checkpoint["seed"]
            if self.seed is not None:
                from utils import set_seed
                set_seed(self.seed)

        # 恢复pending points
        if "_pending_X_unit" in checkpoint and checkpoint["_pending_X_unit"] is not None:
            self._pending_X_unit = checkpoint["_pending_X_unit"].to(self.device)

        # 重新fit模型
        if len(self.X_all) > 0:
            self.model = TabPFNSurrogate(
                train_X=self.X_all,
                train_Y=self.Y_all,
                **self.model_config
            )
            
        print(f"Checkpoint loaded: {checkpoint_path.name}")
        print(f"  - Iteration: {self.iteration}")
        print(f"  - Evaluated points: {len(self.X_all)}")
        print(f"  - Best value: {self.best_Y:.4f}")
        
    def export_next_experiments(self, n: Optional[int] = None, filename: Optional[str] = None):
        """
        导出下一批实验设计点
        
        Args:
            n: 实验点数量
            filename: 保存文件名（不含扩展名）
        
        Returns:
            导出的文件路径
        """
        # 获取推荐点
        X_unit, X_scaled = self.get_next_experiments(n)
        
        # 准备文件名
        if filename is None:
            filename = f"designs_iter{self.iteration}_n{len(X_unit)}"
            
        # 创建DataFrame
        data = {}
        
        # 单位空间坐标
        X_unit_np = X_unit.cpu().numpy()
        for i in range(X_unit_np.shape[1]):
            data[f"x{i}_unit"] = X_unit_np[:, i]
            
        # 缩放空间坐标
        X_scaled_np = X_scaled.cpu().numpy()
        for i in range(X_scaled_np.shape[1]):
            data[f"x{i}_scaled"] = X_scaled_np[:, i]
            
        df = pd.DataFrame(data)
        
        # 保存为CSV和Excel
        csv_path = self.experiment_dir / f"{filename}.csv"
        excel_path = self.experiment_dir / f"{filename}.xlsx"
        
        df.to_csv(csv_path, index=False)
        df.to_excel(excel_path, index=False)
        
        print(f"Exported {len(X_unit)} experiment designs to:")
        print(f"  - {csv_path}")
        print(f"  - {excel_path}")
        
        # 临时保存这批点，以便后续update
        self._pending_X_unit = X_unit
        
        return csv_path, excel_path
        
    def import_results(self, results_file: Union[str, Path], X_unit: Optional[torch.Tensor] = None):
        """
        从文件导入实验结果
        
        Args:
            results_file: 结果文件路径（CSV或Excel）
            X_unit: 对应的单位空间点（如果为None，使用最近export的点）
        """
        results_file = Path(results_file)
        
        # 读取结果
        if results_file.suffix == '.csv':
            df = pd.read_csv(results_file)
        elif results_file.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(results_file)
        else:
            raise ValueError(f"Unsupported file format: {results_file.suffix}")
            
        # 提取Y值
        if 'y' in df.columns:
            Y = torch.tensor(df['y'].values, dtype=torch.float32).unsqueeze(-1)
        elif 'Y' in df.columns:
            Y = torch.tensor(df['Y'].values, dtype=torch.float32).unsqueeze(-1)
        else:
            # 假设第一列是Y值
            Y = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32).unsqueeze(-1)
            
        # 获取对应的X
        if X_unit is None:
            if hasattr(self, '_pending_X_unit'):
                X_unit = self._pending_X_unit
            else:
                raise ValueError("No pending X_unit found. Please provide X_unit.")
                
        # 更新模型
        self.update_with_results(X_unit, Y)
        
        print(f"Imported {len(Y)} results from {results_file.name}")
        
    def initialize(
        self, 
        n_init: int = 5, 
        method: str = "lhs",
        X_init: Optional[torch.Tensor] = None,
        Y_init: Optional[torch.Tensor] = None
    ):
        """初始化数据"""
        if X_init is not None and Y_init is not None:
            # 使用提供的数据
            self.tell(X_init, Y_init)
        else:
            # 生成初始样本
            X_init, Y_init = generate_initial_samples(
                self.problem, n_init, method=method, device=self.device
            )
            self.tell(X_init, Y_init)
            
        print(f"初始化完成: {len(self.X_all)} 个点, 最佳值: {self.best_Y:.4f}")
        
        # 保存初始checkpoint
        self.save_checkpoint("init")
        
    def ask(self, n: Optional[int] = None) -> torch.Tensor:
        """
        获取下一批推荐点
        
        Args:
            n: 推荐点数量（默认为 self.q）
            
        Returns:
            推荐的点（在 [0,1] 空间内）
        """
        if self.model is None:
            raise RuntimeError("模型未初始化，请先调用 initialize()")
            
        n = n or self.q
        
        # 生成候选点
        if self.mode == "subspace":
            # 计算梯度子空间
            V_r, H, grad_info = compute_gradient_subspace(
                self.model,
                self.X_all,
                self.r,
                self.n_grad_samples,
                self.device,
                sampling_method=self.subspace_sampling
            )
            
            # 保存子空间信息
            self.V_r_history.append(V_r.cpu())
            self.H_history.append(H.cpu())
            
            # 更新历史
            if self.history["subspace_dims"] is not None:
                self.history["subspace_dims"].append(V_r.shape[1])
            if self.history["gradient_norms"] is not None:
                self.history["gradient_norms"].append(grad_info["mean_grad_norm"])
                
            # 在子空间中生成候选点
            x_ref = self.X_all.mean(dim=0)
            candidates = generate_candidates(
                self.problem.dim,
                self.n_candidates,
                self.device,
                strategy="subspace",
                V_r=V_r,
                x_ref=x_ref
            )
        else:
            # 全空间候选点
            candidates = generate_candidates(
                self.problem.dim,
                self.n_candidates,
                self.device,
                strategy="fullspace"
            )
            
        # 选择批次
        batch = select_batch(
            self.model,
            candidates,
            n,
            self.best_Y,
            self.acquisition_type,
            self.parallel_strategy,
            self.ucb_beta,
            mc_samples=512
        )
        
        return batch
        
    def tell(self, X: torch.Tensor, Y: torch.Tensor):
        """
        更新模型with评估结果
        
        Args:
            X: 评估的点 (n, d)
            Y: 观测值 (n, 1) 或 (n,)
        """
        # 确保正确的形状和设备
        X = X.to(self.device, dtype=torch.float32)
        Y = Y.to(self.device, dtype=torch.float32)
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
            
        # 更新数据
        self.X_all = torch.cat([self.X_all, X])
        self.Y_all = torch.cat([self.Y_all, Y])
        
        # 更新最佳值
        if Y.max().item() > self.best_Y:
            self.best_Y = Y.max().item()
            best_idx = self.Y_all.argmax()
            self.best_X = self.X_all[best_idx].clone()
            
        # 更新历史
        for _ in range(len(X)):
            self.history["best_values"].append(self.Y_all.max().item())
            
        # 更新或创建模型
        if self.model is None:
            self.model = TabPFNSurrogate(
                train_X=self.X_all,
                train_Y=self.Y_all,
                **self.model_config
            )
        else:
            self.model.model.clear_cache()
            self.model.condition_on_observations(X, Y)
            
        # 更新迭代计数
        self.iteration += 1
        
    def get_next_experiments(self, n: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        便捷方法：获取下一批实验点（已缩放到问题空间）
        
        Returns:
            X_unit: 单位空间 [0,1] 中的点
            X_scaled: 缩放到问题空间的点
        """
        X_unit = self.ask(n)
        X_scaled = self.problem.scale(X_unit)
        return X_unit, X_scaled
        
    def update_with_results(self, X_unit: torch.Tensor, Y: torch.Tensor):
        """
        便捷方法：用实验结果更新
        
        Args:
            X_unit: 单位空间中的点（ask() 返回的）
            Y: 实验观测值
        """
        self.tell(X_unit, Y)
        
    def get_results(self) -> Dict:
        """获取当前结果（与 run_pfnbo 兼容的格式）"""
        results = {
            "model_type": f"{self.mode.upper()}-PFNBO-ActiveLearning",
            "mode": self.mode,
            "r": self.r if self.mode == "subspace" else None,
            "best_X": self.best_X,
            "best_X_scaled": self.problem.scale(self.best_X.unsqueeze(0)) if self.best_X is not None else None,
            "best_Y": self.best_Y,
            "all_X": self.X_all,
            "all_Y": self.Y_all,
            "best_values": self.history["best_values"],
            "n_evaluations": len(self.Y_all),
            "n_iterations": self.iteration,
            "batch_size": self.q,
            "parallel_strategy": self.parallel_strategy if self.q > 1 else None,
            
            # Subspace 相关
            "V_r_history": self.V_r_history if self.mode == "subspace" else None,
            "H_history": self.H_history if self.mode == "subspace" else None,
            "subspace_dims": self.history["subspace_dims"],
            "gradient_norms": self.history["gradient_norms"],
        }
        
        return results
    
    def plot_convergence(self, save_path: Optional[Union[str, Path]] = None):
        """绘制收敛曲线"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制最佳值曲线
        ax.plot(self.history["best_values"], 'b-', linewidth=2, label='Best Value')
        ax.set_xlabel('Evaluation')
        ax.set_ylabel('Objective Value')
        ax.set_title(f'{self.experiment_name} - Convergence')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save_path is None:
            save_path = self.experiment_dir / "convergence.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    @property
    def n_evaluated(self) -> int:
        """已评估的点数"""
        return len(self.X_all)
    
    @property
    def current_best(self) -> Tuple[torch.Tensor, float]:
        """当前最佳点和值"""
        return self.best_X, self.best_Y


# 完整的使用示例和运行建议
if __name__ == "__main__":
    from problem import Ackley, Rosenbrock
    from utils import set_seed
    
    # ========== 完整的实验流程示例 ==========
    
    print("=== 完整的实验室优化流程示例 ===\n")
    
    # 1. 第一天：初始化实验
    print("Day 1: 初始化实验")
    print("-" * 50)
    
    # 设置问题和学习器
    problem = Ackley(dim=50, noise_std=0.1)
    
    learner = PFNBOActiveLearner(
        problem=problem,
        mode="subspace",
        q=5,  # 每批5个实验
        acquisition_type="ei",
        r=15,
        seed=42,  # 固定随机种子
        experiment_name="ackley_lab_experiment",
        reg_model_path="weights/tabpfn-v2-regressor.ckpt",
        device='mps'
    )
    
    # 初始化（LHS采样）
    learner.initialize(n_init=10, method="lhs")
    
    # 导出第一批实验设计
    csv_path, excel_path = learner.export_next_experiments(n=5)
    print(f"\n请进行实验，设计点已保存至: {excel_path}")
    
    # 保存checkpoint
    learner.save_checkpoint("day1")
    
    # 2. 第七天：第一批实验完成
    print("\n\nDay 7: 第一批实验完成")
    print("-" * 50)
    
    # 创建新的learner实例并加载之前的状态
    learner2 = PFNBOActiveLearner(
        problem=problem,
        mode="subspace",
        q=5,
        acquisition_type="ei",
        r=5,
        seed=42,
        experiment_name="ackley_lab_experiment",  # 使用相同的实验名
        reg_model_path="weights/tabpfn-v2-regressor.ckpt",
        device='mps'
    )
    
    # 加载checkpoint
    learner2.load_checkpoint()
    
    # 模拟实验结果（实际使用时从Excel/CSV读取）
    # 这里我们模拟已经完成了之前导出的5个实验
    X_unit = learner._pending_X_unit  # 之前导出的点
    X_scaled = problem.scale(X_unit)
    Y_results, _ = problem.evaluate(X_scaled)
    
    # 更新结果
    learner2.update_with_results(X_unit, Y_results)
    print(f"更新了 {len(Y_results)} 个实验结果")
    print(f"当前最佳值: {learner2.best_Y:.4f}")
    
    # 导出下一批实验
    csv_path, excel_path = learner2.export_next_experiments(n=5)
    
    # 保存新的checkpoint
    learner2.save_checkpoint("day7")
    
    # 绘制收敛曲线
    learner2.plot_convergence()
    
    # 3. 第十四天：第二批实验完成
    print("\n\nDay 14: 第二批实验完成")
    print("-" * 50)
    
    # 再次加载
    learner3 = PFNBOActiveLearner(
        problem=problem,
        mode="subspace",
        q=5,
        acquisition_type="ei",
        r=5,
        seed=42,
        experiment_name="ackley_lab_experiment",
        reg_model_path="weights/tabpfn-v2-regressor.ckpt",
        device='mps'
    )
    
    learner3.load_checkpoint()
    
    # 模拟从文件导入结果
    # 首先创建一个模拟的结果文件
    results_df = pd.DataFrame({
        'y': np.random.randn(5) * 0.1 - 5.0  # 模拟的实验结果
    })
    results_path = learner3.experiment_dir / "results_batch2.csv"
    results_df.to_csv(results_path, index=False)
    
    # 导入结果
    learner3.import_results(results_path)
    
    print(f"总评估次数: {learner3.n_evaluated}")
    print(f"当前最佳值: {learner3.best_Y:.4f}")
    
    # 获取最终结果
    final_results = learner3.get_results()
    
    # 保存最终checkpoint
    learner3.save_checkpoint("final")
    
    # 绘制最终收敛曲线
    learner3.plot_convergence()
    
    print("\n实验完成！")
    print(f"所有数据已保存在: {learner3.experiment_dir}")
    
    # ========== 使用建议 ==========
    print("\n\n=== 完整运行建议 ===")
    print("""
    1. 初始化阶段：
       - 设置合适的随机种子确保可重现性
       - 选择合适的初始采样数（建议 2*dim 到 5*dim）
       - 使用 LHS 或 Sobol 进行初始采样
    
    2. 实验循环：
       - 每次导出合理数量的实验点（考虑实验能力）
       - 保存checkpoint时使用有意义的标签（如日期）
       - 定期备份整个实验目录
    
    3. 数据管理：
       - 所有数据自动保存为CSV格式
       - Excel文件方便实验人员使用
       - JSON配置文件记录所有参数
    
    4. 恢复机制：
       - 使用load_checkpoint()恢复状态
       - 模型会自动重新训练
       - 可以从任意checkpoint继续
    
    5. 结果分析：
       - 使用plot_convergence()查看优化进展
       - all_data.csv包含所有历史数据
       - all_data_scaled.csv包含原始空间的数据
    
    6. 注意事项：
       - 保持problem定义的一致性
       - 不要修改已保存的数据文件
       - 定期检查收敛情况，适时停止
    """)