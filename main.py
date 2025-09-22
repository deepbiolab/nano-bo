import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
from ask_tell import PFNBOActiveLearner
from meproblem import MediumOptimizationProblem


class MediumOptimizationActiveLearner(PFNBOActiveLearner):
    """
    专门用于培养基优化的 Active Learning 接口
    """
    
    def __init__(
        self,
        excel_path: str,
        target_column: str = "Y",
        maximize: bool = True,
        use_log_scale: Optional[List[str]] = None,
        **pfnbo_kwargs
    ):
        # 创建问题实例
        self.problem = MediumOptimizationProblem(
            excel_path=excel_path,
            target_column=target_column,
            maximize=maximize,
            use_log_scale=use_log_scale
        )
        
        # 初始化父类
        super().__init__(problem=self.problem, **pfnbo_kwargs)
        
        # 跟踪轮次
        self.current_round = 0
        if self.problem.historical_rounds is not None and len(self.problem.historical_rounds) > 0:
            self.current_round = self.problem.historical_rounds.max().item() + 1
            
    def initialize_from_history(self):
        """使用历史数据初始化"""
        if self.problem.historical_X is not None and len(self.problem.historical_X) > 0:
            self.tell(self.problem.historical_X, self.problem.historical_Y)
            print(f"从历史数据初始化: {len(self.problem.historical_X)} 个数据点")
            print(f"当前最佳值: {self.best_Y:.4f}")
        else:
            print("警告: 没有历史数据可用于初始化")
            
    def suggest_experiments(self, n: Optional[int] = None) -> Dict[str, List[float]]:
        """
        建议下一批实验条件
        
        Returns:
            包含每个变量建议值的字典
        """
        # 获取推荐点
        X_unit = self.ask(n)
        X_real = self.problem.scale(X_unit)
        
        # 转换为易读的字典格式
        suggestions = []
        for i in range(len(X_unit)):
            exp_dict = {}
            for j, var_name in enumerate(self.problem.variable_names):
                exp_dict[var_name] = X_real[i, j].item()
                
            # 添加常量
            for const_name in self.problem.constant_names:
                # 从历史数据获取常量值
                if len(self.problem.data_df) > 0:
                    const_val = self.problem.data_df[const_name].iloc[0]
                    exp_dict[const_name] = const_val
                    
            suggestions.append(exp_dict)
            
        # 保存推荐的点用于后续更新
        self._pending_X_unit = X_unit
        
        return suggestions
    
    def update_with_experiments(self, results: List[Dict[str, float]]):
        """
        使用实验结果更新模型
        
        Args:
            results: 实验结果列表，每个字典包含目标值
        """
        # 提取Y值
        Y_values = []
        for result in results:
            if self.problem.target_column not in result:
                raise ValueError(f"结果中缺少目标列 '{self.problem.target_column}'")
            y_val = result[self.problem.target_column]
            if not self.problem.maximize:
                y_val = -y_val  # 转换为最大化问题
            Y_values.append(y_val)
            
        Y = torch.tensor(Y_values, dtype=torch.float32).unsqueeze(-1)
        
        # 更新模型
        self.tell(self._pending_X_unit, Y)
        
        # 更新问题中的历史数据
        self.problem.add_new_observations(self._pending_X_unit, Y, self.current_round)
        
        print(f"更新完成: 添加了 {len(results)} 个新数据点")
        print(f"当前最佳值: {self.best_Y:.4f}")
        
        # 增加轮次
        self.current_round += 1
        
    def get_optimization_summary(self) -> Dict:
        """获取优化摘要"""
        summary = {
            "total_experiments": len(self.X_all),
            "rounds_completed": self.current_round,
            "best_value": self.best_Y if self.problem.maximize else -self.best_Y,
            "best_conditions": {},
            "improvement_over_initial": None
        }
        
        # 最佳条件
        if self.best_X is not None:
            best_X_real = self.problem.scale(self.best_X.unsqueeze(0)).squeeze()
            for i, var_name in enumerate(self.problem.variable_names):
                summary["best_conditions"][var_name] = best_X_real[i].item()
                
        # 计算相对初始的改进
        if self.problem.historical_rounds is not None:
            initial_Y = self.problem.historical_Y[self.problem.historical_rounds == 0]
            if len(initial_Y) > 0:
                initial_best = initial_Y.max().item()
                improvement = (self.best_Y - initial_best) / abs(initial_best) * 100
                summary["improvement_over_initial"] = improvement
                
        return summary
    
    def save_all_results(self, output_path: Optional[str] = None):
        """保存所有结果到Excel文件"""
        self.problem.save_results(output_path)


# 实际使用示例
if __name__ == "__main__":
    from utils import set_seed
    set_seed(42)
    
    # 创建优化器
    optimizer = MediumOptimizationActiveLearner(
        excel_path="medium_optimization_example.xlsx",
        target_column="Y",
        maximize=True,
        use_log_scale=['glucose', 'yeast_extract', 'peptone'],
        # PFNBO 参数
        mode="subspace",
        q=3,  # 每次建议3个实验
        r=3,  # 子空间维度
        acquisition_type="ei",
        reg_model_path="weights/tabpfn-v2-regressor.ckpt"
    )
    
    # 从历史数据初始化
    optimizer.initialize_from_history()
    
    # 模拟优化循环
    for round_i in range(3):
        print(f"\n=== 第 {round_i + 1} 轮优化 ===")
        
        # 获取实验建议
        suggestions = optimizer.suggest_experiments(n=3)
        
        print("建议的实验条件:")
        for i, exp in enumerate(suggestions):
            print(f"\n实验 {i+1}:")
            for var, val in exp.items():
                if var in optimizer.problem.variable_names:
                    print(f"  {var}: {val:.2f}")
                    
        # 模拟实验结果（实际使用时替换为真实实验）
        results = []
        for i, exp in enumerate(suggestions):
            # 这里应该是真实的实验结果
            # 模拟：基于条件计算产量
            glucose = exp['glucose']
            yeast = exp['yeast_extract']
            peptone = exp['peptone']
            
            # 假设的产量模型
            yield_value = (
                0.1 * glucose + 
                0.15 * yeast + 
                0.1 * peptone + 
                np.random.normal(0, 0.1)
            )
            yield_value = max(0.1, min(5.0, yield_value))  # 限制范围
            
            results.append({
                optimizer.problem.target_column: yield_value
            })
            print(f"\n实验 {i+1} 结果: {yield_value:.3f}")
            
        # 更新模型
        optimizer.update_with_experiments(results)
        
        # 显示优化进展
        summary = optimizer.get_optimization_summary()
        print(f"\n优化摘要:")
        print(f"  总实验数: {summary['total_experiments']}")
        print(f"  最佳值: {summary['best_value']:.3f}")
        if summary['improvement_over_initial'] is not None:
            print(f"  相对初始改进: {summary['improvement_over_initial']:.1f}%")
            
    # 保存所有结果
    optimizer.save_all_results("medium_optimization_results.xlsx")
    
    # 显示最终最佳条件
    print("\n=== 最终最佳条件 ===")
    summary = optimizer.get_optimization_summary()
    for var, val in summary['best_conditions'].items():
        print(f"{var}: {val:.2f}")