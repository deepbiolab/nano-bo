# medium_optimization_problem.py

import torch
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from problem import BenchmarkProblem, DataType


class MediumOptimizationProblem(BenchmarkProblem):
    """
    培养基优化问题
    
    从Excel文件读取实验设计空间和历史数据
    支持混合变量类型（连续变量和常量）
    """
    name = 'Ackley'
    available_dimensions = (1, None)
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0
    optimal_value = 0.0
    def __init__(
        self,
        excel_path: str,
        target_column: str = "Y",
        maximize: bool = True,
        noise_std: float = 0.0,
        use_log_scale: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Args:
            excel_path: Excel文件路径
            target_column: 目标列名称
            maximize: 是否最大化目标（False表示最小化）
            noise_std: 添加的噪声标准差
            use_log_scale: 需要使用对数尺度的变量名列表
        """
        self.excel_path = excel_path
        self.target_column = target_column
        self.maximize = maximize
        self.noise_std = noise_std
        self.use_log_scale = use_log_scale or []
        
        # 读取数据
        self._load_data()
        
        # 设置维度和边界
        dim = len(self.variable_names)
        bounds = [(row['lowerbound'], row['upperbound']) for _, row in self.range_df.iterrows()]
        
        # 初始化父类
        super().__init__(
            dim=dim,
            bounds=bounds,
            num_objectives=1,
            num_constraints=0,
            tags=["Medium Optimization", "Real-world", "Mixed Variables"]
        )
        
        # 存储历史数据用于参考
        self.historical_X = None
        self.historical_Y = None
        self.historical_rounds = None
        self._process_historical_data()
        
    def _load_data(self):
        """从Excel文件加载数据"""
        # 读取range sheet
        self.range_df = pd.read_excel(self.excel_path, sheet_name='range')
        
        # 验证必需的列
        required_cols = ['name', 'lowerbound', 'upperbound', 'type']
        if not all(col in self.range_df.columns for col in required_cols):
            raise ValueError(f"Range sheet must contain columns: {required_cols}")
            
        # 分离变量和常量
        self.variables_df = self.range_df[self.range_df['type'] == 'real'].copy()
        self.constants_df = self.range_df[self.range_df['type'] == 'constant'].copy()
        
        self.variable_names = self.variables_df['name'].tolist()
        self.constant_names = self.constants_df['name'].tolist()
        self.all_names = self.range_df['name'].tolist()
        
        # 读取data sheet
        self.data_df = pd.read_excel(self.excel_path, sheet_name='data')
        
        # 验证数据列
        if self.target_column not in self.data_df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data sheet")
            
        # 提取元信息列
        self.meta_columns = [col for col in self.data_df.columns 
                            if col not in self.all_names + [self.target_column, 'round']]
        
    def _process_historical_data(self):
        """处理历史数据，转换为torch tensors"""
        if len(self.data_df) == 0:
            return
            
        # 提取变量值（只包含real类型）
        X_list = []
        Y_list = []
        rounds_list = []
        
        for _, row in self.data_df.iterrows():
            # 提取变量值
            x_values = []
            for var_name in self.variable_names:
                if var_name in row:
                    x_values.append(row[var_name])
                else:
                    raise ValueError(f"Variable '{var_name}' not found in data")
                    
            X_list.append(x_values)
            Y_list.append(row[self.target_column])
            
            if 'round' in row:
                rounds_list.append(row['round'])
            else:
                rounds_list.append(0)
                
        # 转换为tensors
        self.historical_X_real = torch.tensor(X_list, dtype=torch.float32)
        self.historical_Y = torch.tensor(Y_list, dtype=torch.float32).unsqueeze(-1)
        self.historical_rounds = torch.tensor(rounds_list, dtype=torch.int32)
        
        # 如果需要最小化，取负值
        if not self.maximize:
            self.historical_Y = -self.historical_Y
            
        # 将真实空间的X转换到[0,1]空间
        self.historical_X = self._inverse_scale(self.historical_X_real)
        
        # 计算当前最优
        best_idx = self.historical_Y.argmax()
        self.optimum = self.historical_Y[best_idx].item()
        self.x_opt = self.historical_X[best_idx].unsqueeze(0)
        
    def _inverse_scale(self, X_real: torch.Tensor) -> torch.Tensor:
        """将真实空间的值转换到[0,1]空间"""
        X_unit = torch.zeros_like(X_real)
        
        for i, (_, row) in enumerate(self.variables_df.iterrows()):
            var_name = row['name']
            lb = row['lowerbound']
            ub = row['upperbound']
            
            if var_name in self.use_log_scale:
                # 对数尺度
                X_unit[:, i] = (torch.log(X_real[:, i]) - np.log(lb)) / (np.log(ub) - np.log(lb))
            else:
                # 线性尺度
                X_unit[:, i] = (X_real[:, i] - lb) / (ub - lb)
                
        return torch.clamp(X_unit, 0, 1)
    
    def scale(self, X: torch.Tensor) -> torch.Tensor:
        """将[0,1]空间的值转换到真实空间"""
        if X.dim() == 1:
            X = X.unsqueeze(0)
            
        X_real = torch.zeros_like(X)
        
        for i, (_, row) in enumerate(self.variables_df.iterrows()):
            var_name = row['name']
            lb = row['lowerbound']
            ub = row['upperbound']
            
            if var_name in self.use_log_scale:
                # 对数尺度
                log_val = X[:, i] * (np.log(ub) - np.log(lb)) + np.log(lb)
                X_real[:, i] = torch.exp(log_val)
            else:
                # 线性尺度
                X_real[:, i] = X[:, i] * (ub - lb) + lb
                
        return X_real
    
    def _evaluate_implementation(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估函数 - 在实际使用中应该被覆盖
        这里提供一个基于历史数据的代理评估
        """
        # 这是一个占位符 - 实际使用时应该调用真实的实验
        # 这里我们使用最近邻插值作为示例
        if self.historical_X is None or len(self.historical_X) == 0:
            raise RuntimeError("No historical data available for evaluation")
            
        # 计算到历史点的距离
        distances = torch.cdist(X, self.historical_X)
        nearest_idx = distances.argmin(dim=1)
        
        # 返回最近点的值（加噪声）
        Y = self.historical_Y[nearest_idx]
        if self.noise_std > 0:
            Y = Y + torch.randn_like(Y) * self.noise_std
            
        return torch.zeros(X.shape[0], 0), Y  # 无约束
    
    def get_context_data(self) -> Dict[str, torch.Tensor]:
        """
        获取用于TabPFN的上下文数据
        包括常量信息
        """
        context = {
            'historical_X': self.historical_X,
            'historical_Y': self.historical_Y,
            'historical_rounds': self.historical_rounds,
        }
        
        # 添加常量信息
        if len(self.constants_df) > 0:
            const_values = []
            for _, row in self.data_df.iterrows():
                const_row = []
                for const_name in self.constant_names:
                    if const_name in row:
                        const_row.append(row[const_name])
                const_values.append(const_row)
                
            if const_values:
                context['constants'] = torch.tensor(const_values, dtype=torch.float32)
                context['constant_names'] = self.constant_names
                
        return context
    
    def add_new_observations(self, X_new: torch.Tensor, Y_new: torch.Tensor, round_num: int):
        """添加新的实验观察结果"""
        # 更新历史数据
        if self.historical_X is None:
            self.historical_X = X_new
            self.historical_Y = Y_new
            self.historical_rounds = torch.full((len(X_new),), round_num, dtype=torch.int32)
        else:
            self.historical_X = torch.cat([self.historical_X, X_new])
            self.historical_Y = torch.cat([self.historical_Y, Y_new])
            new_rounds = torch.full((len(X_new),), round_num, dtype=torch.int32)
            self.historical_rounds = torch.cat([self.historical_rounds, new_rounds])
            
        # 更新最优值
        best_idx = self.historical_Y.argmax()
        self.optimum = self.historical_Y[best_idx].item()
        self.x_opt = self.historical_X[best_idx].unsqueeze(0)
        
    def save_results(self, output_path: Optional[str] = None):
        """保存结果到新的Excel文件"""
        if output_path is None:
            output_path = self.excel_path.replace('.xlsx', '_results.xlsx')
            
        # 准备数据
        results_data = []
        
        for i in range(len(self.historical_X)):
            row = {}
            
            # 添加变量值（真实空间）
            X_real = self.scale(self.historical_X[i].unsqueeze(0)).squeeze()
            for j, var_name in enumerate(self.variable_names):
                row[var_name] = X_real[j].item()
                
            # 添加目标值
            y_val = self.historical_Y[i].item()
            if not self.maximize:
                y_val = -y_val  # 恢复原始值
            row[self.target_column] = y_val
            
            # 添加轮次
            row['round'] = self.historical_rounds[i].item()
            
            results_data.append(row)
            
        # 创建DataFrame并保存
        results_df = pd.DataFrame(results_data)
        
        with pd.ExcelWriter(output_path) as writer:
            # 保存range信息
            self.range_df.to_excel(writer, sheet_name='range', index=False)
            # 保存结果数据
            results_df.to_excel(writer, sheet_name='data', index=False)
            
        print(f"Results saved to {output_path}")
        
    def get_variable_importance(self) -> Dict[str, float]:
        """基于历史数据计算变量重要性（简单相关性）"""
        if self.historical_X is None or len(self.historical_X) < 3:
            return {var: 0.0 for var in self.variable_names}
            
        importance = {}
        Y_numpy = self.historical_Y.numpy().flatten()
        
        for i, var_name in enumerate(self.variable_names):
            X_var = self.historical_X[:, i].numpy()
            # 计算相关系数
            corr = np.corrcoef(X_var, Y_numpy)[0, 1]
            importance[var_name] = abs(corr)
            
        return importance


# 使用示例
if __name__ == "__main__":
    # 创建示例Excel文件
    import pandas as pd
    
    # 创建range数据
    range_data = {
        'name': ['glucose', 'yeast_extract', 'peptone', 'NaCl', 'pH', 'temperature'],
        'lowerbound': [10, 5, 5, 0.5, 6.0, 25],
        'upperbound': [50, 20, 20, 5.0, 8.0, 25],
        'type': ['real', 'real', 'real', 'real', 'real', 'constant']
    }
    
    # 创建历史数据
    data_records = []
    np.random.seed(42)
    
    # 初始设计（round 0）
    for i in range(10):
        record = {
            'glucose': np.random.uniform(10, 50),
            'yeast_extract': np.random.uniform(5, 20),
            'peptone': np.random.uniform(5, 20),
            'NaCl': np.random.uniform(0.5, 5.0),
            'pH': np.random.uniform(6.0, 8.0),
            'temperature': 25,
            'Y': np.random.uniform(0.5, 2.0),  # 产量
            'round': 0,
            'exp_code': f'EXP_{i:03d}'
        }
        data_records.append(record)
    
    # 保存到Excel
    with pd.ExcelWriter('medium_optimization_example.xlsx') as writer:
        pd.DataFrame(range_data).to_excel(writer, sheet_name='range', index=False)
        pd.DataFrame(data_records).to_excel(writer, sheet_name='data', index=False)
    
    # 使用Problem类
    problem = MediumOptimizationProblem(
        excel_path='medium_optimization_example.xlsx',
        target_column='Y',
        maximize=True,
        use_log_scale=['glucose', 'yeast_extract', 'peptone']  # 使用对数尺度
    )
    
    print(f"问题维度: {problem.dim}")
    print(f"变量名: {problem.variable_names}")
    print(f"常量名: {problem.constant_names}")
    print(f"历史数据点数: {len(problem.historical_X)}")
    print(f"当前最优值: {problem.optimum:.4f}")
    
    # 测试缩放
    x_test = torch.rand(3, problem.dim)
    x_scaled = problem.scale(x_test)
    print(f"\n测试缩放:")
    print(f"单位空间: {x_test[0]}")
    print(f"真实空间: {x_scaled[0]}")
    
    # 获取变量重要性
    importance = problem.get_variable_importance()
    print(f"\n变量重要性:")
    for var, imp in importance.items():
        print(f"  {var}: {imp:.3f}")