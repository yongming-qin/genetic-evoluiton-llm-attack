#!/usr/bin/env python3
"""配置加载器 - 用于加载和管理并行评估系统的配置"""

import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class GeneticAlgorithmConfig:
    """遗传算法配置"""
    generations: int = 10
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 2

@dataclass
class ModelConfig:
    """模型配置"""
    attack_model: str = "deepseek_r1"
    target_model: str = "gpt-oss-20b"
    evaluation_models: List[str] = None
    
    def __post_init__(self):
        if self.evaluation_models is None:
            self.evaluation_models = ["deepseek_judge", "gpt4o_judge", "qwen_judge"]

@dataclass
class ParallelEvaluationConfig:
    """并行评估配置"""
    enabled: bool = True
    max_workers: int = 3
    timeout_seconds: int = 30
    retry_attempts: int = 2
    consensus_threshold: float = 0.6
    agreement_threshold: float = 0.7

@dataclass
class FitnessCalculationConfig:
    """适应度计算配置"""
    consensus_weight: float = 0.4
    agreement_weight: float = 0.3
    diversity_weight: float = 0.2
    success_rate_weight: float = 0.1

@dataclass
class OutputConfig:
    """输出配置"""
    save_detailed_results: bool = True
    generate_comparison_report: bool = True
    save_intermediate_generations: bool = False
    output_directory: str = "results"
    filename_prefix: str = "parallel_genetic_attack"

@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    save_to_file: bool = True
    log_file: str = "parallel_evaluation.log"

@dataclass
class PerformanceConfig:
    """性能配置"""
    batch_size: int = 5
    memory_limit_mb: int = 2048
    enable_caching: bool = True
    cache_size: int = 100

@dataclass
class SafetyConfig:
    """安全配置"""
    content_filter_enabled: bool = True
    max_prompt_length: int = 2000
    blocked_keywords: List[str] = None
    
    def __post_init__(self):
        if self.blocked_keywords is None:
            self.blocked_keywords = [
                "explicit_violence",
                "illegal_activities", 
                "personal_information"
            ]

@dataclass
class ExperimentalConfig:
    """实验性功能配置"""
    adaptive_fitness: bool = False
    dynamic_population: bool = False
    multi_objective_optimization: bool = False

@dataclass
class SystemConfig:
    """系统总配置"""
    description: str = "并行评估系统配置"
    version: str = "1.0"
    genetic_algorithm: GeneticAlgorithmConfig = None
    models: ModelConfig = None
    parallel_evaluation: ParallelEvaluationConfig = None
    fitness_calculation: FitnessCalculationConfig = None
    output: OutputConfig = None
    logging: LoggingConfig = None
    performance: PerformanceConfig = None
    safety: SafetyConfig = None
    experimental: ExperimentalConfig = None
    
    def __post_init__(self):
        if self.genetic_algorithm is None:
            self.genetic_algorithm = GeneticAlgorithmConfig()
        if self.models is None:
            self.models = ModelConfig()
        if self.parallel_evaluation is None:
            self.parallel_evaluation = ParallelEvaluationConfig()
        if self.fitness_calculation is None:
            self.fitness_calculation = FitnessCalculationConfig()
        if self.output is None:
            self.output = OutputConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.safety is None:
            self.safety = SafetyConfig()
        if self.experimental is None:
            self.experimental = ExperimentalConfig()

class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_file: str = "parallel_config.json"):
        self.config_file = config_file
        self.config: Optional[SystemConfig] = None
    
    def load_config(self) -> SystemConfig:
        """加载配置文件"""
        if not os.path.exists(self.config_file):
            print(f"配置文件 {self.config_file} 不存在，使用默认配置")
            self.config = SystemConfig()
            self.save_config()  # 保存默认配置
            return self.config
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.config = self._parse_config(config_data)
            print(f"成功加载配置文件: {self.config_file}")
            return self.config
            
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            print("使用默认配置")
            self.config = SystemConfig()
            return self.config
    
    def _parse_config(self, config_data: Dict[str, Any]) -> SystemConfig:
        """解析配置数据"""
        # 解析各个配置部分
        genetic_config = GeneticAlgorithmConfig(
            **config_data.get('genetic_algorithm', {})
        )
        
        model_config = ModelConfig(
            **config_data.get('models', {})
        )
        
        parallel_config = ParallelEvaluationConfig(
            **config_data.get('parallel_evaluation', {})
        )
        
        fitness_config = FitnessCalculationConfig(
            **config_data.get('fitness_calculation', {})
        )
        
        output_config = OutputConfig(
            **config_data.get('output', {})
        )
        
        logging_config = LoggingConfig(
            **config_data.get('logging', {})
        )
        
        performance_config = PerformanceConfig(
            **config_data.get('performance', {})
        )
        
        safety_config = SafetyConfig(
            **config_data.get('safety', {})
        )
        
        experimental_config = ExperimentalConfig(
            **config_data.get('experimental', {})
        )
        
        return SystemConfig(
            description=config_data.get('description', '并行评估系统配置'),
            version=config_data.get('version', '1.0'),
            genetic_algorithm=genetic_config,
            models=model_config,
            parallel_evaluation=parallel_config,
            fitness_calculation=fitness_config,
            output=output_config,
            logging=logging_config,
            performance=performance_config,
            safety=safety_config,
            experimental=experimental_config
        )
    
    def save_config(self, config: Optional[SystemConfig] = None) -> bool:
        """保存配置到文件"""
        if config is None:
            config = self.config
        
        if config is None:
            print("没有配置可保存")
            return False
        
        try:
            config_dict = asdict(config)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            print(f"配置已保存到: {self.config_file}")
            return True
            
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    def get_config(self) -> SystemConfig:
        """获取当前配置"""
        if self.config is None:
            return self.load_config()
        return self.config
    
    def update_config(self, **kwargs) -> bool:
        """更新配置"""
        if self.config is None:
            self.load_config()
        
        try:
            # 更新配置字段
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    print(f"警告: 未知配置字段 {key}")
            
            return True
            
        except Exception as e:
            print(f"更新配置失败: {e}")
            return False
    
    def validate_config(self, config: Optional[SystemConfig] = None) -> List[str]:
        """验证配置"""
        if config is None:
            config = self.config
        
        if config is None:
            return ["配置未加载"]
        
        errors = []
        
        # 验证遗传算法配置
        ga_config = config.genetic_algorithm
        if ga_config.generations <= 0:
            errors.append("遗传算法代数必须大于0")
        if ga_config.population_size <= 0:
            errors.append("种群大小必须大于0")
        if not 0 <= ga_config.mutation_rate <= 1:
            errors.append("变异率必须在0-1之间")
        if not 0 <= ga_config.crossover_rate <= 1:
            errors.append("交叉率必须在0-1之间")
        
        # 验证并行评估配置
        pe_config = config.parallel_evaluation
        if pe_config.max_workers <= 0:
            errors.append("最大工作线程数必须大于0")
        if pe_config.timeout_seconds <= 0:
            errors.append("超时时间必须大于0")
        if not 0 <= pe_config.consensus_threshold <= 1:
            errors.append("共识阈值必须在0-1之间")
        if not 0 <= pe_config.agreement_threshold <= 1:
            errors.append("一致性阈值必须在0-1之间")
        
        # 验证适应度计算配置
        fc_config = config.fitness_calculation
        total_weight = (fc_config.consensus_weight + 
                       fc_config.agreement_weight + 
                       fc_config.diversity_weight + 
                       fc_config.success_rate_weight)
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"适应度权重总和应为1.0，当前为{total_weight:.3f}")
        
        # 验证模型配置
        model_config = config.models
        if not model_config.evaluation_models:
            errors.append("至少需要一个评估模型")
        
        return errors
    
    def print_config_summary(self, config: Optional[SystemConfig] = None):
        """打印配置摘要"""
        if config is None:
            config = self.config
        
        if config is None:
            print("配置未加载")
            return
        
        print("配置摘要:")
        print(f"  版本: {config.version}")
        print(f"  描述: {config.description}")
        print(f"\n遗传算法:")
        print(f"  代数: {config.genetic_algorithm.generations}")
        print(f"  种群大小: {config.genetic_algorithm.population_size}")
        print(f"  变异率: {config.genetic_algorithm.mutation_rate}")
        print(f"  交叉率: {config.genetic_algorithm.crossover_rate}")
        print(f"\n模型配置:")
        print(f"  攻击模型: {config.models.attack_model}")
        print(f"  目标模型: {config.models.target_model}")
        print(f"  评估模型: {', '.join(config.models.evaluation_models)}")
        print(f"\n并行评估:")
        print(f"  启用: {config.parallel_evaluation.enabled}")
        print(f"  最大工作线程: {config.parallel_evaluation.max_workers}")
        print(f"  超时时间: {config.parallel_evaluation.timeout_seconds}s")
        print(f"  共识阈值: {config.parallel_evaluation.consensus_threshold}")
        print(f"\n输出配置:")
        print(f"  输出目录: {config.output.output_directory}")
        print(f"  文件前缀: {config.output.filename_prefix}")
        print(f"  详细结果: {config.output.save_detailed_results}")

def create_default_config_file(filename: str = "parallel_config.json"):
    """创建默认配置文件"""
    loader = ConfigLoader(filename)
    config = SystemConfig()
    loader.config = config
    return loader.save_config()

def load_config_from_file(filename: str = "parallel_config.json") -> SystemConfig:
    """从文件加载配置"""
    loader = ConfigLoader(filename)
    return loader.load_config()

if __name__ == "__main__":
    # 测试配置加载器
    print("测试配置加载器")
    
    # 创建配置加载器
    loader = ConfigLoader()
    
    # 加载配置
    config = loader.load_config()
    
    # 打印配置摘要
    loader.print_config_summary()
    
    # 验证配置
    errors = loader.validate_config()
    if errors:
        print(f"\n配置验证错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"\n✓ 配置验证通过")