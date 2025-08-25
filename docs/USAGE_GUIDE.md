# 并行评估系统使用指南

## 概述

本系统实现了一个基于遗传算法的LLM攻击框架，支持使用多个评估模型进行并行评估，以提供更全面和可靠的攻击效果分析。

## 核心特性

### 1. 并行评估
- **多模型评估**: 同时使用 DeepSeek、GPT-4o、Qwen 三种评估模型
- **共识机制**: 基于多个评估器的一致性计算共识分数
- **性能对比**: 提供不同评估器的性能分析和对比

### 2. 配置管理
- **配置文件**: 支持JSON格式的配置文件
- **参数覆盖**: 配置文件可覆盖命令行参数
- **配置验证**: 自动验证配置参数的有效性

### 3. 结果分析
- **详细报告**: 生成包含共识分数、一致性水平的详细报告
- **可视化数据**: 提供进化历史和评估器性能的统计信息
- **对比分析**: 支持传统评估与并行评估的对比

## 快速开始

### 1. 基本使用

```bash
# 使用默认参数运行
python main.py

# 指定参数运行
python main.py --generations 10 --population 20 --parallel-eval

# 使用配置文件
python main.py --config parallel_config.json --use-config
```

### 2. 运行演示

```bash
# 运行完整演示
python demo_parallel_evaluation.py

# 快速测试
python demo_parallel_evaluation.py --quick

# 运行对比测试
python run_parallel_evaluation.py --comparison
```

### 3. 配置文件使用

```bash
# 创建默认配置文件
python config_loader.py

# 使用自定义配置
python main.py --config my_config.json --use-config
```

## 详细使用说明

### 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--generations` | 进化代数 | 10 |
| `--population` | 种群大小 | 20 |
| `--model` | 攻击模型 | deepseek_r1 |
| `--target-model` | 目标模型 | gpt-oss-20b |
| `--parallel-eval` | 启用并行评估 | False |
| `--use-hf-evaluator` | 使用HF评估器 | False |
| `--config` | 配置文件路径 | parallel_config.json |
| `--use-config` | 使用配置文件覆盖参数 | False |
| `--output` | 输出文件 | genetic_attack_results.json |
| `--verbose` | 详细输出 | False |

### 配置文件结构

```json
{
  "genetic_algorithm": {
    "generations": 10,
    "population_size": 20,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8
  },
  "models": {
    "attack_model": "deepseek_r1",
    "target_model": "gpt-oss-20b",
    "evaluation_models": ["deepseek_judge", "gpt4o_judge", "qwen_judge"]
  },
  "parallel_evaluation": {
    "enabled": true,
    "max_workers": 3,
    "timeout_seconds": 30,
    "consensus_threshold": 0.6
  }
}
```

## 使用场景

### 1. 研究场景

**学术研究**: 评估不同LLM的安全性和鲁棒性
```bash
python main.py --generations 50 --population 100 --parallel-eval --target-model gpt-oss-20b
```

**对比分析**: 比较不同评估方法的效果
```bash
python run_parallel_evaluation.py --comparison
```

### 2. 开发场景

**快速测试**: 验证系统功能
```bash
python demo_parallel_evaluation.py --quick
```

**参数调优**: 使用配置文件调整参数
```bash
# 编辑 parallel_config.json
python main.py --use-config
```

### 3. 生产场景

**批量评估**: 对多个模型进行评估
```bash
# 创建不同的配置文件
python main.py --config model1_config.json --use-config
python main.py --config model2_config.json --use-config
```

## 结果解读

### 1. 基本指标

- **适应度分数**: 个体在进化过程中的表现
- **共识分数**: 多个评估器对同一攻击的一致性评分
- **一致性水平**: 评估器之间的意见一致程度

### 2. 并行评估指标

- **成功率**: 评估器成功完成评估的比例
- **平均执行时间**: 评估器的平均响应时间
- **多样性分数**: 评估结果的多样性程度

### 3. 进化指标

- **进化改进**: 最终代与初始代的适应度差异
- **收敛速度**: 达到稳定状态所需的代数
- **种群多样性**: 种群中个体的差异程度

## 性能优化

### 1. 参数调优

```json
{
  "parallel_evaluation": {
    "max_workers": 3,        // 根据CPU核心数调整
    "timeout_seconds": 30,   // 根据网络状况调整
    "batch_size": 5          // 根据内存大小调整
  },
  "performance": {
    "enable_caching": true,   // 启用缓存提高性能
    "cache_size": 100,       // 缓存大小
    "memory_limit_mb": 2048  // 内存限制
  }
}
```

### 2. 资源管理

- **内存优化**: 设置合适的批处理大小
- **网络优化**: 调整超时时间和重试次数
- **并发控制**: 根据系统资源调整工作线程数

## 故障排除

### 1. 常见问题

**配置文件错误**:
```bash
# 验证配置文件
python config_loader.py
```

**模型初始化失败**:
- 检查模型配置
- 验证API密钥
- 确认网络连接

**并行评估超时**:
- 增加超时时间
- 减少并发数量
- 检查网络状况

### 2. 调试模式

```bash
# 启用详细输出
python main.py --verbose

# 使用小参数测试
python main.py --generations 2 --population 3 --verbose
```

### 3. 日志分析

```bash
# 查看日志文件
tail -f parallel_evaluation.log

# 分析错误信息
grep "ERROR" parallel_evaluation.log
```

## 扩展开发

### 1. 添加新的评估器

```python
# 创建新的评估器类
class CustomJudge:
    def __init__(self):
        # 初始化代码
        pass
    
    def judge_hidden_motivation(self, prompt, response):
        # 评估逻辑
        return score, reasoning

# 在 ParallelEvaluator 中注册
evaluator.add_judge("custom_judge", CustomJudge())
```

### 2. 自定义适应度函数

```python
# 在 GeneticAlgorithm 中重写适应度计算
def _calculate_parallel_fitness(self, evaluation_result):
    # 自定义适应度计算逻辑
    return fitness_score
```

### 3. 添加新的配置选项

```python
# 在 config_loader.py 中扩展配置类
@dataclass
class CustomConfig:
    custom_parameter: float = 0.5
    custom_option: bool = True
```

## 最佳实践

### 1. 参数设置

- **小规模测试**: 先用小参数验证功能
- **逐步扩大**: 确认无误后增加参数规模
- **监控资源**: 注意内存和CPU使用情况

### 2. 结果分析

- **多次运行**: 进行多次实验确保结果稳定性
- **对比分析**: 比较不同配置的效果
- **记录实验**: 保存实验配置和结果

### 3. 安全考虑

- **内容过滤**: 启用内容过滤机制
- **访问控制**: 限制对敏感模型的访问
- **结果审查**: 人工审查生成的攻击内容

## 技术支持

### 1. 文档资源

- `PARALLEL_EVALUATION_README.md`: 详细技术文档
- `parallel_config.json`: 配置文件示例
- `demo_parallel_evaluation.py`: 使用示例

### 2. 示例脚本

- `run_parallel_evaluation.py`: 完整运行示例
- `parallel_evaluation_example.py`: 基础使用示例
- `config_loader.py`: 配置管理示例

### 3. 调试工具

- 配置验证器
- 性能监控
- 错误日志分析

---

**注意**: 本系统仅用于研究和测试目的，请确保在合法合规的框架内使用。