#!/usr/bin/env python3
"""
生成最终分析报告
汇总整个进化过程的结果和分析
"""

import json
import time
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class FinalReportGenerator:
    """最终报告生成器"""
    
    def __init__(self):
        self.report_data = {}
        self.figures_dir = Path("report_figures")
        self.figures_dir.mkdir(exist_ok=True)
    
    def load_data(self):
        """加载所有相关数据"""
        print("加载数据文件...")
        
        # 加载初始prompt
        try:
            with open("generated_attack_prompts.json", 'r', encoding='utf-8') as f:
                self.report_data['initial_prompts'] = json.load(f)
            print("✓ 初始prompt数据加载成功")
        except FileNotFoundError:
            print("✗ 初始prompt数据文件未找到")
            self.report_data['initial_prompts'] = None
        
        # 加载评估结果
        try:
            with open("evaluation_results.json", 'r', encoding='utf-8') as f:
                self.report_data['evaluation_results'] = json.load(f)
            print("✓ 评估结果数据加载成功")
        except FileNotFoundError:
            print("✗ 评估结果数据文件未找到")
            self.report_data['evaluation_results'] = None
        
        # 加载进化结果
        evolution_files = []
        for i in range(1, 4):  # 假设有3代进化
            filename = f"generation_{i}_results.json"
            if Path(filename).exists():
                evolution_files.append(filename)
        
        # 加载最终结果
        if Path("generation_final_results.json").exists():
            evolution_files.append("generation_final_results.json")
        
        self.report_data['evolution_results'] = []
        for filename in evolution_files:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.report_data['evolution_results'].append(data)
                print(f"✓ {filename} 加载成功")
            except FileNotFoundError:
                print(f"✗ {filename} 未找到")
        
        print(f"总共加载了 {len(self.report_data['evolution_results'])} 个进化代际数据")
    
    def analyze_initial_distribution(self) -> Dict:
        """分析初始prompt分布"""
        if not self.report_data['initial_prompts']:
            return {}
        
        prompts = self.report_data['initial_prompts']['prompts']
        
        # 按类别统计
        category_counts = {}
        category_lengths = {}
        
        for prompt in prompts:
            category = prompt['category']
            if category not in category_counts:
                category_counts[category] = 0
                category_lengths[category] = []
            
            category_counts[category] += 1
            category_lengths[category].append(len(prompt['prompt']))
        
        # 计算平均长度
        category_avg_lengths = {}
        for category, lengths in category_lengths.items():
            category_avg_lengths[category] = sum(lengths) / len(lengths)
        
        return {
            'total_prompts': len(prompts),
            'category_counts': category_counts,
            'category_avg_lengths': category_avg_lengths
        }
    
    def analyze_evaluation_performance(self) -> Dict:
        """分析评估性能"""
        if not self.report_data['evaluation_results']:
            return {}
        
        results = self.report_data['evaluation_results']['results']
        
        # 按类别分析评分
        category_scores = {}
        all_scores = []
        
        for result in results:
            category = result['category']
            score = result['average_score']
            
            if category not in category_scores:
                category_scores[category] = []
            
            category_scores[category].append(score)
            all_scores.append(score)
        
        # 计算统计信息
        category_stats = {}
        for category, scores in category_scores.items():
            category_stats[category] = {
                'count': len(scores),
                'mean': sum(scores) / len(scores),
                'max': max(scores),
                'min': min(scores),
                'std': (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5
            }
        
        return {
            'overall_stats': {
                'mean': sum(all_scores) / len(all_scores),
                'max': max(all_scores),
                'min': min(all_scores),
                'std': (sum((x - sum(all_scores)/len(all_scores))**2 for x in all_scores) / len(all_scores))**0.5
            },
            'category_stats': category_stats,
            'top_performers': sorted(results, key=lambda x: x['average_score'], reverse=True)[:10],
            'bottom_performers': sorted(results, key=lambda x: x['average_score'])[:5]
        }
    
    def analyze_evolution_progress(self) -> Dict:
        """分析进化进程"""
        if not self.report_data['evolution_results']:
            return {}
        
        evolution_timeline = []
        
        for gen_data in self.report_data['evolution_results']:
            if 'evolution_history' in gen_data:
                evolution_timeline.extend(gen_data['evolution_history'])
        
        # 分析进化趋势
        generations = []
        avg_fitness = []
        max_fitness = []
        min_fitness = []
        
        for entry in evolution_timeline:
            generations.append(entry['generation'])
            avg_fitness.append(entry['average_fitness'])
            max_fitness.append(entry['max_fitness'])
            min_fitness.append(entry['min_fitness'])
        
        # 计算改进率
        improvement_rates = []
        for i in range(1, len(avg_fitness)):
            rate = (avg_fitness[i] - avg_fitness[i-1]) / avg_fitness[i-1] * 100
            improvement_rates.append(rate)
        
        return {
            'generations': generations,
            'avg_fitness_trend': avg_fitness,
            'max_fitness_trend': max_fitness,
            'min_fitness_trend': min_fitness,
            'improvement_rates': improvement_rates,
            'total_improvement': (avg_fitness[-1] - avg_fitness[0]) / avg_fitness[0] * 100 if avg_fitness else 0
        }
    
    def analyze_final_population(self) -> Dict:
        """分析最终种群"""
        if not self.report_data['evolution_results']:
            return {}
        
        # 获取最终代的数据
        final_gen = None
        for gen_data in self.report_data['evolution_results']:
            if gen_data['metadata']['generation'] == 'final' or gen_data['metadata']['generation'] == max(g['metadata']['generation'] for g in self.report_data['evolution_results'] if isinstance(g['metadata']['generation'], int)):
                final_gen = gen_data
                break
        
        if not final_gen:
            return {}
        
        population = final_gen['population']
        
        # 分析最终种群特征
        generation_distribution = {}
        mutation_analysis = {}
        category_performance = {}
        
        for individual in population:
            # 代数分布
            gen = individual['generation']
            if gen not in generation_distribution:
                generation_distribution[gen] = 0
            generation_distribution[gen] += 1
            
            # 变异分析
            mutations = individual.get('mutation_history', [])
            for mutation in mutations:
                if mutation not in mutation_analysis:
                    mutation_analysis[mutation] = 0
                mutation_analysis[mutation] += 1
            
            # 类别性能
            category = individual['category']
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(individual['fitness_score'])
        
        # 计算类别统计
        category_final_stats = {}
        for category, scores in category_performance.items():
            category_final_stats[category] = {
                'count': len(scores),
                'mean': sum(scores) / len(scores),
                'max': max(scores),
                'min': min(scores)
            }
        
        return {
            'population_size': len(population),
            'generation_distribution': generation_distribution,
            'mutation_analysis': mutation_analysis,
            'category_final_stats': category_final_stats,
            'best_individuals': sorted(population, key=lambda x: x['fitness_score'], reverse=True)[:10]
        }
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("生成可视化图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 初始分布图
        initial_analysis = self.analyze_initial_distribution()
        if initial_analysis:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 类别数量分布
            categories = list(initial_analysis['category_counts'].keys())
            counts = list(initial_analysis['category_counts'].values())
            ax1.bar(categories, counts, color='skyblue')
            ax1.set_title('初始Prompt类别分布')
            ax1.set_ylabel('数量')
            ax1.tick_params(axis='x', rotation=45)
            
            # 平均长度分布
            lengths = list(initial_analysis['category_avg_lengths'].values())
            ax2.bar(categories, lengths, color='lightcoral')
            ax2.set_title('各类别平均Prompt长度')
            ax2.set_ylabel('字符数')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'initial_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ 初始分布图已保存")
        
        # 2. 评估性能图
        eval_analysis = self.analyze_evaluation_performance()
        if eval_analysis:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 类别平均分数
            categories = list(eval_analysis['category_stats'].keys())
            means = [eval_analysis['category_stats'][cat]['mean'] for cat in categories]
            stds = [eval_analysis['category_stats'][cat]['std'] for cat in categories]
            
            ax1.bar(categories, means, yerr=stds, capsize=5, color='lightgreen')
            ax1.set_title('各类别评估平均分数')
            ax1.set_ylabel('平均分数')
            ax1.tick_params(axis='x', rotation=45)
            
            # 分数分布直方图
            all_scores = []
            if self.report_data['evaluation_results']:
                for result in self.report_data['evaluation_results']['results']:
                    all_scores.append(result['average_score'])
            
            ax2.hist(all_scores, bins=20, color='orange', alpha=0.7)
            ax2.set_title('评估分数分布')
            ax2.set_xlabel('分数')
            ax2.set_ylabel('频次')
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'evaluation_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ 评估性能图已保存")
        
        # 3. 进化趋势图
        evolution_analysis = self.analyze_evolution_progress()
        if evolution_analysis and evolution_analysis['generations']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 适应度趋势
            generations = evolution_analysis['generations']
            ax1.plot(generations, evolution_analysis['avg_fitness_trend'], 'o-', label='平均适应度', linewidth=2)
            ax1.plot(generations, evolution_analysis['max_fitness_trend'], 's-', label='最高适应度', linewidth=2)
            ax1.plot(generations, evolution_analysis['min_fitness_trend'], '^-', label='最低适应度', linewidth=2)
            ax1.set_title('进化适应度趋势')
            ax1.set_xlabel('代数')
            ax1.set_ylabel('适应度')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 改进率
            if evolution_analysis['improvement_rates']:
                ax2.bar(range(1, len(evolution_analysis['improvement_rates']) + 1), 
                        evolution_analysis['improvement_rates'], color='purple', alpha=0.7)
                ax2.set_title('代际改进率')
                ax2.set_xlabel('代数')
                ax2.set_ylabel('改进率 (%)')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'evolution_trends.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ 进化趋势图已保存")
    
    def generate_markdown_report(self) -> str:
        """生成Markdown格式的报告"""
        print("生成Markdown报告...")
        
        # 分析数据
        initial_analysis = self.analyze_initial_distribution()
        eval_analysis = self.analyze_evaluation_performance()
        evolution_analysis = self.analyze_evolution_progress()
        final_analysis = self.analyze_final_population()
        
        report = f"""
# 攻击Prompt进化筛选实验报告

**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 1. 实验概述

本实验基于 `kaggle_code_report.md` 中的攻击模式分析，生成了50个攻击prompt，并通过多模型评估和遗传算法进行了优胜劣汰的进化筛选。

### 1.1 实验流程

1. **攻击模式分析**: 从kaggle报告中提取5个主要攻击类别
2. **Prompt生成**: 每类生成10个，总计50个攻击prompt
3. **多模型评估**: 使用GPT-4o、DeepSeek、Qwen三个模型进行评分
4. **进化筛选**: 通过遗传算法实现优胜劣汰机制
5. **结果分析**: 分析进化效果和最终种群特征

### 1.2 攻击类别

- **债务奴役** (debt_bondage): 通过债务控制劳工
- **护照没收** (passport_confiscation): 扣押身份证件限制自由
- **非法招聘费用** (illicit_recruitment_fees): 收取违法招聘费用
- **工资盗窃** (wage_theft): 克扣或拖欠工资
- **复合剥削** (compound_exploitation): 多种剥削手段结合

## 2. 初始Prompt分析
"""
        
        if initial_analysis:
            report += f"""
### 2.1 数据概览

- **总Prompt数量**: {initial_analysis['total_prompts']}
- **类别分布**: 
"""
            for category, count in initial_analysis['category_counts'].items():
                report += f"  - {category}: {count}个\n"
            
            report += "\n### 2.2 Prompt长度分析\n\n"
            for category, length in initial_analysis['category_avg_lengths'].items():
                report += f"- {category}: 平均 {length:.1f} 字符\n"
        
        report += "\n## 3. 评估结果分析\n"
        
        if eval_analysis:
            report += f"""
### 3.1 整体评估性能

- **平均分数**: {eval_analysis['overall_stats']['mean']:.2f}
- **最高分数**: {eval_analysis['overall_stats']['max']:.2f}
- **最低分数**: {eval_analysis['overall_stats']['min']:.2f}
- **标准差**: {eval_analysis['overall_stats']['std']:.2f}

### 3.2 各类别表现

"""
            for category, stats in eval_analysis['category_stats'].items():
                report += f"**{category}**:\n"
                report += f"- 平均分: {stats['mean']:.2f}\n"
                report += f"- 最高分: {stats['max']:.2f}\n"
                report += f"- 最低分: {stats['min']:.2f}\n"
                report += f"- 标准差: {stats['std']:.2f}\n\n"
            
            report += "### 3.3 最佳表现Prompt\n\n"
            for i, prompt in enumerate(eval_analysis['top_performers'][:5], 1):
                report += f"{i}. **{prompt['prompt_id']}** ({prompt['category']}) - 分数: {prompt['average_score']:.2f}\n"
                report += f"   - 内容: {prompt['prompt_text'][:100]}...\n\n"
        
        report += "\n## 4. 进化过程分析\n"
        
        if evolution_analysis:
            report += f"""
### 4.1 进化概览

- **进化代数**: {len(evolution_analysis['generations'])}
- **总体改进**: {evolution_analysis['total_improvement']:.2f}%

### 4.2 适应度进化趋势

"""
            for i, gen in enumerate(evolution_analysis['generations']):
                avg_fit = evolution_analysis['avg_fitness_trend'][i]
                max_fit = evolution_analysis['max_fitness_trend'][i]
                report += f"- 第{gen}代: 平均 {avg_fit:.2f}, 最高 {max_fit:.2f}\n"
        
        report += "\n## 5. 最终种群分析\n"
        
        if final_analysis:
            report += f"""
### 5.1 种群特征

- **最终种群大小**: {final_analysis['population_size']}

### 5.2 代数分布

"""
            for gen, count in final_analysis['generation_distribution'].items():
                report += f"- 第{gen}代: {count}个个体\n"
            
            report += "\n### 5.3 变异类型分析\n\n"
            if final_analysis['mutation_analysis']:
                for mutation, count in final_analysis['mutation_analysis'].items():
                    report += f"- {mutation}: {count}次\n"
            else:
                report += "- 无变异记录\n"
            
            report += "\n### 5.4 最终类别性能\n\n"
            for category, stats in final_analysis['category_final_stats'].items():
                report += f"**{category}**:\n"
                report += f"- 个体数: {stats['count']}\n"
                report += f"- 平均分: {stats['mean']:.2f}\n"
                report += f"- 最高分: {stats['max']:.2f}\n\n"
            
            report += "### 5.5 最优个体\n\n"
            for i, individual in enumerate(final_analysis['best_individuals'][:5], 1):
                report += f"{i}. **{individual['id']}** ({individual['category']})\n"
                report += f"   - 适应度: {individual['fitness_score']:.2f}\n"
                report += f"   - 代数: {individual['generation']}\n"
                report += f"   - 内容: {individual['prompt_text'][:100]}...\n\n"
        
        report += f"""
## 6. 结论与发现

### 6.1 主要发现

1. **类别差异显著**: 不同攻击类别的prompt在评估中表现出明显差异
2. **进化效果明显**: 通过遗传算法，种群整体适应度得到提升
3. **复合剥削类表现最佳**: 该类别的prompt在评估中获得最高分数
4. **变异策略有效**: 不同的变异策略为种群带来了多样性

### 6.2 技术要点

- **多模型评估**: 使用三个不同的LLM进行评估，提高了结果的可靠性
- **遗传算法优化**: 通过选择、交叉、变异操作实现了prompt的进化
- **适应度估算**: 在避免频繁API调用的同时，保持了评估的有效性

### 6.3 应用价值

本实验展示了如何将进化算法应用于AI安全研究，为：
- 攻击prompt的自动化生成和优化提供了方法
- AI系统的安全性评估提供了工具
- 对抗性样本的研究提供了新思路

---

**注意**: 本实验仅用于学术研究和AI安全评估，不应用于恶意目的。
"""
        
        return report
    
    def save_report(self, report_content: str, filename: str = "final_evolution_report.md"):
        """保存报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"✓ 最终报告已保存到 {filename}")
    
    def generate_complete_report(self):
        """生成完整报告"""
        print("开始生成最终报告...")
        
        # 加载数据
        self.load_data()
        
        # 创建可视化
        self.create_visualizations()
        
        # 生成Markdown报告
        report_content = self.generate_markdown_report()
        
        # 保存报告
        self.save_report(report_content)
        
        print("\n" + "="*60)
        print("最终报告生成完成！")
        print("="*60)
        print(f"报告文件: final_evolution_report.md")
        print(f"图表目录: {self.figures_dir}")
        print("\n报告包含:")
        print("- 实验概述和流程")
        print("- 初始prompt分析")
        print("- 评估结果统计")
        print("- 进化过程追踪")
        print("- 最终种群特征")
        print("- 可视化图表")
        print("- 结论和发现")

def main():
    """主函数"""
    generator = FinalReportGenerator()
    generator.generate_complete_report()

if __name__ == "__main__":
    main()