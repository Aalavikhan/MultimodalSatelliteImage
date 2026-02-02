# -*- coding: utf-8 -*-
"""
Comprehensive Visualization for Model Comparison
Generates publication-ready graphs showcasing your model's superiority
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from matplotlib.patches import Rectangle
from matplotlib import gridspec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
colors = {
    'traditional': '#FF6B6B',
    'deep_learning': '#4ECDC4',
    'proposed': '#45B7D1',
    'highlight': '#FFA07A'
}


def load_results(comparison_results_path='comparison_results.json', 
                 your_model_metrics=None):
    """
    Load comparison results and your model's metrics
    
    Args:
        comparison_results_path: Path to comparison_results.json
        your_model_metrics: Dict with your model's test metrics
    """
    # Load comparison results
    with open(comparison_results_path, 'r') as f:
        results = json.load(f)
    
    # Add your model results (from your training)
    if your_model_metrics is not None:
        results['Swin + Tabular (Ours)'] = your_model_metrics
    else:
        # Default placeholder - replace with actual results
        results['Swin + Tabular (Ours)'] = {
            'r2': 0.9876,
            'rmse': 6.12,
            'mape': 12.05,
            'wape': 9.80,
            'mae': 4.50,
            'within_5': 45.2,
            'within_10': 68.5
        }
    
    return results


def create_comparison_table(results, save_path='comparison_table.png'):
    """Create a beautiful comparison table"""
    
    # Prepare data
    models = list(results.keys())
    categories = []
    
    for model in models:
        if model in ['Linear Regression']:
            categories.append('Traditional ML')
        elif model in ['Random Forest', 'SVM']:
            categories.append('Traditional ML')
        elif model in ['VGG16', 'ResNet-50']:
            categories.append('Deep Learning Baselines')
        else:
            categories.append('Proposed Framework')
    
    data = []
    for model in models:
        data.append([
            model,
            f"{results[model]['r2']:.4f}",
            f"{results[model]['rmse']:.2f}",
            f"{results[model]['mape']:.2f}",
            f"{results[model]['wape']:.2f}"
        ])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=data,
        colLabels=['Model Architecture', 'R² Score', 'RMSE (kWh)', 'MAPE (%)', 'WAPE (%)'],
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.15, 0.15, 0.15, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style rows
    for i, model in enumerate(models, 1):
        # Color by category
        if categories[i-1] == 'Traditional ML':
            color = '#FFE5E5'
        elif categories[i-1] == 'Deep Learning Baselines':
            color = '#E5F5F9'
        else:
            color = '#D4EDDA'  # Highlight your model
        
        for j in range(5):
            cell = table[(i, j)]
            cell.set_facecolor(color)
            
            # Bold for your model
            if categories[i-1] == 'Proposed Framework':
                cell.set_text_props(weight='bold')
    
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison table saved: {save_path}")
    plt.close()


def create_metrics_comparison_bar(results, save_path='metrics_comparison.png'):
    """Create bar charts comparing all metrics"""
    
    models = list(results.keys())
    metrics = ['r2', 'rmse', 'mape', 'wape']
    metric_names = ['R² Score ↑', 'RMSE (kWh) ↓', 'MAPE (%) ↓', 'WAPE (%) ↓']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[model][metric] for model in models]
        
        # Colors: highlight your model
        bar_colors = ['#3498db' if 'Ours' not in model else '#2ecc71' for model in models]
        
        bars = axes[idx].bar(range(len(models)), values, color=bar_colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{val:.2f}' if metric != 'r2' else f'{val:.4f}',
                          ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        axes[idx].set_ylabel(name, fontsize=12, fontweight='bold')
        axes[idx].set_xticks(range(len(models)))
        axes[idx].set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add best/worst indicators
        if metric == 'r2':
            best_idx = np.argmax(values)
            axes[idx].axhline(y=values[best_idx], color='green', linestyle='--', alpha=0.5, label='Best')
        else:
            best_idx = np.argmin(values)
            axes[idx].axhline(y=values[best_idx], color='green', linestyle='--', alpha=0.5, label='Best')
        
        axes[idx].legend()
        axes[idx].set_title(name, fontsize=14, fontweight='bold')
    
    plt.suptitle('Comprehensive Metrics Comparison Across All Models', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Metrics comparison saved: {save_path}")
    plt.close()


def create_radar_chart(results, save_path='radar_comparison.png'):
    """Create radar chart for multi-metric comparison"""
    
    # Select top models for clarity
    selected_models = [
        'Linear Regression',
        'Random Forest', 
        'VGG16',
        'ResNet-50',
        'Swin + Tabular (Ours)'
    ]
    
    # Normalize metrics to 0-100 scale
    metrics = ['r2', 'rmse', 'mape', 'wape']
    labels = ['R² Score', 'RMSE', 'MAPE', 'WAPE']
    
    # Get all values for normalization
    all_values = {m: [] for m in metrics}
    for model in results:
        for metric in metrics:
            all_values[metric].append(results[model][metric])
    
    # Normalize (0-100 scale, higher is better)
    def normalize(value, metric):
        min_val = min(all_values[metric])
        max_val = max(all_values[metric])
        
        if metric == 'r2':
            # Higher is better, map directly
            return ((value - min_val) / (max_val - min_val + 1e-8)) * 100
        else:
            # Lower is better, invert
            return (1 - (value - min_val) / (max_val - min_val + 1e-8)) * 100
    
    # Prepare data
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors_radar = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#2ecc71']
    
    for idx, model in enumerate(selected_models):
        values = [normalize(results[model][m], m) for m in metrics]
        values += values[:1]  # Complete the circle
        
        linewidth = 3 if 'Ours' in model else 1.5
        alpha = 0.7 if 'Ours' in model else 0.3
        
        ax.plot(angles, values, 'o-', linewidth=linewidth, 
               label=model, color=colors_radar[idx], alpha=alpha)
        ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('Multi-Metric Performance Comparison\n(Higher is Better for All Axes)', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Radar chart saved: {save_path}")
    plt.close()


def create_improvement_analysis(results, save_path='improvement_analysis.png'):
    """Show improvement of your model over baselines"""
    
    your_model = 'Swin + Tabular (Ours)'
    baselines = [m for m in results.keys() if m != your_model]
    
    metrics = ['r2', 'rmse', 'mape', 'wape']
    improvements = {metric: [] for metric in metrics}
    
    for metric in metrics:
        your_value = results[your_model][metric]
        
        for baseline in baselines:
            baseline_value = results[baseline][metric]
            
            if metric == 'r2':
                # Higher is better
                improvement = ((your_value - baseline_value) / baseline_value) * 100
            else:
                # Lower is better
                improvement = ((baseline_value - your_value) / baseline_value) * 100
            
            improvements[metric].append(improvement)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    metric_names = {
        'r2': 'R² Score Improvement (%)',
        'rmse': 'RMSE Reduction (%)',
        'mape': 'MAPE Reduction (%)',
        'wape': 'WAPE Reduction (%)'
    }
    
    for idx, metric in enumerate(metrics):
        values = improvements[metric]
        colors_imp = ['green' if v > 0 else 'red' for v in values]
        
        bars = axes[idx].barh(baselines, values, color=colors_imp, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            axes[idx].text(width, bar.get_y() + bar.get_height()/2,
                          f'{val:+.1f}%',
                          ha='left' if width > 0 else 'right',
                          va='center', fontsize=10, fontweight='bold')
        
        axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=1)
        axes[idx].set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
        axes[idx].set_title(metric_names[metric], fontsize=13, fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)
        
        # Add average improvement line
        avg_improvement = np.mean(values)
        axes[idx].axvline(x=avg_improvement, color='blue', linestyle='--', 
                         linewidth=2, alpha=0.7, label=f'Avg: {avg_improvement:+.1f}%')
        axes[idx].legend()
    
    plt.suptitle('Performance Improvement of Proposed Model Over Baselines', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Improvement analysis saved: {save_path}")
    plt.close()


def create_accuracy_distribution(results, save_path='accuracy_distribution.png'):
    """Compare within-tolerance accuracy"""
    
    models = list(results.keys())
    
    # Extract within_5 and within_10 if available
    within_5 = []
    within_10 = []
    
    for model in models:
        within_5.append(results[model].get('within_5', 0))
        within_10.append(results[model].get('within_10', 0))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, within_5, width, label='Within ±5%', 
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, within_10, width, label='Within ±10%', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Percentage of Predictions (%)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Accuracy Within Tolerance Levels', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Accuracy distribution saved: {save_path}")
    plt.close()


def create_comprehensive_dashboard(results, save_path='comprehensive_dashboard.png'):
    """Create a comprehensive dashboard with all key metrics"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    models = list(results.keys())
    
    # 1. R² Comparison (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    r2_values = [results[m]['r2'] for m in models]
    colors_r2 = ['#2ecc71' if 'Ours' in m else '#3498db' for m in models]
    bars = ax1.barh(models, r2_values, color=colors_r2, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('R² Score', fontweight='bold')
    ax1.set_title('R² Score Comparison', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, r2_values):
        ax1.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                ha='left', va='center', fontsize=8, fontweight='bold')
    
    # 2. RMSE Comparison (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    rmse_values = [results[m]['rmse'] for m in models]
    colors_rmse = ['#2ecc71' if 'Ours' in m else '#e74c3c' for m in models]
    bars = ax2.bar(range(len(models)), rmse_values, color=colors_rmse, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('RMSE (kWh)', fontweight='bold')
    ax2.set_title('RMSE Comparison (Lower is Better)', fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, rmse_values):
        ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 3. MAPE Comparison (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    mape_values = [results[m]['mape'] for m in models]
    colors_mape = ['#2ecc71' if 'Ours' in m else '#f39c12' for m in models]
    bars = ax3.bar(range(len(models)), mape_values, color=colors_mape, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('MAPE (%)', fontweight='bold')
    ax3.set_title('MAPE Comparison (Lower is Better)', fontweight='bold')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, mape_values):
        ax3.text(bar.get_x() + bar.get_width()/2, val, f'{val:.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 4. WAPE Comparison (Middle Left)
    ax4 = fig.add_subplot(gs[1, 0])
    wape_values = [results[m]['wape'] for m in models]
    colors_wape = ['#2ecc71' if 'Ours' in m else '#9b59b6' for m in models]
    bars = ax4.barh(models, wape_values, color=colors_wape, alpha=0.8, edgecolor='black')
    ax4.set_xlabel('WAPE (%)', fontweight='bold')
    ax4.set_title('WAPE Comparison (Lower is Better)', fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, wape_values):
        ax4.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                ha='left', va='center', fontsize=8, fontweight='bold')
    
    # 5. Grouped Bar Chart (Middle Center & Right - span 2 columns)
    ax5 = fig.add_subplot(gs[1, 1:])
    metrics = ['R²', 'RMSE', 'MAPE', 'WAPE']
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, model in enumerate(models):
        values = [
            results[model]['r2'] * 100,  # Scale R² to 0-100
            results[model]['rmse'],
            results[model]['mape'],
            results[model]['wape']
        ]
        offset = (i - len(models)/2) * width
        color = '#2ecc71' if 'Ours' in model else f'C{i}'
        ax5.bar(x + offset, values, width, label=model, alpha=0.8, edgecolor='black', color=color)
    
    ax5.set_xlabel('Metrics', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Value', fontweight='bold', fontsize=11)
    ax5.set_title('All Metrics Side-by-Side', fontweight='bold', fontsize=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.legend(fontsize=8, ncol=2)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Performance Summary Table (Bottom - span all columns)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create summary statistics
    your_model = 'Swin + Tabular (Ours)'
    summary_data = []
    
    for metric in ['r2', 'rmse', 'mape', 'wape']:
        your_val = results[your_model][metric]
        all_vals = [results[m][metric] for m in models if m != your_model]
        
        if metric == 'r2':
            best = max(all_vals)
            improvement = ((your_val - best) / best) * 100
            rank = '1st' if your_val > best else '2nd'
        else:
            best = min(all_vals)
            improvement = ((best - your_val) / best) * 100
            rank = '1st' if your_val < best else '2nd'
        
        summary_data.append([
            metric.upper(),
            f'{your_val:.4f}' if metric == 'r2' else f'{your_val:.2f}',
            f'{best:.4f}' if metric == 'r2' else f'{best:.2f}',
            f'{improvement:+.1f}%',
            rank
        ])
    
    table = ax6.table(
        cellText=summary_data,
        colLabels=['Metric', 'Your Model', 'Best Baseline', 'Improvement', 'Rank'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.15, 0.15, 0.15, 0.1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, 5):
        for j in range(5):
            cell = table[(i, j)]
            cell.set_facecolor('#ecf0f1')
            if j == 3:  # Improvement column
                improvement_val = float(summary_data[i-1][3].rstrip('%'))
                if improvement_val > 0:
                    cell.set_facecolor('#d5f4e6')
            if j == 4:  # Rank column
                if summary_data[i-1][4] == '1st':
                    cell.set_facecolor('#d5f4e6')
                    cell.set_text_props(weight='bold')
    
    fig.suptitle('Comprehensive Model Comparison Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comprehensive dashboard saved: {save_path}")
    plt.close()


def generate_all_visualizations(comparison_results_path='comparison_results.json',
                                your_model_metrics=None):
    """Generate all visualization graphs"""
    
    print("="*80)
    print("📊 GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    
    # Load results
    results = load_results(comparison_results_path, your_model_metrics)
    
    print(f"\n✅ Loaded results for {len(results)} models")
    
    # Generate all visualizations
    print("\n🎨 Creating visualizations...")
    
    create_comparison_table(results, 'comparison_table.png')
    create_metrics_comparison_bar(results, 'metrics_comparison.png')
    create_radar_chart(results, 'radar_comparison.png')
    create_improvement_analysis(results, 'improvement_analysis.png')
    create_accuracy_distribution(results, 'accuracy_distribution.png')
    create_comprehensive_dashboard(results, 'comprehensive_dashboard.png')
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS GENERATED!")
    print("="*80)
    print("\n📁 Generated files:")
    print("   1. comparison_table.png - Publication-ready table")
    print("   2. metrics_comparison.png - Bar charts for all metrics")
    print("   3. radar_comparison.png - Multi-metric radar chart")
    print("   4. improvement_analysis.png - Improvement over baselines")
    print("   5. accuracy_distribution.png - Within-tolerance accuracy")
    print("   6. comprehensive_dashboard.png - All-in-one dashboard")


if __name__ == "__main__":
    # Example: Load your model's test metrics
    # Replace this with actual metrics from your training
    your_metrics = {
        'r2': 0.9876,
        'rmse': 6.12,
        'mape': 12.05,
        'wape': 9.80,
        'mae': 4.50,
        'smape': 11.5,
        'within_5': 45.2,
        'within_10': 68.5
    }
    
    # Generate all visualizations
    generate_all_visualizations(
        comparison_results_path='comparison_results.json',
        your_model_metrics=your_metrics
    )
