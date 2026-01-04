#!/usr/bin/env python3
"""
Professional plotting script for benchmark results analysis.
Creates bar charts showing performance metrics across different TP/PP combinations.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom color scheme - professional and pleasing
COLORS = {
    'requests_per_sec': '#2E86AB',      # Deep blue
    'input_tokens_per_sec': '#A23B72',  # Burgundy
    'output_tokens_per_sec': '#F18F01', # Orange
    'total_tokens_per_sec': '#C73E1D'   # Red
}

def load_and_process_data(csv_path):
    """Load CSV and process successful experiments."""
    df = pd.read_csv(csv_path)

    # Filter successful experiments only
    successful = df[df['status'] == 'success'].copy()

    # Ensure numeric columns are properly typed
    numeric_cols = ['requests_per_sec', 'input_tokens_per_sec',
                   'output_tokens_per_sec', 'total_tokens_per_sec', 'tokens_per_dollar']
    for col in numeric_cols:
        if col in successful.columns:
            successful[col] = pd.to_numeric(successful[col], errors='coerce')

    # Sort by tp, then pp
    successful = successful.sort_values(['tp', 'pp'])

    # Create labels for x-axis
    successful['config'] = successful.apply(lambda x: f'TP{x["tp"]}\nPP{x["pp"]}', axis=1)

    return successful

def create_performance_plot(df, save_path=None):
    """Create professional bar chart for performance metrics."""

    # Set up professional style
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'

    # Metrics to plot
    metrics = ['requests_per_sec', 'input_tokens_per_sec', 'output_tokens_per_sec', 'total_tokens_per_sec']
    metric_labels = ['Requests/sec', 'Input tokens/sec', 'Output tokens/sec', 'Total tokens/sec']

    # Enhanced color scheme - more professional and distinct
    COLORS_ENHANCED = {
        'requests_per_sec': '#1F4E79',      # Dark blue
        'input_tokens_per_sec': '#8B4513',  # Dark brown
        'output_tokens_per_sec': '#FF6B35', # Bright orange (not pink)
        'total_tokens_per_sec': '#DC143C'   # Crimson red
    }

    # Set up the plot with better proportions
    fig = plt.figure(figsize=(18, 12), facecolor='white')

    # Main performance plot with more space
    ax1 = plt.subplot(2, 1, 1)

    # Number of configurations and metrics
    n_configs = len(df)
    n_metrics = len(metrics)

    # Optimized bar width and spacing
    bar_width = 0.20
    group_width = n_metrics * bar_width + 0.08  # Add gap between groups
    group_centers = np.arange(n_configs) * (group_width + 0.15)  # More spacing between groups

    # Create bars for each metric
    max_value = 0
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        # Position for this metric's bars
        x_positions = group_centers - (group_width/2) + (i * bar_width) + (bar_width/2)

        # Values and errors (if available)
        values = df[metric].values
        # Filter out NaN values and ensure numeric
        values = values[~np.isnan(values)]
        if len(values) > 0:
            max_value = max(max_value, max(values))

        # Create bars with enhanced styling
        bars = ax1.bar(x_positions, values, bar_width,
                      label=label, color=COLORS_ENHANCED[metric],
                      alpha=0.9, edgecolor='white', linewidth=1.5,
                      zorder=3)

        # Add value labels on bars for throughput metrics
        if metric in ['total_tokens_per_sec', 'input_tokens_per_sec', 'output_tokens_per_sec']:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                # Format throughput metrics with 0 decimal places
                label_text = f'{value:.0f}'

                ax1.text(bar.get_x() + bar.get_width()/2., height + max_value*0.03,
                        label_text, ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='#1a1a1a',
                        rotation=45,  # Rotate 45 degrees as requested
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                                edgecolor='none', alpha=0.9))

    # Customize the main plot with better typography
    ax1.set_ylabel('Performance Metrics', fontsize=14, fontweight='bold', labelpad=20)
    ax1.set_title('DeepSeek-R1-Distill-Llama-70B Performance Analysis\n' +
                 'TP/PP Parallelism Configurations (Input: 8192 tokens, Output: 2048 tokens)',
                 fontsize=18, fontweight='bold', pad=30, color='#1a1a1a')

    # Set x-axis ticks and labels - NOW WITH LABELS!
    ax1.set_xticks(group_centers)
    ax1.set_xticklabels(df['config'].values, fontsize=12, fontweight='bold',
                       rotation=0, ha='center')

    # Make y-axis tick labels larger
    ax1.tick_params(axis='y', labelsize=12)

    # Enhanced grid
    ax1.grid(axis='y', alpha=0.3, linestyle=':', color='#666666', zorder=1)
    ax1.set_axisbelow(True)

    # Add subtle background pattern
    ax1.set_facecolor('#fafafa')

    # Enhanced legend with better positioning
    legend = ax1.legend(title='Performance Metrics', title_fontsize=13, fontsize=11,
                       bbox_to_anchor=(1.02, 0.98), loc='upper left',
                       frameon=True, fancybox=True, shadow=True, borderpad=1.5,
                       labelspacing=1.2)
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('#cccccc')

    # Second subplot for efficiency analysis and cost efficiency
    ax2 = plt.subplot(2, 1, 2)

    # Calculate efficiency metrics (scaling efficiency relative to baseline)
    # Find the configuration with minimum GPUs as baseline
    total_gpus = df['tp'] * df['pp']
    min_gpus_idx = total_gpus.idxmin()
    baseline_gpus = total_gpus.loc[min_gpus_idx]
    baseline_perf = df.loc[min_gpus_idx, 'total_tokens_per_sec']
    
    # Calculate efficiency: (per-GPU performance) / (baseline per-GPU performance) * 100
    per_gpu_perf = df['total_tokens_per_sec'] / total_gpus
    baseline_per_gpu = baseline_perf / baseline_gpus
    efficiency = (per_gpu_perf / baseline_per_gpu) * 100

    # Get tokens per dollar (handle missing values, ensure alignment with df index)
    tokens_per_dollar = df['tokens_per_dollar'].fillna(0)
    
    # Ensure values are in the same order as the dataframe (which matches group_centers)
    efficiency_values = efficiency.loc[df.index].values
    tokens_per_dollar_values = tokens_per_dollar.loc[df.index].values

    # Create efficiency bars on left y-axis
    bar_width_eff = group_width * 0.35
    bars_eff = ax2.bar(group_centers - bar_width_eff/2, efficiency_values, 
                      width=bar_width_eff, color='#2E8B57', alpha=0.85, 
                      edgecolor='white', linewidth=1.5, zorder=3, label='Scaling Efficiency (%)')

    # Add efficiency value labels
    for bar, eff in zip(bars_eff, efficiency_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{eff:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#1a1a1a',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                         edgecolor='none', alpha=0.8))

    # Create secondary y-axis for tokens per dollar
    ax2_right = ax2.twinx()

    # Create tokens per dollar bars on right y-axis
    bars_cost = ax2_right.bar(group_centers + bar_width_eff/2, tokens_per_dollar_values,
                              width=bar_width_eff, color='#4169E1', alpha=0.85,
                              edgecolor='white', linewidth=1.5, zorder=3, 
                              label='Tokens per Dollar')

    # Add tokens per dollar value labels
    max_tokens_per_dollar = tokens_per_dollar_values.max() if len(tokens_per_dollar_values) > 0 and tokens_per_dollar_values.max() > 0 else 1
    for bar, tpd in zip(bars_cost, tokens_per_dollar_values):
        if tpd > 0:  # Only label if value exists
            height = bar.get_height()
            ax2_right.text(bar.get_x() + bar.get_width()/2., height + max_tokens_per_dollar*0.02,
                          f'{tpd:.0f}', ha='center', va='bottom',
                          fontsize=9, fontweight='bold', color='#1a1a1a',
                          bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                                   edgecolor='none', alpha=0.8))

    # Customize left y-axis (efficiency)
    ax2.set_xlabel('Parallelism Configuration (TP/PP)', fontsize=14, fontweight='bold', labelpad=20)
    ax2.set_ylabel('Scaling Efficiency (%)', fontsize=14, fontweight='bold', labelpad=20, color='#2E8B57')
    ax2.set_title('Parallelization Scaling Efficiency & Cost Efficiency Analysis', 
                 fontsize=16, fontweight='bold', pad=20)

    ax2.set_xticks(group_centers)
    ax2.set_xticklabels(df['config'].values, fontsize=12, fontweight='bold')

    # Make y-axis tick labels larger
    ax2.tick_params(axis='y', labelsize=12, labelcolor='#2E8B57')

    # Customize right y-axis (tokens per dollar)
    ax2_right.set_ylabel('Tokens per Dollar', fontsize=14, fontweight='bold', 
                        labelpad=20, color='#4169E1')
    ax2_right.tick_params(axis='y', labelsize=12, labelcolor='#4169E1')

    # Add efficiency grid and reference line at 100%
    ax2.grid(axis='y', alpha=0.3, linestyle=':', color='#666666', zorder=1)
    ax2.axhline(y=100, color='#DC143C', linestyle='--', alpha=0.8, linewidth=2.5,
                label='Perfect Linear Scaling', zorder=2)
    ax2.set_axisbelow(True)
    ax2.set_facecolor('#fafafa')

    # Combine legends from both axes and position outside on the right (further right to avoid y-axis overlap)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    legend2 = ax2.legend(lines1 + lines2, labels1 + labels2, 
                       bbox_to_anchor=(1.1, 0.98), loc='upper left',
                       frameon=True, fancybox=True, shadow=True, fontsize=11, 
                       borderpad=1.5, labelspacing=1.2, title='Efficiency Metrics', 
                       title_fontsize=13)
    legend2.get_frame().set_alpha(0.95)
    legend2.get_frame().set_edgecolor('#cccccc')

    # Adjust layout for better spacing
    plt.subplots_adjust(hspace=0.25, top=0.92, bottom=0.08, left=0.08, right=0.85)

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Professional plot saved to: {save_path}")
    else:
        plt.show()

    return fig, (ax1, ax2)

def create_summary_table(df):
    """Create a summary table of key metrics."""
    summary_cols = ['config', 'tp', 'pp', 'requests_per_sec',
                   'input_tokens_per_sec', 'output_tokens_per_sec', 'total_tokens_per_sec']

    summary = df[summary_cols].copy()

    # Round values for display
    numeric_cols = ['requests_per_sec', 'input_tokens_per_sec',
                   'output_tokens_per_sec', 'total_tokens_per_sec']
    summary[numeric_cols] = summary[numeric_cols].round(2)

    return summary

def main():
    """Main function to run the analysis."""

    # File paths
    csv_file = sys.argv[1]
    # get the parent directory of the csv file
    input_dir = os.path.dirname(csv_file)
    print(f"Input directory: {input_dir}")
    output_plot = f'{input_dir}/benchmark_performance_analysis.pdf'
    print(f"Output plot: {output_plot}")

    # Check if CSV exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        return

    # Load and process data
    print("Loading benchmark results...")
    df = load_and_process_data(csv_file)

    if len(df) == 0:
        print("No successful experiments found in the CSV.")
        return

    print(f"Found {len(df)} successful experiments:")
    print(df[['tp', 'pp', 'requests_per_sec', 'total_tokens_per_sec']].to_string(index=False))

    # Create summary table
    summary = create_summary_table(df)
    print("\nPerformance Summary:")
    print(summary.to_string(index=False))

    # Create the plot
    print("\nGenerating professional plot...")
    fig, ax = create_performance_plot(df, save_path=output_plot)

    # Also show plot if running interactively
    try:
        plt.show()
    except:
        pass  # In case we're not in an interactive environment

    print("Analysis complete!")

if __name__ == "__main__":
    main()
