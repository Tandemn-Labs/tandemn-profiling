#!/usr/bin/env python3
"""
Professional plotting script for benchmark results analysis.
Creates bar charts showing performance metrics across different TP/PP combinations.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
                   'output_tokens_per_sec', 'total_tokens_per_sec', 'tokens_per_dollar', 'cost_for_run_usd',
                   'avg_sm_util_pct', 'avg_mem_bw_util_pct', 'elapsed_time']
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

    # Set up the plot with better proportions (3 subplots now)
    fig = plt.figure(figsize=(18, 16), facecolor='white')

    # Main performance plot with more space
    ax1 = plt.subplot(3, 1, 1)

    # Number of configurations and metrics
    n_configs = len(df)
    n_metrics = len(metrics)

    # Optimized bar width and spacing
    bar_width = 0.20
    group_width = n_metrics * bar_width + 0.08  # Add gap between groups
    group_spacing = 0.5  # Increased spacing between parallelism groups for better separation
    group_centers = np.arange(n_configs) * (group_width + group_spacing)  # More spacing between groups

    # Create bars for each metric with error bars showing variance
    max_value = 0
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        # Position for this metric's bars
        x_positions = group_centers - (group_width/2) + (i * bar_width) + (bar_width/2)

        # Get mean values for bar heights
        mean_col = f'{metric}_mean'
        min_col = f'{metric}_min'
        max_col = f'{metric}_max'
        
        if mean_col not in df.columns:
            # Fallback to original column if aggregation didn't happen
            values = df[metric].fillna(0).values
            errors_lower = None
            errors_upper = None
        else:
            values = df[mean_col].fillna(0).values
            # Calculate error bars: lower = mean - min, upper = max - mean
            if min_col in df.columns and max_col in df.columns:
                means = df[mean_col].fillna(0).values
                mins = df[min_col].fillna(0).values
                maxs = df[max_col].fillna(0).values
                errors_lower = np.maximum(0, means - mins)  # Distance from mean to min (ensure non-negative)
                errors_upper = np.maximum(0, maxs - means)  # Distance from mean to max (ensure non-negative)
                # Combine into error array format: [[lower1, lower2, ...], [upper1, upper2, ...]]
                errors = np.array([errors_lower, errors_upper])
            else:
                errors = None
        
        # Filter out NaN values and ensure numeric
        values = values[~np.isnan(values)]
        if len(values) > 0:
            max_value = max(max_value, max(values))

        # Create bars with enhanced styling and error bars
        if errors is not None:
            bars = ax1.bar(x_positions, values, bar_width,
                          yerr=errors, capsize=3,
                          label=label, color=COLORS_ENHANCED[metric],
                          alpha=0.9, edgecolor='white', linewidth=1.5,
                          error_kw={'elinewidth': 1.5, 'ecolor': '#333333', 'alpha': 0.7, 'capthick': 1.5},
                          zorder=3)
        else:
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

    # Create secondary y-axis for cost (total price)
    ax1_right = ax1.twinx()
    
    # Get cost values (use mean if aggregated)
    cost_col = 'cost_for_run_usd_mean' if 'cost_for_run_usd_mean' in df.columns else 'cost_for_run_usd'
    cost_values = df[cost_col].fillna(0).values
    max_cost = max(cost_values) if len(cost_values) > 0 and max(cost_values) > 0 else 1
    
    # Get cost error bars if available
    cost_min_col = 'cost_for_run_usd_min' if 'cost_for_run_usd_min' in df.columns else None
    cost_max_col = 'cost_for_run_usd_max' if 'cost_for_run_usd_max' in df.columns else None
    cost_errors = None
    if cost_min_col and cost_max_col:
        cost_means = df[cost_col].fillna(0).values
        cost_mins = df[cost_min_col].fillna(0).values
        cost_maxs = df[cost_max_col].fillna(0).values
        cost_errors_lower = np.maximum(0, cost_means - cost_mins)
        cost_errors_upper = np.maximum(0, cost_maxs - cost_means)
        cost_errors = np.array([cost_errors_lower, cost_errors_upper])
    
    # Position for cost bars (fifth bar, after the four performance metrics)
    cost_x_positions = group_centers - (group_width/2) + (4 * bar_width) + (bar_width/2)
    
    # Create cost bars on right y-axis with error bars
    if cost_errors is not None:
        bars_cost = ax1_right.bar(cost_x_positions, cost_values, bar_width,
                                 yerr=cost_errors, capsize=3,
                                 label='Total Cost ($)', color='#8B008B',  # Dark magenta/purple
                                 alpha=0.9, edgecolor='white', linewidth=1.5,
                                 error_kw={'elinewidth': 1.5, 'ecolor': '#333333', 'alpha': 0.7, 'capthick': 1.5},
                                 zorder=3)
    else:
        bars_cost = ax1_right.bar(cost_x_positions, cost_values, bar_width,
                                 label='Total Cost ($)', color='#8B008B',  # Dark magenta/purple
                                 alpha=0.9, edgecolor='white', linewidth=1.5,
                                 zorder=3)
    
    # Add cost value labels
    for bar, cost in zip(bars_cost, cost_values):
        if cost > 0:  # Only label if value exists
            height = bar.get_height()
            label_text = f'${cost:.2f}'
            ax1_right.text(bar.get_x() + bar.get_width()/2., height + max_cost*0.03,
                         label_text, ha='center', va='bottom',
                         fontsize=9, fontweight='bold', color='#1a1a1a',
                         rotation=45,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                                 edgecolor='none', alpha=0.9))

    # Create tertiary y-axis for elapsed time (total time)
    ax1_right_time = ax1.twinx()
    # Offset the time axis to the right of the cost axis
    ax1_right_time.spines['right'].set_position(('outward', 60))
    
    # Get elapsed time values (in seconds) - use mean if aggregated
    time_col = 'elapsed_time_mean' if 'elapsed_time_mean' in df.columns else 'elapsed_time'
    time_values = df[time_col].fillna(0).values
    max_time = max(time_values) if len(time_values) > 0 and max(time_values) > 0 else 1
    
    # Get time error bars if available
    time_min_col = 'elapsed_time_min' if 'elapsed_time_min' in df.columns else None
    time_max_col = 'elapsed_time_max' if 'elapsed_time_max' in df.columns else None
    time_errors = None
    if time_min_col and time_max_col:
        time_means = df[time_col].fillna(0).values
        time_mins = df[time_min_col].fillna(0).values
        time_maxs = df[time_max_col].fillna(0).values
        time_errors_lower = np.maximum(0, time_means - time_mins)
        time_errors_upper = np.maximum(0, time_maxs - time_means)
        time_errors = np.array([time_errors_lower, time_errors_upper])
    
    # Position for time bars (sixth bar, after cost)
    time_x_positions = group_centers - (group_width/2) + (5 * bar_width) + (bar_width/2)
    
    # Create time bars on tertiary y-axis with error bars
    if time_errors is not None:
        bars_time = ax1_right_time.bar(time_x_positions, time_values, bar_width,
                                       yerr=time_errors, capsize=3,
                                       label='Total Time (s)', color='#FF1493',  # Deep pink
                                       alpha=0.9, edgecolor='white', linewidth=1.5,
                                       error_kw={'elinewidth': 1.5, 'ecolor': '#333333', 'alpha': 0.7, 'capthick': 1.5},
                                       zorder=3)
    else:
        bars_time = ax1_right_time.bar(time_x_positions, time_values, bar_width,
                                       label='Total Time (s)', color='#FF1493',  # Deep pink
                                       alpha=0.9, edgecolor='white', linewidth=1.5,
                                       zorder=3)
    
    # Add time value labels
    for bar, time_val in zip(bars_time, time_values):
        if time_val > 0:  # Only label if value exists
            height = bar.get_height()
            # Format time: show as seconds with 1 decimal place, or minutes if > 60s
            if time_val >= 60:
                minutes = int(time_val // 60)
                seconds = time_val % 60
                label_text = f'{minutes}m{seconds:.0f}s'
            else:
                label_text = f'{time_val:.1f}s'
            ax1_right_time.text(bar.get_x() + bar.get_width()/2., height + max_time*0.03,
                              label_text, ha='center', va='bottom',
                              fontsize=9, fontweight='bold', color='#1a1a1a',
                              rotation=45,
                              bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                                      edgecolor='none', alpha=0.9))

    # Customize the main plot with better typography
    ax1.set_ylabel('Performance Metrics', fontsize=14, fontweight='bold', labelpad=20)
    
    # Get input/output length (use first value or aggregated column)
    input_len_col = 'max_input_length_first' if 'max_input_length_first' in df.columns else 'max_input_length'
    output_len_col = 'max_output_length_first' if 'max_output_length_first' in df.columns else 'max_output_length'
    input_len = df[input_len_col].values[0] if len(df) > 0 and input_len_col in df.columns else 'N/A'
    output_len = df[output_len_col].values[0] if len(df) > 0 and output_len_col in df.columns else 'N/A'
    
    ax1.set_title('DeepSeek-R1-Distill-Llama-70B Performance Analysis\n' +
                 f'TP/PP Parallelism Configurations (Input: {input_len} tokens, Output: {output_len} tokens)',
                 fontsize=18, fontweight='bold', pad=30, color='#1a1a1a')

    # Set x-axis ticks and labels - NOW WITH LABELS!
    ax1.set_xticks(group_centers)
    ax1.set_xticklabels(df['config'].values, fontsize=12, fontweight='bold',
                       rotation=0, ha='center')

    # Make y-axis tick labels larger
    ax1.tick_params(axis='y', labelsize=12)

    # Customize right y-axis (cost)
    ax1_right.set_ylabel('Total Cost ($)', fontsize=14, fontweight='bold', 
                         labelpad=20, color='#8B008B')
    ax1_right.tick_params(axis='y', labelsize=12, labelcolor='#8B008B')

    # Customize tertiary y-axis (time)
    ax1_right_time.set_ylabel('Total Time (s)', fontsize=14, fontweight='bold',
                              labelpad=20, color='#FF1493')
    ax1_right_time.tick_params(axis='y', labelsize=12, labelcolor='#FF1493')
    
    # Align all three y-axes to have the same number of ticks and intervals
    # Calculate max values for normalization (use mean columns if aggregated)
    req_col = 'requests_per_sec_mean' if 'requests_per_sec_mean' in df.columns else 'requests_per_sec'
    input_col = 'input_tokens_per_sec_mean' if 'input_tokens_per_sec_mean' in df.columns else 'input_tokens_per_sec'
    output_col = 'output_tokens_per_sec_mean' if 'output_tokens_per_sec_mean' in df.columns else 'output_tokens_per_sec'
    total_col = 'total_tokens_per_sec_mean' if 'total_tokens_per_sec_mean' in df.columns else 'total_tokens_per_sec'
    
    requests_per_sec_vals = df[req_col].fillna(0).values
    input_tokens_per_sec_vals = df[input_col].fillna(0).values
    output_tokens_per_sec_vals = df[output_col].fillna(0).values
    total_tokens_per_sec_vals = df[total_col].fillna(0).values
    
    max_perf = max(max(requests_per_sec_vals), max(input_tokens_per_sec_vals), 
                   max(output_tokens_per_sec_vals), max(total_tokens_per_sec_vals))
    max_perf = max_perf if max_perf > 0 else 1
    max_cost = max(cost_values) if len(cost_values) > 0 and max(cost_values) > 0 else 1
    max_time = max(time_values) if len(time_values) > 0 and max(time_values) > 0 else 1
    
    # Use 5 ticks for all axes
    num_ticks = 5
    
    # Set y-limits and ticks for performance axis (left)
    ax1.set_ylim([0, max_perf * 1.1])
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
    
    # Set y-limits and ticks for cost axis (right inner)
    ax1_right.set_ylim([0, max_cost * 1.1])
    ax1_right.yaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
    
    # Set y-limits and ticks for time axis (right outer)
    ax1_right_time.set_ylim([0, max_time * 1.1])
    ax1_right_time.yaxis.set_major_locator(MaxNLocator(nbins=num_ticks))

    # Enhanced grid (only horizontal lines, no vertical grid)
    ax1.grid(axis='y', alpha=0.3, linestyle=':', color='#666666', zorder=1)
    ax1.grid(axis='x', visible=False)  # Explicitly disable vertical grid
    ax1.set_axisbelow(True)
    # Disable grid on time axis
    ax1_right_time.grid(False)
    
    # Add vertical separator lines between parallelism configurations
    # Place separators at the boundaries between groups (not in the middle)
    for i in range(n_configs - 1):
        # Calculate the position between this group and the next
        separator_x = group_centers[i] + group_width/2 + group_spacing/2
        ax1.axvline(x=separator_x, color='#999999', linestyle='-', linewidth=1.5, 
                   alpha=0.5, zorder=2)

    # Add subtle background pattern
    ax1.set_facecolor('#fafafa')

    # Enhanced legend with better positioning - combine all axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    lines3, labels3 = ax1_right_time.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, 
                       title='Performance Metrics, Cost & Time', title_fontsize=13, fontsize=11,
                       bbox_to_anchor=(1.1, 0.98), loc='upper left',
                       frameon=True, fancybox=True, shadow=True, borderpad=1.5,
                       labelspacing=1.2)
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('#cccccc')

    # Second subplot for efficiency analysis and cost efficiency
    ax2 = plt.subplot(3, 1, 2)

    # Calculate efficiency metrics (scaling efficiency relative to baseline)
    # Find the configuration with minimum GPUs as baseline
    total_gpus = df['tp'] * df['pp']
    min_gpus_idx = total_gpus.idxmin()
    baseline_gpus = total_gpus.loc[min_gpus_idx]
    # Use mean column if aggregated
    perf_col = 'total_tokens_per_sec_mean' if 'total_tokens_per_sec_mean' in df.columns else 'total_tokens_per_sec'
    baseline_perf = df.loc[min_gpus_idx, perf_col]
    
    # Calculate PERFORMANCE scaling efficiency: (per-GPU performance) / (baseline per-GPU performance) * 100
    per_gpu_perf = df[perf_col] / total_gpus
    baseline_per_gpu = baseline_perf / baseline_gpus
    performance_efficiency = (per_gpu_perf / baseline_per_gpu) * 100

    # Calculate COST efficiency (tokens per dollar efficiency relative to baseline)
    # Use mean column if aggregated
    tpd_col = 'tokens_per_dollar_mean' if 'tokens_per_dollar_mean' in df.columns else 'tokens_per_dollar'
    tokens_per_dollar = df[tpd_col].fillna(0)
    baseline_tokens_per_dollar = tokens_per_dollar.loc[min_gpus_idx]
    
    # Cost efficiency: (tokens_per_dollar / baseline_tokens_per_dollar) * 100
    # This shows how cost efficiency scales compared to baseline
    if baseline_tokens_per_dollar > 0:
        cost_efficiency = (tokens_per_dollar / baseline_tokens_per_dollar) * 100
    else:
        cost_efficiency = pd.Series([0] * len(df), index=df.index)
    
    # Ensure values are in the same order as the dataframe (which matches group_centers)
    performance_efficiency_values = performance_efficiency.loc[df.index].values
    cost_efficiency_values = cost_efficiency.loc[df.index].values

    # Create performance efficiency bars on left y-axis
    bar_width_eff = group_width * 0.35
    bars_perf_eff = ax2.bar(group_centers - bar_width_eff/2, performance_efficiency_values, 
                      width=bar_width_eff, color='#2E8B57', alpha=0.85, 
                      edgecolor='white', linewidth=1.5, zorder=3, label='Performance Scaling Efficiency (%)')

    # Add performance efficiency value labels
    for bar, eff in zip(bars_perf_eff, performance_efficiency_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{eff:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#1a1a1a',
                rotation=45,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                         edgecolor='none', alpha=0.8))

    # Create secondary y-axis for cost efficiency
    ax2_right = ax2.twinx()

    # Create cost efficiency bars on right y-axis
    bars_cost_eff = ax2_right.bar(group_centers + bar_width_eff/2, cost_efficiency_values,
                              width=bar_width_eff, color='#4169E1', alpha=0.85,
                              edgecolor='white', linewidth=1.5, zorder=3, 
                              label='Cost Efficiency (%)')

    # Add cost efficiency value labels
    for bar, ceff in zip(bars_cost_eff, cost_efficiency_values):
        if ceff > 0:  # Only label if value exists
            height = bar.get_height()
            ax2_right.text(bar.get_x() + bar.get_width()/2., height + 2,
                          f'{ceff:.1f}%', ha='center', va='bottom',
                          fontsize=9, fontweight='bold', color='#1a1a1a',
                          rotation=45,
                          bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                                   edgecolor='none', alpha=0.8))

    # Customize left y-axis (performance efficiency)
    # ax2.set_xlabel('Parallelism Configuration (TP/PP)', fontsize=14, fontweight='bold', labelpad=20)
    ax2.set_ylabel('Performance Scaling Efficiency (%)', fontsize=14, fontweight='bold', labelpad=20, color='#2E8B57')
    ax2.set_title('Performance Scaling Efficiency & Cost Efficiency Analysis', 
                 fontsize=16, fontweight='bold', pad=20)

    ax2.set_xticks(group_centers)
    ax2.set_xticklabels(df['config'].values, fontsize=12, fontweight='bold')

    # Make y-axis tick labels larger
    ax2.tick_params(axis='y', labelsize=12, labelcolor='#2E8B57')

    # Customize right y-axis (cost efficiency)
    ax2_right.set_ylabel('Cost Efficiency (%)', fontsize=14, fontweight='bold', 
                        labelpad=20, color='#4169E1')
    ax2_right.tick_params(axis='y', labelsize=12, labelcolor='#4169E1')

    # Calculate max values for both efficiency metrics to align y-axes
    max_perf_eff = max(performance_efficiency_values) if len(performance_efficiency_values) > 0 else 100
    max_cost_eff = max(cost_efficiency_values) if len(cost_efficiency_values) > 0 else 100
    # Add some padding (10% or at least 20 units)
    max_value = max(max_perf_eff, max_cost_eff, 100)
    y_max = max_value * 1.1 if max_value > 0 else 120
    y_min = 0
    
    # Set both axes to the same limits so 100% aligns at the same visual height
    ax2.set_ylim([y_min, y_max])
    ax2_right.set_ylim([y_min, y_max])

    # Add efficiency grid and reference lines at 100%
    ax2.grid(axis='y', alpha=0.3, linestyle=':', color='#666666', zorder=1)
    ax2.axhline(y=100, color='#DC143C', linestyle='--', alpha=0.8, linewidth=2.5,
                label='Perfect Performance Scaling', zorder=2)
    # Add reference line for cost efficiency on right axis (will align with left axis 100% line)
    ax2_right.axhline(y=100, color='#8B008B', linestyle='--', alpha=0.8, linewidth=2.5,
                     label='Perfect Cost Efficiency', zorder=2)
    ax2.set_axisbelow(True)
    ax2.set_facecolor('#fafafa')

    # Combine legends from both axes and position outside on the right (further right to avoid y-axis overlap)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    legend2 = ax2.legend(lines1 + lines2, labels1 + labels2, 
                       bbox_to_anchor=(1.1, 0.98), loc='upper left',
                       frameon=True, fancybox=True, shadow=True, fontsize=11, 
                       borderpad=1.5, labelspacing=1.2, title='Efficiency Metrics (%)', 
                       title_fontsize=13)
    legend2.get_frame().set_alpha(0.95)
    legend2.get_frame().set_edgecolor('#cccccc')

    # Third subplot for GPU utilization metrics
    ax3 = plt.subplot(3, 1, 3)
    
    # Get GPU utilization metrics (use mean if aggregated)
    sm_col = 'avg_sm_util_pct_mean' if 'avg_sm_util_pct_mean' in df.columns else 'avg_sm_util_pct'
    membw_col = 'avg_mem_bw_util_pct_mean' if 'avg_mem_bw_util_pct_mean' in df.columns else 'avg_mem_bw_util_pct'
    avg_sm_util = df[sm_col].fillna(0)
    avg_mem_bw_util = df[membw_col].fillna(0)
    
    # Get error bars for GPU utilization if available
    sm_min_col = 'avg_sm_util_pct_min' if 'avg_sm_util_pct_min' in df.columns else None
    sm_max_col = 'avg_sm_util_pct_max' if 'avg_sm_util_pct_max' in df.columns else None
    membw_min_col = 'avg_mem_bw_util_pct_min' if 'avg_mem_bw_util_pct_min' in df.columns else None
    membw_max_col = 'avg_mem_bw_util_pct_max' if 'avg_mem_bw_util_pct_max' in df.columns else None
    
    sm_errors = None
    membw_errors = None
    if sm_min_col and sm_max_col:
        sm_means = df[sm_col].fillna(0).values
        sm_mins = df[sm_min_col].fillna(0).values
        sm_maxs = df[sm_max_col].fillna(0).values
        sm_errors_lower = np.maximum(0, sm_means - sm_mins)
        sm_errors_upper = np.maximum(0, sm_maxs - sm_means)
        sm_errors = np.array([sm_errors_lower, sm_errors_upper])
    
    if membw_min_col and membw_max_col:
        membw_means = df[membw_col].fillna(0).values
        membw_mins = df[membw_min_col].fillna(0).values
        membw_maxs = df[membw_max_col].fillna(0).values
        membw_errors_lower = np.maximum(0, membw_means - membw_mins)
        membw_errors_upper = np.maximum(0, membw_maxs - membw_means)
        membw_errors = np.array([membw_errors_lower, membw_errors_upper])
    
    # Ensure values are aligned with group_centers
    avg_sm_values = avg_sm_util.loc[df.index].values
    avg_mem_bw_values = avg_mem_bw_util.loc[df.index].values
    
    # Create bars for GPU utilization with error bars
    bar_width_util = group_width * 0.35
    if sm_errors is not None:
        bars_sm = ax3.bar(group_centers - bar_width_util/2, avg_sm_values,
                          width=bar_width_util, yerr=sm_errors, capsize=3,
                          color='#FF6B35', alpha=0.85,
                          edgecolor='white', linewidth=1.5, zorder=3,
                          error_kw={'elinewidth': 1.5, 'ecolor': '#333333', 'alpha': 0.7, 'capthick': 1.5},
                          label='Avg SM Utilization (%)')
    else:
        bars_sm = ax3.bar(group_centers - bar_width_util/2, avg_sm_values,
                          width=bar_width_util, color='#FF6B35', alpha=0.85,
                          edgecolor='white', linewidth=1.5, zorder=3, 
                          label='Avg SM Utilization (%)')
    
    # Add SM utilization value labels
    for bar, sm_val in zip(bars_sm, avg_sm_values):
        if sm_val > 0:  # Only label if value exists
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{sm_val:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#1a1a1a',
                    rotation=45,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                             edgecolor='none', alpha=0.8))
    
    # Create secondary y-axis for memory bandwidth (though both are percentages, using same scale)
    ax3_right = ax3.twinx()
    
    # Create memory bandwidth bars with error bars
    if membw_errors is not None:
        bars_membw = ax3_right.bar(group_centers + bar_width_util/2, avg_mem_bw_values,
                                   width=bar_width_util, yerr=membw_errors, capsize=3,
                                   color='#4ECDC4', alpha=0.85,
                                   edgecolor='white', linewidth=1.5, zorder=3,
                                   error_kw={'elinewidth': 1.5, 'ecolor': '#333333', 'alpha': 0.7, 'capthick': 1.5},
                                   label='Avg Memory BW Utilization (%)')
    else:
        bars_membw = ax3_right.bar(group_centers + bar_width_util/2, avg_mem_bw_values,
                                   width=bar_width_util, color='#4ECDC4', alpha=0.85,
                                   edgecolor='white', linewidth=1.5, zorder=3,
                                   label='Avg Memory BW Utilization (%)')
    
    # Add memory bandwidth value labels
    for bar, membw_val in zip(bars_membw, avg_mem_bw_values):
        if membw_val > 0:  # Only label if value exists
            height = bar.get_height()
            ax3_right.text(bar.get_x() + bar.get_width()/2., height + 2,
                          f'{membw_val:.1f}%', ha='center', va='bottom',
                          fontsize=9, fontweight='bold', color='#1a1a1a',
                          rotation=45,
                          bbox=dict(boxstyle="round,pad=0.2", facecolor='white',
                                   edgecolor='none', alpha=0.8))
    
    # Customize left y-axis (SM utilization)
    # ax3.set_xlabel('Parallelism Configuration (TP/PP)', fontsize=14, fontweight='bold', labelpad=20)
    ax3.set_ylabel('Avg SM Utilization (%)', fontsize=14, fontweight='bold', 
                   labelpad=20, color='#FF6B35')
    ax3.set_title('GPU Utilization Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax3.set_xticks(group_centers)
    ax3.set_xticklabels(df['config'].values, fontsize=12, fontweight='bold')
    
    # Make y-axis tick labels larger
    ax3.tick_params(axis='y', labelsize=12, labelcolor='#FF6B35')
    ax3.set_ylim([0, 105])  # 0-100% range with some padding
    
    # Customize right y-axis (memory bandwidth)
    ax3_right.set_ylabel('Avg Memory BW Utilization (%)', fontsize=14, fontweight='bold',
                        labelpad=20, color='#4ECDC4')
    ax3_right.tick_params(axis='y', labelsize=12, labelcolor='#4ECDC4')
    ax3_right.set_ylim([0, 105])  # 0-100% range with some padding
    
    # Add grid and reference lines
    ax3.grid(axis='y', alpha=0.3, linestyle=':', color='#666666', zorder=1)
    ax3.axhline(y=100, color='#DC143C', linestyle='--', alpha=0.5, linewidth=1.5,
                zorder=2, label='100% Utilization')
    ax3.set_axisbelow(True)
    ax3.set_facecolor('#fafafa')
    
    # Combine legends from both axes
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax3_right.get_legend_handles_labels()
    legend3 = ax3.legend(lines3 + lines4, labels3 + labels4,
                       bbox_to_anchor=(1.1, 0.98), loc='upper left',
                       frameon=True, fancybox=True, shadow=True, fontsize=11,
                       borderpad=1.5, labelspacing=1.2, title='GPU Utilization Metrics',
                       title_fontsize=13)
    legend3.get_frame().set_alpha(0.95)
    legend3.get_frame().set_edgecolor('#cccccc')

    # Adjust layout for better spacing (3 subplots now, increased right margin for third y-axis)
    plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.06, left=0.08, right=0.90)

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Professional plot saved to: {save_path}")
    else:
        plt.show()

    return fig, (ax1, ax2, ax3)

def create_summary_table(df):
    """Create a summary table of key metrics."""
    # Use aggregated column names if available, otherwise use original
    req_col = 'requests_per_sec_mean' if 'requests_per_sec_mean' in df.columns else 'requests_per_sec'
    input_col = 'input_tokens_per_sec_mean' if 'input_tokens_per_sec_mean' in df.columns else 'input_tokens_per_sec'
    output_col = 'output_tokens_per_sec_mean' if 'output_tokens_per_sec_mean' in df.columns else 'output_tokens_per_sec'
    total_col = 'total_tokens_per_sec_mean' if 'total_tokens_per_sec_mean' in df.columns else 'total_tokens_per_sec'
    
    summary_cols = ['config', 'tp', 'pp', req_col, input_col, output_col, total_col]
    
    # Only include columns that exist
    available_cols = [col for col in summary_cols if col in df.columns]
    summary = df[available_cols].copy()

    # Round values for display
    numeric_cols = [req_col, input_col, output_col, total_col]
    numeric_cols = [col for col in numeric_cols if col in summary.columns]
    if numeric_cols:
        summary[numeric_cols] = summary[numeric_cols].round(2)

    return summary

def find_and_merge_csvs(directory):
    """Find all benchmark_results*.csv files in subdirectories and merge them."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all benchmark_results*.csv files recursively
    csv_files = list(directory.rglob("benchmark_results*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No benchmark_results*.csv files found in {directory}")
    
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file.relative_to(directory)}")
    
    # Load and merge all CSVs
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                dfs.append(df)
                print(f"  ✓ Loaded {len(df)} rows from {csv_file.name}")
        except Exception as e:
            print(f"  ⚠️  Skipped {csv_file.name}: {e}")
    
    if not dfs:
        raise ValueError("No valid data found in any CSV files")
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\nMerged {len(merged_df)} total rows from {len(dfs)} CSV files")
    
    # Save merged CSV in the input directory
    merged_csv_path = directory / "benchmark_results_merged.csv"
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Saved merged CSV to: {merged_csv_path}")
    
    return merged_df, merged_csv_path

def main():
    """Main function to run the analysis."""

    # Accept directory path instead of CSV file
    if len(sys.argv) < 2:
        print("Usage: python plot_benchmark_results.py <directory_path>")
        print("Example: python plot_benchmark_results.py result-16384in_2048out/aws-g6e-L40S")
        return
    
    input_dir = Path(sys.argv[1])
    
    print(f"Input directory: {input_dir}")
    output_plot = input_dir / 'benchmark_performance_analysis.pdf'
    print(f"Output plot: {output_plot}")

    # Check if directory exists
    if not input_dir.exists():
        print(f"Error: Directory not found at {input_dir}")
        return

    # Find and merge all CSV files from subdirectories
    print("\nSearching for benchmark_results*.csv files in subdirectories...")
    try:
        merged_df, merged_csv_path = find_and_merge_csvs(input_dir)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Process merged data (filter successful experiments and prepare for plotting)
    print("\nProcessing merged data...")
    # Filter successful experiments only
    successful = merged_df[merged_df['status'] == 'success'].copy()

    # Ensure numeric columns are properly typed
    numeric_cols = ['requests_per_sec', 'input_tokens_per_sec',
                   'output_tokens_per_sec', 'total_tokens_per_sec', 'tokens_per_dollar', 'cost_for_run_usd',
                   'avg_sm_util_pct', 'avg_mem_bw_util_pct', 'elapsed_time']
    for col in numeric_cols:
        if col in successful.columns:
            successful[col] = pd.to_numeric(successful[col], errors='coerce')

    # Group by TP and PP to aggregate multiple runs
    print("\nGrouping by TP/PP configuration...")
    grouped = successful.groupby(['tp', 'pp'], as_index=False)
    
    # Calculate statistics for each metric
    agg_dict = {}
    for col in numeric_cols:
        if col in successful.columns:
            agg_dict[col] = ['mean', 'min', 'max', 'std']
    
    # Also keep first value for non-numeric columns we need
    for col in ['max_input_length', 'max_output_length', 'model']:
        if col in successful.columns:
            agg_dict[col] = 'first'
    
    df_agg = grouped.agg(agg_dict)
    
    # Flatten column names (e.g., 'requests_per_sec_mean', 'requests_per_sec_min', etc.)
    df_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in df_agg.columns.values]
    
    # Rename tp and pp columns
    df_agg = df_agg.rename(columns={'tp_': 'tp', 'pp_': 'pp'})
    
    # Sort by tp, then pp
    df_agg = df_agg.sort_values(['tp', 'pp'])
    
    # Create labels for x-axis
    df_agg['config'] = df_agg.apply(lambda x: f'TP{x["tp"]}\nPP{x["pp"]}', axis=1)
    
    print(f"Aggregated to {len(df_agg)} unique TP/PP configurations")
    for _, row in df_agg.iterrows():
        tp, pp = row['tp'], row['pp']
        count = len(successful[(successful['tp'] == tp) & (successful['pp'] == pp)])
        print(f"  TP{tp}/PP{pp}: {count} run(s)")
    
    df = df_agg

    if len(df) == 0:
        print("No successful experiments found in the CSV.")
        return

    print(f"Found {len(df)} unique TP/PP configurations:")
    # Use aggregated column names if available, otherwise use original
    req_col = 'requests_per_sec_mean' if 'requests_per_sec_mean' in df.columns else 'requests_per_sec'
    total_col = 'total_tokens_per_sec_mean' if 'total_tokens_per_sec_mean' in df.columns else 'total_tokens_per_sec'
    print(df[['tp', 'pp', req_col, total_col]].to_string(index=False))

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
