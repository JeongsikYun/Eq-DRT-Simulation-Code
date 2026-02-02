import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import glob
from datetime import datetime
import json
import argparse
import numpy as np
from scipy import stats as scipy_stats

"""
analyze_system_performance.py
============================
Analyzes DARP simulation sweep results and generates tradeoff visualizations.
Assumes objective function: OFV = (1 - alpha)*Mean_Delay + alpha*Tail_Delay,
where alpha=0.0 is mean-only and alpha=1.0 is CVaR-only.

Input:
  - Sweep directory containing 1_all_simulations_summary.csv (and optionally
    detailed_results/ for CVaR, occupancy, computation times).
  - Use --dir to specify a sweep path; otherwise uses a fixed default path.

Outputs (selected via interactive menu):
  2  : Combined 2x2 tradeoff grid (Gini, Tail, Detour, VKT vs Avg Delay).
  3  : Gini vs Avg Delay tradeoff (full + filtered demand levels).
  4  : Tail Delay vs Avg Delay tradeoff.
  5  : Detour Factor vs Avg Delay tradeoff.
  6  : VKT vs Avg Delay tradeoff.
  7-1: Demand-specific plots (average over seeds): Gini & CVaR vs Alpha per demand.
  7-2: Demand-specific plots (per seed): same metrics by seed.
  8  : Occupancy heatmap (Demand x Alpha).
  9  : Effectiveness analysis vs alpha=0.0 (efficiency loss vs equity/risk improvement).
  10 : CVaR 30% vs Avg Delay Pareto frontier.
  11 : Demand-specific CVaR 30% vs Alpha plots.
  12 : Mean ± 95% CI statistics (Avg Delay, Gini, CVaR, Runtime per 1-min opt) by Demand and Alpha;
       prints to console and writes CSV per demand.

Alpha=1.0 is excluded from all plots (CVaR-only scenario not plotted).
Usage: python analyze_system_performance.py [--dir path/to/sweep_YYYYMMDD_HHMMSS]
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_latest_sweep_dir(base_dir=None):
    """Return the latest sweep directory under base_dir (default: BASE_DIR/Alpha_Demand_Sweep_Results)."""
    search_base = base_dir if base_dir is not None else os.path.join(BASE_DIR, "Alpha_Demand_Sweep_Results")
    if not os.path.isdir(search_base):
        print(f"Warning: directory not found: '{search_base}'")
        return None
    sweep_dirs = glob.glob(os.path.join(search_base, "sweep_*"))
    if not sweep_dirs:
        print(f"Warning: no sweep_* directory found under '{search_base}'")
        return None
    return max(sweep_dirs, key=lambda p: os.path.basename(p))

def extract_multiplier(demand_str):
    """Extract numeric multiplier from strings like 'demand_1.5x'."""
    if isinstance(demand_str, str):
        match = re.search(r'(\d+\.?\d*)x', demand_str)
        if match:
            return float(match.group(1))
    return None

def _mean_ci_95(series):
    """Return 'Mean ± 95%% CI' string (t-distribution). For n<2 returns 'mean (n=1)'."""
    arr = np.asarray(series.dropna())
    n = len(arr)
    if n == 0:
        return "—", np.nan, np.nan
    mean_val = float(np.mean(arr))
    if n < 2:
        return f"{mean_val:.4g} (n=1)", mean_val, np.nan
    sem = float(np.std(arr, ddof=1)) / (n ** 0.5)
    t_val = scipy_stats.t.ppf(0.975, n - 1)
    half = t_val * sem
    return f"{mean_val:.4g} ± {half:.4g}", mean_val, half

def _load_runtime_per_opt_by_alpha_demand(sweep_dir):
    """Load per (Alpha, Demand) the list of mean computation time per 1-min optimization (per Seed, Scenario) from computation_times_*.csv. Returns dict[(alpha, demand)] -> list of floats."""
    detailed_dir = os.path.join(sweep_dir, 'detailed_results')
    if not os.path.exists(detailed_dir):
        return {}
    out = {}
    for alpha_dir in glob.glob(os.path.join(detailed_dir, 'alpha_*')):
        am = re.search(r'alpha_([0-9\.]+)', alpha_dir)
        if not am:
            continue
        alpha = float(am.group(1))
        for csv_path in glob.glob(os.path.join(alpha_dir, 'computation_times_demand_*.csv')):
            dm = re.search(r'computation_times_(demand_[0-9\.]+x)\.csv', csv_path.replace('\\', '/'))
            if not dm:
                continue
            demand = dm.group(1)
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if df.empty or 'ComputationTime' not in df.columns:
                continue
            if 'Alpha' in df.columns:
                df = df[df['Alpha'] == alpha].copy()
            if df.empty:
                continue
            id_cols = [c for c in ['Seed', 'Scenario'] if c in df.columns]
            if not id_cols:
                run_means = [df['ComputationTime'].mean()]
            else:
                run_means = df.groupby(id_cols)['ComputationTime'].mean().tolist()
            key = (alpha, demand)
            out[key] = run_means
    return out

def print_stats_by_demand_alpha(sweep_dir, full_df):
    """Print Mean ± 95%% CI per demand and alpha for: Avg_Delay, Gini_Delay, Tail_Delay, Runtime per 1-min opt."""
    column_mapping = {
        'AvgDelay': 'Avg_Delay',
        'TailMeanDelay': 'Tail_Delay',
        'GiniDelay': 'Gini_Delay',
    }
    df = full_df.copy()
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    df = df.dropna(subset=['Alpha']).copy()
    if df.empty:
        print("No rows with Alpha; skipping stats.")
        return
    demands = sorted(df['Demand'].unique(), key=lambda d: (extract_multiplier(d) or 0))
    runtime_by_key = _load_runtime_per_opt_by_alpha_demand(sweep_dir)

    for demand in demands:
        sub = df[df['Demand'] == demand]
        alphas = sorted(sub['Alpha'].unique())
        print(f"\n{'='*80}")
        print(f"Demand: {demand}")
        print(f"{'='*80}")
        rows_out = []
        for alpha in alphas:
            grp = sub[sub['Alpha'] == alpha]
            ad_str, ad_mean, ad_ci = _mean_ci_95(grp['Avg_Delay'])
            gi_str, gi_mean, gi_ci = _mean_ci_95(grp['Gini_Delay'])
            td_str, td_mean, td_ci = _mean_ci_95(grp['Tail_Delay'])
            key = (alpha, demand)
            run_list = runtime_by_key.get(key, [])
            if run_list:
                rt_str, rt_mean, rt_ci = _mean_ci_95(pd.Series(run_list))
            else:
                rt_str, rt_mean, rt_ci = "—", np.nan, np.nan
            row = {
                'Alpha': alpha,
                'Avg_Delay': ad_str,
                'Gini_Delay': gi_str,
                'Tail_Delay (CVaR)': td_str,
                'Runtime_per_1min_opt (s)': rt_str,
            }
            rows_out.append(row)
            print(f"  Alpha {alpha:.2f}  |  Avg Delay: {ad_str}  |  Gini: {gi_str}  |  Tail Delay: {td_str}  |  Runtime/1min opt: {rt_str}")
        out_csv = os.path.join(sweep_dir, f"stats_Mean_CI95_by_Alpha_{demand}.csv")
        pd.DataFrame(rows_out).to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"  Saved: {out_csv}")
    print()

def calculate_cvar_from_excel(sweep_dir, cvar_percentile=0.3):
    """Compute CVaR from passenger delay columns in detailed_results Excel files. Returns DataFrame with Alpha, Demand, Seed, CVaR_30."""
    detailed_dir = os.path.join(sweep_dir, 'detailed_results')
    if not os.path.exists(detailed_dir):
        print(f"Warning: {detailed_dir} not found.")
        return pd.DataFrame()
    cvar_results = []
    alpha_folders = glob.glob(os.path.join(detailed_dir, 'alpha_*'))
    for alpha_folder in alpha_folders:
        alpha_match = re.search(r'alpha_([0-9\.]+)', alpha_folder)
        if not alpha_match:
            continue
        alpha = float(alpha_match.group(1))
        
        excel_files_s1 = glob.glob(os.path.join(alpha_folder, 'results_s1_demand_*.xlsx'))
        excel_files_s2 = glob.glob(os.path.join(alpha_folder, 'results_s2_demand_*.xlsx'))
        excel_files = excel_files_s1 + excel_files_s2
        
        for excel_file in excel_files:
            demand_match = re.search(r'demand_([0-9\.]+)x', excel_file)
            if not demand_match:
                continue
            demand_str = f"demand_{demand_match.group(1)}x"
            
            try:
                xl_file = pd.ExcelFile(excel_file)
                possible_sheets = ['passenger_results', 'Passenger_Results', 'passengers', 'Passengers', 
                                 'results', 'Results', xl_file.sheet_names[0]]
                
                df = None
                for sheet_name in possible_sheets:
                    if sheet_name in xl_file.sheet_names:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        break
                
                if df is None or df.empty:
                    continue
                
                delay_col = None
                possible_delay_cols = ['Delay', 'delay', 'TotalDelay', 'Total_Delay', 'total_delay',
                                      'WaitTime', 'wait_time', 'WaitingTime', 'waiting_time']
                
                for col in possible_delay_cols:
                    if col in df.columns:
                        delay_col = col
                        break
                
                if delay_col is None:
                    print(f"Warning: delay column not found in {excel_file}. Columns: {df.columns.tolist()[:10]}")
                    continue
                
                if 'Seed' in df.columns or 'seed' in df.columns:
                    seed_col = 'Seed' if 'Seed' in df.columns else 'seed'
                    seeds = df[seed_col].unique()
                else:
                    seeds = [0]
                    df['Seed'] = 0
                    seed_col = 'Seed'
                
                for seed in seeds:
                    seed_data = df[df[seed_col] == seed]
                    delays = seed_data[delay_col].dropna().values
                    
                    if len(delays) == 0:
                        continue
                    
                    threshold_idx = int(len(delays) * (1 - cvar_percentile))
                    sorted_delays = np.sort(delays)
                    cvar = np.mean(sorted_delays[threshold_idx:])
                    
                    cvar_results.append({
                        'Alpha': alpha,
                        'Demand': demand_str,
                        'Seed': seed,
                        'CVaR_30': cvar
                    })
                    
            except Exception as e:
                print(f"Warning: error processing {excel_file}: {e}")
                continue
    
    if cvar_results:
        print(f"CVaR computed: {len(cvar_results)} simulations")
        return pd.DataFrame(cvar_results)
    print("Warning: could not compute CVaR data.")
    return pd.DataFrame()

def plot_combined_tradeoffs(df, output_dir):
    """Plot Gini, Tail Delay, Detour, VKT tradeoffs in a 2x2 grid."""
    
    combined_df = df[df['Alpha'] < 1.0].copy()
    if combined_df.empty:
        print("Warning: no simulation data for combined tradeoff plot.")
        return
        
    column_mapping = {
        'AvgDelay': 'Avg_Delay',
        'TailMeanDelay': 'Tail_Delay', 
        'GiniDelay': 'Gini_Delay',
        'AvgDetourFactor': 'Avg_Detour',
        'TotalVKT': 'Total_VKT'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in combined_df.columns and new_col not in combined_df.columns:
            combined_df[new_col] = combined_df[old_col]

    combined_agg = combined_df.groupby(['Alpha', 'Demand']).agg(
        Avg_Delay=('Avg_Delay', 'mean'),
        Gini_Delay=('Gini_Delay', 'mean'),
        Tail_Delay=('Tail_Delay', 'mean'),
        Avg_Detour=('Avg_Detour', 'mean'),
        Total_VKT=('Total_VKT', 'mean')
    ).reset_index()
    
    combined_agg['Multiplier'] = combined_agg['Demand'].apply(extract_multiplier)
    combined_agg.dropna(subset=['Multiplier'], inplace=True)
    combined_agg = combined_agg.sort_values(['Alpha', 'Multiplier'])

    combined_agg.rename(columns={'Multiplier': 'Delta', 'Alpha': 'Alpha (α)'}, inplace=True)

    unique_demands = sorted(combined_agg['Delta'].unique())
    palette = sns.color_palette("plasma", n_colors=len(unique_demands))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, axes = plt.subplots(2, 2, figsize=(28, 20))

    plot_configs = [
        {'ax': axes[0, 0], 'y': 'Gini_Delay', 'title': 'Efficiency vs. Equity', 'ylabel': 'Delay Gini Coefficient (Equity)'},
        {'ax': axes[0, 1], 'y': 'Tail_Delay', 'title': 'Efficiency vs. Risk', 'ylabel': 'Average Tail Delay (Risk)'},
        {'ax': axes[1, 0], 'y': 'Avg_Detour', 'title': 'Efficiency vs. Service Quality', 'ylabel': 'Average Detour Factor (QoS)'},
        {'ax': axes[1, 1], 'y': 'Total_VKT', 'title': 'Efficiency vs. Operational Cost', 'ylabel': 'Total VKT (Cost)'}
    ]
    
    for i, config in enumerate(plot_configs):
        ax = config['ax']
        legend_flag = 'full' if i == 0 else False
        sns.scatterplot(data=combined_agg, x='Avg_Delay', y=config['y'], hue='Delta',
                       palette=palette, size='Alpha (α)', sizes=(80, 800),
                       marker='o', alpha=0.8, ax=ax, legend=legend_flag)
        for multiplier in unique_demands:
            subset = combined_agg[combined_agg['Delta'] == multiplier].sort_values('Alpha (α)')
            if len(subset) > 1:
                ax.plot(subset['Avg_Delay'], subset[config['y']],
                         color=palette[unique_demands.index(multiplier)], linestyle='-', linewidth=3, alpha=0.7)
        ax.set_ylabel(config['ylabel'], fontsize=24)
        ax.set_xlabel('Average Delay (Efficiency, lower is better)', fontsize=24)
        ax.grid(True, which='both', linestyle='--', linewidth=1)
        ax.tick_params(axis='both', which='major', labelsize=20)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

    fig.legend(handles, labels, bbox_to_anchor=(0.88, 0.5), loc='center left', fontsize=24, labelspacing=2.0)
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.85, top=0.98)
    output_filename = os.path.join(output_dir, f'2_combined_tradeoffs_{timestamp}.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_filename}")
    plt.close(fig)

def plot_tradeoffs(df, output_dir, selected_plots=None):
    """Generate Gini, Tail Delay, Detour, VKT tradeoff plots."""
    if selected_plots is None:
        selected_plots = [3, 4, 5, 6]
    combined_df = df[df['Alpha'] < 1.0].copy()
    if combined_df.empty:
        print("Warning: no data for tradeoff plots.")
        return
        
    column_mapping = {
        'AvgDelay': 'Avg_Delay',
        'TailMeanDelay': 'Tail_Delay', 
        'GiniDelay': 'Gini_Delay',
        'AvgDetourFactor': 'Avg_Detour',
        'TotalVKT': 'Total_VKT'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in combined_df.columns and new_col not in combined_df.columns:
            combined_df[new_col] = combined_df[old_col]

    combined_agg = combined_df.groupby(['Alpha', 'Demand']).agg(
        Avg_Delay=('Avg_Delay', 'mean'),
        Gini_Delay=('Gini_Delay', 'mean'),
        Tail_Delay=('Tail_Delay', 'mean'),
        Avg_Detour=('Avg_Detour', 'mean'),
        Total_VKT=('Total_VKT', 'mean')
    ).reset_index()
    
    combined_agg['Multiplier'] = combined_agg['Demand'].apply(extract_multiplier)
    combined_agg.dropna(subset=['Multiplier'], inplace=True)
    combined_agg = combined_agg.sort_values(['Alpha', 'Multiplier'])

    combined_agg.rename(columns={'Multiplier': 'Delta'}, inplace=True)
    unique_demands = sorted(combined_agg['Delta'].unique())
    palette = sns.color_palette("plasma", n_colors=len(unique_demands))
    demand_color_map = dict(zip(unique_demands, palette))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if 3 in selected_plots:
        fig, ax = plt.subplots(figsize=(32, 16))
        scatter = sns.scatterplot(data=combined_agg, x='Avg_Delay', y='Gini_Delay', hue='Delta',
                       palette=demand_color_map, size='Alpha', sizes=(150, 1200),
                       marker='o', alpha=0.8, ax=ax)
        for multiplier in unique_demands:
            subset = combined_agg[combined_agg['Delta'] == multiplier].sort_values('Alpha')
            if len(subset) > 1:
                ax.plot(subset['Avg_Delay'], subset['Gini_Delay'],
                         color=demand_color_map[multiplier], linestyle='-', linewidth=3, alpha=0.7)
        ax.set_xlabel('Average delay (min) (efficiency, lower is better)', fontsize=32)
        ax.set_ylabel('Gini coefficient of delay (equity, lower is better)', fontsize=32)
        handles, labels = ax.get_legend_handles_labels()
        combined_handles = []
        combined_labels = []
        fixed_demand_order = [0.5, 0.75, 1.0, 1.25, 1.5]
        fixed_markers = ['o', 's', '^', 'D', 'X']
        fixed_colors = sns.color_palette("viridis", n_colors=5)
        
        combined_labels.append('Demand rate\n(requests/min)')
        combined_handles.append(plt.Rectangle((0,0),1,1, fill=False, edgecolor='none', visible=False))
        for i, demand_val in enumerate(fixed_demand_order):
            if demand_val in unique_demands:
                handle = plt.scatter([], [], marker=fixed_markers[i], s=300, 
                                   color=fixed_colors[i], alpha=0.8)
                combined_handles.append(handle)
                combined_labels.append(f'{demand_val:.2f}')
        
        combined_labels.append('Alpha')
        combined_handles.append(plt.Rectangle((0,0),1,1, fill=False, edgecolor='none', visible=False))
        alpha_values = sorted(combined_agg['Alpha'].unique())
        for alpha_val in alpha_values:
            size = 150 + (1200-150) * (alpha_val - min(alpha_values)) / (max(alpha_values) - min(alpha_values)) if len(alpha_values) > 1 else 675
            handle = plt.scatter([], [], s=size, color='gray', alpha=0.6)
            combined_handles.append(handle)
            combined_labels.append(f'α={alpha_val:.1f}')
        
        combined_legend = ax.legend(combined_handles, combined_labels,
                                   bbox_to_anchor=(1.015, 1), loc='upper left',
                                   fontsize=26, labelspacing=1.3, handletextpad=0.8)
        for text in combined_legend.get_texts():
            if 'Demand rate' in text.get_text() or 'Alpha' in text.get_text():
                text.set_horizontalalignment('left')
                text.set_x(0)
        ax.grid(True, which='both', linestyle='--', linewidth=1)
        ax.tick_params(axis='both', which='major', labelsize=28)
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.82, top=0.98)
        output_filename_gini = os.path.join(output_dir, f'3_tradeoff_plot_Gini_{timestamp}.png')
        plt.savefig(output_filename_gini, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_filename_gini}")
        plt.close(fig)

        filtered_demand_levels = [0.5, 0.75, 1.0, 1.25, 1.5]
        filtered_df = combined_agg[combined_agg['Delta'].isin(filtered_demand_levels)]
        actual_demand_levels = sorted(filtered_df['Delta'].unique())
        print(f"Filtered demand levels: {actual_demand_levels}")

        if not filtered_df.empty:
            fixed_demand_levels = [0.5, 0.75, 1.0, 1.25, 1.5]
            fixed_markers = ['o', 's', '^', 'D', 'X']
            fixed_colors = sns.color_palette("viridis", n_colors=5)
            
            markers_map = {level: marker for level, marker in zip(fixed_demand_levels, fixed_markers)}
            palette_map = {level: color for level, color in zip(fixed_demand_levels, fixed_colors)}

            fig, ax = plt.subplots(figsize=(32, 18))
            scatter = sns.scatterplot(data=filtered_df,
                           x='Avg_Delay', y='Gini_Delay',
                           hue='Delta', style='Delta',
                           markers=markers_map, palette=palette_map,
                           size='Alpha', sizes=(150, 1200), alpha=0.8, ax=ax)
            ax.set_xlabel('Average delay (min)', fontsize=48)
            ax.set_ylabel('Gini coefficient of delay', fontsize=48)
            handles, labels = ax.get_legend_handles_labels()
            combined_handles_filtered = []
            combined_labels_filtered = []
            combined_labels_filtered.append('Demand rate\n(requests/min)')
            combined_handles_filtered.append(plt.Rectangle((0,0),1,1, fill=False, edgecolor='none', visible=False))
            for i, demand_val in enumerate(fixed_demand_levels):
                if demand_val in actual_demand_levels:
                    handle = plt.scatter([], [], marker=fixed_markers[i], s=300,
                                       color=fixed_colors[i], alpha=0.8)
                    combined_handles_filtered.append(handle)
                    combined_labels_filtered.append(f'{demand_val:.2f}')
            combined_labels_filtered.append('Alpha')
            combined_handles_filtered.append(plt.Rectangle((0,0),1,1, fill=False, edgecolor='none', visible=False))
            
            alpha_values = sorted(filtered_df['Alpha'].unique())
            for alpha_val in alpha_values:
                size = 150 + (1200-150) * (alpha_val - min(alpha_values)) / (max(alpha_values) - min(alpha_values)) if len(alpha_values) > 1 else 675
                handle = plt.scatter([], [], s=size, color='gray', alpha=0.6)
                combined_handles_filtered.append(handle)
                combined_labels_filtered.append(f'α={alpha_val:.1f}')
            
            combined_legend_filtered = ax.legend(combined_handles_filtered, combined_labels_filtered,
                                               bbox_to_anchor=(1.015, 1), loc='upper left',
                                               fontsize=39, labelspacing=0.8, handletextpad=0.8)
            for text in combined_legend_filtered.get_texts():
                if 'Demand rate' in text.get_text() or 'Alpha' in text.get_text():
                    text.set_horizontalalignment('left')
                    text.set_x(0)
            
            ax.grid(True, which='both', linestyle='--', linewidth=1)
            ax.tick_params(axis='both', which='major', labelsize=42)
            plt.subplots_adjust(left=0.08, bottom=0.08, right=0.82, top=0.98)
            output_filename_gini_filtered = os.path.join(output_dir, f'3_tradeoff_plot_Gini_filtered_{timestamp}.png')
            plt.savefig(output_filename_gini_filtered, dpi=300, bbox_inches='tight')
            print(f"Filtered Gini tradeoff plot saved: '{output_filename_gini_filtered}'")
            plt.close(fig)
        else:
            print("Warning: no data for filtered demand levels; skipping filtered Gini plot.")

    if 4 in selected_plots:
        plt.figure(figsize=(18, 14))
        sns.scatterplot(data=combined_agg, x='Avg_Delay', y='Tail_Delay', hue='Delta',
                       palette=demand_color_map, size='Alpha', sizes=(80, 800),                       marker='o', alpha=0.8, legend='full')
        for multiplier in unique_demands:
            subset = combined_agg[combined_agg['Delta'] == multiplier].sort_values('Alpha')
            if len(subset) > 1:
                plt.plot(subset['Avg_Delay'], subset['Tail_Delay'],
                         color=demand_color_map[multiplier], linestyle='-', linewidth=3, alpha=0.7)
        plt.xlabel('Average Delay (Efficiency, lower is better)', fontsize=28)
        plt.ylabel('Average Tail Delay (Risk, lower is better)', fontsize=28)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20, labelspacing=2.0)
        plt.grid(True, which='both', linestyle='--', linewidth=1)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.subplots_adjust(left=0.08, bottom=0.1, right=0.82, top=0.98)
        output_filename_tail = os.path.join(output_dir, f'4_tradeoff_plot_TailDelay_{timestamp}.png')
        plt.savefig(output_filename_tail, dpi=300, bbox_inches='tight')
        print(f"Tail Delay tradeoff plot saved: '{output_filename_tail}'")
        plt.close()
        
    if 5 in selected_plots:
        plt.figure(figsize=(18, 14))
        sns.scatterplot(data=combined_agg, x='Avg_Delay', y='Avg_Detour', hue='Delta',
                       palette=demand_color_map, size='Alpha', sizes=(80, 800),                       marker='o', alpha=0.8, legend='full')
        for multiplier in unique_demands:
            subset = combined_agg[combined_agg['Delta'] == multiplier].sort_values('Alpha')
            if len(subset) > 1:
                plt.plot(subset['Avg_Delay'], subset['Avg_Detour'],
                         color=demand_color_map[multiplier], linestyle='-', linewidth=3, alpha=0.7)
        plt.xlabel('Average Delay (Efficiency, lower is better)', fontsize=28)
        plt.ylabel('Average Detour Factor (QoS, lower is better)', fontsize=28)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20, labelspacing=2.0)
        plt.grid(True, which='both', linestyle='--', linewidth=1)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.subplots_adjust(left=0.08, bottom=0.1, right=0.82, top=0.98)
        output_filename_detour = os.path.join(output_dir, f'5_tradeoff_plot_Detour_{timestamp}.png')
        plt.savefig(output_filename_detour, dpi=300, bbox_inches='tight')
        print(f"Detour tradeoff plot saved: '{output_filename_detour}'")
        plt.close()

    if 6 in selected_plots:
        plt.figure(figsize=(18, 14))
        sns.scatterplot(data=combined_agg, x='Avg_Delay', y='Total_VKT', hue='Delta',
                       palette=demand_color_map, size='Alpha', sizes=(80, 800),                       marker='o', alpha=0.8, legend='full')
        for multiplier in unique_demands:
            subset = combined_agg[combined_agg['Delta'] == multiplier].sort_values('Alpha')
            if len(subset) > 1:
                plt.plot(subset['Avg_Delay'], subset['Total_VKT'],
                         color=demand_color_map[multiplier], linestyle='-', linewidth=3, alpha=0.7)
        plt.xlabel('Average Delay (Efficiency, lower is better)', fontsize=28)
        plt.ylabel('Total Vehicle Kilometers Traveled (Cost, lower is better)', fontsize=28)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20, labelspacing=2.0)
        plt.grid(True, which='both', linestyle='--', linewidth=1)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.subplots_adjust(left=0.08, bottom=0.1, right=0.82, top=0.98)
        output_filename_vkt = os.path.join(output_dir, f'6_tradeoff_plot_VKT_{timestamp}.png')
        plt.savefig(output_filename_vkt, dpi=300, bbox_inches='tight')
        print(f"Saved: '{output_filename_vkt}'")
        plt.close()

def plot_demand_specific_graphs(df, output_dir, graph_type='both'):
    """Plot Gini/Tail/Detour/VKT vs Alpha per demand level. graph_type: 'average', 'individual', or 'both'."""
    df_with_scenario = df.copy()
    df_with_scenario['Scenario'] = df_with_scenario['Alpha'].apply(lambda x: 'S1' if x == 1.0 else 'S2')
    combined_df = df_with_scenario[df_with_scenario['Alpha'] < 1.0].copy()
    if combined_df.empty:
        print("Warning: no data for demand-specific plots.")
        return

    column_mapping = {
        'AvgDelay': 'Avg_Delay',
        'TailMeanDelay': 'Tail_Delay', 
        'GiniDelay': 'Gini_Delay',
        'AvgDetourFactor': 'Avg_Detour',
        'TotalVKT': 'Total_VKT'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in combined_df.columns and new_col not in combined_df.columns:
            combined_df[new_col] = combined_df[old_col]

    combined_agg = combined_df.groupby(['Alpha', 'Demand']).agg(
        Avg_Delay=('Avg_Delay', 'mean'),
        Gini_Delay=('Gini_Delay', 'mean'),
        Tail_Delay=('Tail_Delay', 'mean'),
        Avg_Detour=('Avg_Detour', 'mean'),
        Total_VKT=('Total_VKT', 'mean')
    ).reset_index()
    
    base_plot_folder = os.path.join(output_dir, "7_demand_specific_plots")
    
    if graph_type == 'average':
        plot_folder = os.path.join(base_plot_folder, "average")
    elif graph_type == 'individual':
        plot_folder = os.path.join(base_plot_folder, "individual")
    else:  # graph_type == 'both'
        plot_folder = base_plot_folder
    
    os.makedirs(plot_folder, exist_ok=True)
    print(f"Demand-specific plot folder: '{plot_folder}'")

    unique_demands = combined_df['Demand'].unique()
    unique_seeds = combined_df['Seed'].unique()

    total_graphs = 0
    
    if graph_type in ['average', 'both']:
        if graph_type == 'both':
            avg_plot_folder = os.path.join(base_plot_folder, "average")
            os.makedirs(avg_plot_folder, exist_ok=True)
        else:
            avg_plot_folder = plot_folder
            
        for demand in unique_demands:
            demand_agg = combined_agg[combined_agg['Demand'] == demand].sort_values('Alpha')
            demand_str = demand.replace('.', 'p')
            if demand_agg.empty:
                continue
            print(f"\n{'='*80}")
            print(f"Demand: {demand}")
            print(f"{'='*80}")
            print(f"{'Alpha':>8} | {'Gini Coefficient':>18} | {'Average Delay (min)':>20}")
            print(f"{'-'*80}")
            for _, row in demand_agg.iterrows():
                print(f"{row['Alpha']:>8.1f} | {row['Gini_Delay']:>18.6f} | {row['Avg_Delay']:>20.3f}")
            print(f"{'='*80}\n")

            fig, ax1 = plt.subplots(figsize=(16, 10))
            color1 = 'tab:blue'
            ax1.set_xlabel('Alpha (α)', fontsize=28)
            ax1.set_ylabel('Gini coefficient of delay', color=color1, fontsize=28)
            ax1.plot(demand_agg['Alpha'], demand_agg['Gini_Delay'], color=color1, marker='o', markersize=12, linestyle='-', label='Gini coefficient of delay', linewidth=3)
            ax1.tick_params(axis='y', labelcolor=color1, labelsize=24)
            ax1.tick_params(axis='x', labelsize=24)
            ax1.xaxis.grid(True, which='both', linestyle='--', linewidth=1)
            ax1.yaxis.grid(False)
            ax2 = ax1.twinx()
            color2 = 'tab:green'
            ax2.set_ylabel('Average delay (min)', color=color2, fontsize=28)
            ax2.plot(demand_agg['Alpha'], demand_agg['Avg_Delay'], color=color2, marker='s', markersize=12, linestyle='--', label='Average delay', linewidth=3)
            ax2.tick_params(axis='y', labelcolor=color2, labelsize=24)
            ax2.yaxis.grid(False)

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=20)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
            output_filename_1 = os.path.join(avg_plot_folder, f'demand_{demand_str}_gini_vs_avg.png')
            plt.savefig(output_filename_1, dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax1 = plt.subplots(figsize=(16, 10))
            color1 = 'tab:red'
            ax1.set_xlabel('Alpha (α)', fontsize=28)
            ax1.set_ylabel('CVaR of delay (min)', color=color1, fontsize=28)
            ax1.plot(demand_agg['Alpha'], demand_agg['Tail_Delay'], color=color1, marker='o', markersize=12, linestyle='-', label='CVaR of delay', linewidth=3)
            ax1.tick_params(axis='y', labelcolor=color1, labelsize=24)
            ax1.tick_params(axis='x', labelsize=24)
            ax1.xaxis.grid(True, which='both', linestyle='--', linewidth=1)
            ax1.yaxis.grid(False)
            ax2 = ax1.twinx()
            color2 = 'tab:green'
            ax2.set_ylabel('Average delay (min)', color=color2, fontsize=28)
            ax2.plot(demand_agg['Alpha'], demand_agg['Avg_Delay'], color=color2, marker='s', markersize=12, linestyle='--', label='Average delay', linewidth=3)
            ax2.tick_params(axis='y', labelcolor=color2, labelsize=24)
            ax2.yaxis.grid(False)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=20)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
            output_filename_2 = os.path.join(avg_plot_folder, f'demand_{demand_str}_cvar_vs_avg.png')
            plt.savefig(output_filename_2, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            total_graphs += 2

    if graph_type in ['individual', 'both']:
        if graph_type == 'both':
            ind_plot_folder = os.path.join(base_plot_folder, "individual")
            os.makedirs(ind_plot_folder, exist_ok=True)
        else:
            ind_plot_folder = plot_folder
            
        for demand in unique_demands:
            demand_df = combined_df[combined_df['Demand'] == demand]
            demand_str = demand.replace('.', 'p')
            if demand_df.empty:
                continue

            for seed in unique_seeds:
                seed_data = demand_df[demand_df['Seed'] == seed].sort_values('Alpha')
                
                if seed_data.empty:
                    continue

                fig, ax1 = plt.subplots(figsize=(16, 10))
                color1 = 'tab:blue'
                ax1.set_xlabel('Alpha (α)', fontsize=28)
                ax1.set_ylabel('Gini coefficient of delay', color=color1, fontsize=28)
                gini_col = 'Gini_Delay' if 'Gini_Delay' in seed_data.columns else 'Gini Coefficient of Delay'
                ax1.plot(seed_data['Alpha'], seed_data[gini_col], color=color1, marker='o', markersize=12, linestyle='-', label='Gini coefficient of delay', linewidth=3)
                ax1.tick_params(axis='y', labelcolor=color1, labelsize=24)
                ax1.tick_params(axis='x', labelsize=24)
                ax1.xaxis.grid(True, which='both', linestyle='--', linewidth=1)
                ax1.yaxis.grid(False)

                ax2 = ax1.twinx()
                color2 = 'tab:green'
                ax2.set_ylabel('Average delay (min)', color=color2, fontsize=28)
                ax2.plot(seed_data['Alpha'], seed_data['Avg_Delay'], color=color2, marker='s', markersize=12, linestyle='--', label='Average delay', linewidth=3)
                ax2.tick_params(axis='y', labelcolor=color2, labelsize=24)
                ax2.yaxis.grid(False)
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=20)
                
                plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
                output_filename_1 = os.path.join(ind_plot_folder, f'demand_{demand_str}_seed_{seed}_gini_vs_avg.png')
                plt.savefig(output_filename_1, dpi=300, bbox_inches='tight')
                plt.close(fig)

                fig, ax1 = plt.subplots(figsize=(16, 10))
                color1 = 'tab:red'
                ax1.set_xlabel('Alpha (α)', fontsize=28)
                ax1.set_ylabel('CVaR of delay (min)', color=color1, fontsize=28)
                tail_col = 'Tail_Delay' if 'Tail_Delay' in seed_data.columns else 'Tail Average Delay'
                ax1.plot(seed_data['Alpha'], seed_data[tail_col], color=color1, marker='o', markersize=12, linestyle='-', label='CVaR of delay', linewidth=3)
                ax1.tick_params(axis='y', labelcolor=color1, labelsize=24)
                ax1.tick_params(axis='x', labelsize=24)
                ax1.xaxis.grid(True, which='both', linestyle='--', linewidth=1)
                ax1.yaxis.grid(False)

                ax2 = ax1.twinx()
                color2 = 'tab:green'
                ax2.set_ylabel('Average delay (min)', color=color2, fontsize=28)
                avg_col = 'Avg_Delay' if 'Avg_Delay' in seed_data.columns else 'Average Delay'
                ax2.plot(seed_data['Alpha'], seed_data[avg_col], color=color2, marker='s', markersize=12, linestyle='--', label='Average delay', linewidth=3)
                ax2.tick_params(axis='y', labelcolor=color2, labelsize=24)
                ax2.yaxis.grid(False)
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=20)
                
                plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
                output_filename_2 = os.path.join(ind_plot_folder, f'demand_{demand_str}_seed_{seed}_cvar_vs_avg.png')
                plt.savefig(output_filename_2, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                total_graphs += 2

    if graph_type == 'both':
        print(f"Demand-specific plots done: {total_graphs} plots.")
        print(f"- Average: {os.path.join(base_plot_folder, 'average')}")
        print(f"- Individual: {os.path.join(base_plot_folder, 'individual')}")
    else:
        print(f"Demand-specific plots done: {total_graphs} plots (type: {graph_type})")
        print(f"Output: {plot_folder}")

def plot_occupancy_analysis(sweep_dir, params):
    """Generate occupancy (vehicle load) analysis plots."""
    vehicle_capacity = params.get('vehicle_capacity', 6)
    detailed_dir = os.path.join(sweep_dir, 'detailed_results')
    vehicle_files = glob.glob(os.path.join(detailed_dir, 'alpha_*', 'vehicle_states_*.csv'))
    if not vehicle_files:
        print("Warning: no vehicle_states files found; skipping occupancy analysis.")
        return

    df_list = []
    for f in vehicle_files:
        try:
            match_alpha = re.search(r'alpha_([0-9\.]+)', f)
            match_demand = re.search(r'demand_([0-9\.]+)x\.csv', f)
            
            if not match_alpha or not match_demand:
                print(f"Warning: could not extract Alpha/Demand from path: {f}")
                continue

            alpha = float(match_alpha.group(1))
            demand_str = f"demand_{match_demand.group(1)}x"

            temp_df = pd.read_csv(f)
            if not temp_df.empty:
                temp_df['Alpha'] = alpha
                temp_df['Demand'] = demand_str
                temp_df['Scenario'] = temp_df['Alpha'].apply(lambda x: 'S1' if x == 1.0 else 'S2')
                df_list.append(temp_df)
        except Exception as e:
            print(f"Warning: error processing {f}: {e}")

    if not df_list:
        print("Warning: no valid vehicle state data; skipping occupancy analysis.")
        return
        
    df = pd.concat(df_list, ignore_index=True)
    df = df[df['Scenario'] == 'S2']
    if df.empty:
        print("Warning: no S2 (Alpha != 1.0) vehicle state data; skipping occupancy analysis.")
        return
    
    if 'CurrentLoad' in df.columns:
        df['Occupancy'] = df['CurrentLoad']
    else:
        print("Warning: CurrentLoad column not found; skipping occupancy analysis.")
        return

    df['OccupancyRate'] = df['Occupancy'] / vehicle_capacity

    plot_folder = os.path.join(sweep_dir, "8_occupancy_analysis")
    os.makedirs(plot_folder, exist_ok=True)
    print(f"Occupancy plot folder: '{plot_folder}'")

    heatmap_data = df.groupby(['Demand', 'Alpha'])['OccupancyRate'].mean().reset_index()
    if heatmap_data.empty:
        print("Warning: no grouped data for occupancy heatmap.")
        return
    heatmap_pivot = heatmap_data.pivot(index='Demand', columns='Alpha', values='OccupancyRate')
    if heatmap_pivot.empty:
        print("Warning: empty pivot for occupancy heatmap.")
        return
        
    heatmap_pivot.sort_index(key=lambda x: x.str.extract(r'(\d+\.?\d*)').astype(float).squeeze(), inplace=True)
    
    plt.figure(figsize=(20, 12))
    sns.heatmap(heatmap_pivot, annot=True, fmt=".2%", cmap="viridis", linewidths=1, annot_kws={'fontsize': 16})
    plt.xlabel('Alpha (α)', fontsize=28)
    plt.ylabel('Demand Level', fontsize=28)
    plt.xticks(rotation=45, fontsize=24)
    plt.yticks(rotation=0, fontsize=24)
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.98)
    output_filename = os.path.join(plot_folder, 'heatmap_avg_occupancy_rate.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Occupancy heatmap saved: '{output_filename}'")

def plot_effectiveness_analysis(df, output_dir):
    """Visualize efficiency loss vs equity/risk improvement relative to alpha=0.0 (Alpha as in data; 0.0=mean-only, 1.0=CVaR-only)."""
    combined_df = df[df['Alpha'] < 1.0].copy()
    
    column_mapping = {
        'AvgDelay': 'Avg_Delay',
        'TailMeanDelay': 'Tail_Delay', 
        'GiniDelay': 'Gini_Delay'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in combined_df.columns and new_col not in combined_df.columns:
            combined_df[new_col] = combined_df[old_col]

    combined_agg = combined_df.groupby(['Alpha', 'Demand']).agg(
        Avg_Delay=('Avg_Delay', 'mean'),
        Gini_Delay=('Gini_Delay', 'mean'),
        Tail_Delay=('Tail_Delay', 'mean')
    ).reset_index()
    
    baseline_df = combined_agg[combined_agg['Alpha'] == 0.0].copy()
    if baseline_df.empty:
        print("Warning: no Alpha=0.0 baseline; skipping effectiveness analysis.")
        return
    
    baseline_metrics = baseline_df.groupby(['Demand']).agg(
        Avg_Delay=('Avg_Delay', 'mean'),
        Gini_Delay=('Gini_Delay', 'mean'),
        Tail_Delay=('Tail_Delay', 'mean')
    ).reset_index().set_index('Demand')
    
    if baseline_metrics.empty:
        print("Warning: no Alpha=0.0 baseline; skipping effectiveness analysis.")
        return

    comparison_agg = combined_agg[combined_agg['Alpha'] != 0.0].copy()
    results_equity = []
    results_risk = []
    for _, row in comparison_agg.iterrows():
        demand = row['Demand']
        if demand in baseline_metrics.index:
            baseline = baseline_metrics.loc[demand]
            
            delay_change_pct = ((row['Avg_Delay'] / baseline['Avg_Delay']) - 1) * 100
            gini_change_pct = ((row['Gini_Delay'] / baseline['Gini_Delay']) - 1) * 100
            tail_change_pct = ((row['Tail_Delay'] / baseline['Tail_Delay']) - 1) * 100
            
            results_equity.append({
                'Demand': demand,
                'Alpha': row['Alpha'],
                'Baseline_Avg_Delay': baseline['Avg_Delay'],
                'Alpha_Avg_Delay': row['Avg_Delay'],
                'Delay_Change_Pct': delay_change_pct,
                'Baseline_Gini': baseline['Gini_Delay'],
                'Alpha_Gini': row['Gini_Delay'],
                'Gini_Change_Pct': gini_change_pct
            })
            results_risk.append({
                'Demand': demand,
                'Alpha': row['Alpha'],
                'Baseline_Avg_Delay': baseline['Avg_Delay'],
                'Alpha_Avg_Delay': row['Avg_Delay'],
                'Delay_Change_Pct': delay_change_pct,
                'Baseline_Tail_Delay': baseline['Tail_Delay'],
                'Alpha_Tail_Delay': row['Tail_Delay'],
                'Tail_Change_Pct': tail_change_pct
            })

    if not results_equity:
        print("Warning: could not build comparison data for effectiveness analysis.")
        return

    eff_df_equity = pd.DataFrame(results_equity)
    eff_df_risk = pd.DataFrame(results_risk)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = os.path.join(output_dir, f'9_effectiveness_analysis_data_{timestamp}.xlsx')
    with pd.ExcelWriter(excel_filename) as writer:
        eff_df_equity.to_excel(writer, sheet_name='Equity_Analysis', index=False)
        eff_df_risk.to_excel(writer, sheet_name='Risk_Analysis', index=False)
    print(f"Effectiveness analysis detail Excel saved: '{excel_filename}'")
    
    eff_df = eff_df_equity.copy()
    eff_df['Tail_Change_Pct'] = eff_df_risk['Tail_Change_Pct']
    eff_df['Multiplier'] = eff_df['Demand'].apply(extract_multiplier)
    eff_df.rename(columns={'Multiplier': 'Delta'}, inplace=True)
    
    unique_demands = sorted(eff_df['Delta'].unique())
    palette = sns.color_palette("viridis", n_colors=len(unique_demands))

    plot_folder = os.path.join(output_dir, "9_tradeoff_effectiveness")
    os.makedirs(plot_folder, exist_ok=True)
    plt.figure(figsize=(18, 14))
    sns.scatterplot(data=eff_df, x='Delay_Change_Pct', y='Gini_Change_Pct', hue='Delta',
                   palette=palette, size='Alpha', sizes=(80, 800), alpha=0.8, legend='full')
    plt.xlabel('Average Delay Change (%) (Negative is Better)', fontsize=28)
    plt.ylabel('Gini Coefficient Change (%) (Negative is Better)', fontsize=28)
    plt.axhline(0, color='grey', linestyle='--', linewidth=2)
    plt.axvline(0, color='grey', linestyle='--', linewidth=2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20, labelspacing=2.0)
    plt.grid(True, linewidth=1)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.82, top=0.98)
    plt.savefig(os.path.join(plot_folder, f'effectiveness_gini_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(18, 14))
    sns.scatterplot(data=eff_df, x='Delay_Change_Pct', y='Tail_Change_Pct', hue='Delta',
                   palette=palette, size='Alpha', sizes=(80, 800), alpha=0.8, legend='full')
    plt.xlabel('Average Delay Change (%) (Negative is Better)', fontsize=28)
    plt.ylabel('Tail Delay Change (%) (Negative is Better)', fontsize=28)
    plt.axhline(0, color='grey', linestyle='--', linewidth=2)
    plt.axvline(0, color='grey', linestyle='--', linewidth=2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20, labelspacing=2.0)
    plt.grid(True, linewidth=1)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.82, top=0.98)
    plt.savefig(os.path.join(plot_folder, f'effectiveness_tail_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Effectiveness plots saved: '{plot_folder}'")

def plot_cvar_tradeoffs(df, cvar_df, output_dir):
    """Plot CVaR 30% vs Average Delay Pareto frontier. df: summary with Avg_Delay; cvar_df: Alpha, Demand, Seed, CVaR_30; output_dir: output path."""
    if cvar_df.empty:
        print("Warning: no CVaR data; skipping CVaR tradeoff plot.")
        return
    
    df_for_merge = df[['Alpha', 'Demand', 'Seed', 'Avg_Delay']].copy()
    merged_df = pd.merge(cvar_df, df_for_merge, on=['Alpha', 'Demand', 'Seed'], how='inner')
    
    if merged_df.empty:
        print("Warning: could not merge CVaR and summary data.")
        return

    merged_df = merged_df[merged_df['Alpha'] < 1.0]
    if merged_df.empty:
        print("Warning: no Alpha < 1.0 data; skipping CVaR tradeoff plot.")
        return
    
    cvar_agg = merged_df.groupby(['Alpha', 'Demand']).agg(
        Avg_Delay=('Avg_Delay', 'mean'),
        CVaR_30=('CVaR_30', 'mean')
    ).reset_index()
    
    cvar_agg['Multiplier'] = cvar_agg['Demand'].apply(extract_multiplier)
    cvar_agg.dropna(subset=['Multiplier'], inplace=True)
    cvar_agg = cvar_agg.sort_values(['Alpha', 'Multiplier'])
    
    cvar_agg.rename(columns={'Multiplier': 'Delta'}, inplace=True)
    
    unique_demands = sorted(cvar_agg['Delta'].unique())
    
    filtered_demand_levels = [0.5, 0.75, 1.0, 1.25, 1.5]
    filtered_df = cvar_agg[cvar_agg['Delta'].isin(filtered_demand_levels)]
    actual_demand_levels = sorted(filtered_df['Delta'].unique())
    
    if filtered_df.empty:
        print("Warning: no CVaR data for filtered demand levels.")
        return
    
    fixed_demand_levels = [0.5, 0.75, 1.0, 1.25, 1.5]
    fixed_markers = ['o', 's', '^', 'D', 'X']
    fixed_colors = sns.color_palette("viridis", n_colors=5)
    
    markers_map = {level: marker for level, marker in zip(fixed_demand_levels, fixed_markers)}
    palette_map = {level: color for level, color in zip(fixed_demand_levels, fixed_colors)}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, ax = plt.subplots(figsize=(32, 18))
    
    scatter = sns.scatterplot(data=filtered_df, 
                   x='Avg_Delay', 
                   y='CVaR_30', 
                   hue='Delta',
                   style='Delta',
                   markers=markers_map,
                   palette=palette_map,
                   size='Alpha', 
                   sizes=(150, 1200),
                   alpha=0.8, 
                   ax=ax)
    
    ax.set_xlabel('Average delay (min)', fontsize=48)
    ax.set_ylabel('CVaR 30% of delay (min)', fontsize=48)
    handles, labels = ax.get_legend_handles_labels()
    combined_handles = []
    combined_labels = []
    combined_labels.append('Demand rate\n(requests/min)')
    combined_handles.append(plt.Rectangle((0,0),1,1, fill=False, edgecolor='none', visible=False))
    for i, demand_val in enumerate(fixed_demand_levels):
        if demand_val in actual_demand_levels:
            handle = plt.scatter([], [], marker=fixed_markers[i], s=300,
                               color=fixed_colors[i], alpha=0.8)
            combined_handles.append(handle)
            combined_labels.append(f'{demand_val:.2f}')
    combined_labels.append('Alpha')
    combined_handles.append(plt.Rectangle((0,0),1,1, fill=False, edgecolor='none', visible=False))
    
    alpha_values = sorted(filtered_df['Alpha'].unique())
    for alpha_val in alpha_values:
        size = 150 + (1200-150) * (alpha_val - min(alpha_values)) / (max(alpha_values) - min(alpha_values)) if len(alpha_values) > 1 else 675
        handle = plt.scatter([], [], s=size, color='gray', alpha=0.6)
        combined_handles.append(handle)
        combined_labels.append(f'α={alpha_val:.1f}')
    
    combined_legend = ax.legend(combined_handles, combined_labels, 
                               bbox_to_anchor=(1.015, 1), loc='upper left', 
                               fontsize=39, labelspacing=0.8, handletextpad=0.8)
    
    for text in combined_legend.get_texts():
        if 'Demand rate' in text.get_text() or 'Alpha' in text.get_text():
            text.set_horizontalalignment('left')
            text.set_x(0)
    
    ax.grid(True, which='both', linestyle='--', linewidth=1)
    ax.tick_params(axis='both', which='major', labelsize=42)
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.82, top=0.98)
    
    output_filename = os.path.join(output_dir, f'10_tradeoff_plot_CVaR_30_{timestamp}.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"CVaR 30% tradeoff plot saved: '{output_filename}'")
    plt.close(fig)

def plot_demand_specific_cvar_graphs(df, cvar_df, output_dir):
    """Plot demand-specific CVaR 30% vs Alpha. df: summary with Avg_Delay; cvar_df: Alpha, Demand, Seed, CVaR_30; output_dir: output path."""
    if cvar_df.empty:
        print("Warning: no CVaR data; skipping demand-specific CVaR plots.")
        return
    
    df_for_merge = df[['Alpha', 'Demand', 'Seed', 'Avg_Delay']].copy()
    merged_df = pd.merge(cvar_df, df_for_merge, on=['Alpha', 'Demand', 'Seed'], how='inner')
    if merged_df.empty:
        print("Warning: could not merge CVaR and summary data.")
        return
    merged_df = merged_df[merged_df['Alpha'] < 1.0]
    if merged_df.empty:
        print("Warning: no Alpha < 1.0 data; skipping demand-specific CVaR plots.")
        return
    
    cvar_agg = merged_df.groupby(['Alpha', 'Demand']).agg(
        Avg_Delay=('Avg_Delay', 'mean'),
        CVaR_30=('CVaR_30', 'mean')
    ).reset_index()
    
    plot_folder = os.path.join(output_dir, "11_demand_specific_cvar_plots")
    os.makedirs(plot_folder, exist_ok=True)
    print(f"Demand-specific CVaR plot folder: '{plot_folder}'")
    
    unique_demands = cvar_agg['Demand'].unique()
    total_graphs = 0
    
    for demand in unique_demands:
        demand_data = cvar_agg[cvar_agg['Demand'] == demand].sort_values('Alpha')
        demand_str = demand.replace('.', 'p')
        
        if demand_data.empty:
            continue
        
        print(f"\n{'='*80}")
        print(f"Demand level: {demand}")
        print(f"{'='*80}")
        print(f"{'Alpha':>8} | {'CVaR 30% (min)':>18} | {'Average Delay (min)':>20}")
        print(f"{'-'*80}")
        for _, row in demand_data.iterrows():
            print(f"{row['Alpha']:>8.1f} | {row['CVaR_30']:>18.3f} | {row['Avg_Delay']:>20.3f}")
        print(f"{'='*80}\n")
        
        fig, ax1 = plt.subplots(figsize=(16, 10))
        color1 = 'tab:red'
        ax1.set_xlabel('Alpha (α)', fontsize=28)
        ax1.set_ylabel('CVaR 30% of delay (min)', color=color1, fontsize=28)
        ax1.plot(demand_data['Alpha'], demand_data['CVaR_30'], color=color1, marker='o', 
                markersize=12, linestyle='-', label='CVaR 30%', linewidth=3)
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=24)
        ax1.tick_params(axis='x', labelsize=24)
        ax1.xaxis.grid(True, which='both', linestyle='--', linewidth=1)
        ax1.yaxis.grid(False)
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Average delay (min)', color=color2, fontsize=28)
        ax2.plot(demand_data['Alpha'], demand_data['Avg_Delay'], color=color2, marker='s', 
                markersize=12, linestyle='--', label='Average delay', linewidth=3)
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=24)
        ax2.yaxis.grid(False)        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=20)
        
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
        
        output_filename = os.path.join(plot_folder, f'demand_{demand_str}_cvar30_vs_avg.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        total_graphs += 1
    
    print(f"Demand-specific CVaR plots done: {total_graphs} plots.")

def main():
    """Load data and run selected visualizations."""
    parser = argparse.ArgumentParser(description="Analyze DARP simulation results and generate tradeoff plots. Uses latest sweep by default; use --dir for a specific sweep directory.")
    parser.add_argument('--dir', type=str, help='Path to sweep directory (e.g. path/to/sweep_YYYYMMDD_HHMMSS).')
    args = parser.parse_args()

    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    print("\n=============================================")
    print("Select visualizations to generate:")
    print("2: Combined tradeoffs (2x2 grid)")
    print("3: Gini tradeoff (general + filtered)")
    print("4: Tail Delay tradeoff")
    print("5: Detour Factor tradeoff")
    print("6: VKT tradeoff")
    print("7-1: Demand-specific plots (average)")
    print("7-2: Demand-specific plots (by seed)")
    print("8: Occupancy analysis")
    print("9: Policy effectiveness analysis")
    print("10: CVaR 30% tradeoff (Pareto frontier)")
    print("11: Demand-specific CVaR 30% plots")
    print("12: Stats by demand/alpha (Mean ± 95% CI: Avg Delay, Gini, CVaR, Runtime per 1-min opt)")
    print("0: All visualizations")
    print("Separate multiple choices with commas (e.g. 2,3,10,11)")
    print("---------------------------------------------")
    user_input = input("Enter choice(s): ").strip()
    if user_input == '0':
        selected_plots = [2, 3, 4, 5, 6, '7-1', '7-2', 8, 9, 10, 11, 12]
    else:
        try:
            selected_plots = []
            for x in user_input.split(','):
                x = x.strip()
                if x in ['7-1', '7-2']:
                    selected_plots.append(x)
                else:
                    num = int(x)
                    if num in [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]:
                        selected_plots.append(num)
        except ValueError:
            print("Invalid input. Generating all visualizations.")
            selected_plots = [2, 3, 4, 5, 6, '7-1', '7-2', 8, 9, 10, 11, 12]
    print(f"Selected: {selected_plots}")
    print("=============================================\n")

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: invalid directory: '{args.dir}'")
            return
        sweep_dir = args.dir
        print(f"Using sweep dir: '{sweep_dir}'")
    else:
        sweep_dir = find_latest_sweep_dir()
        if sweep_dir is None:
            print("Error: no sweep_* directory found. Run run_alpha_sweep_SA_delta.py first.")
            return
        print(f"Using sweep dir: '{sweep_dir}'")
    params_file = os.path.join(sweep_dir, 'parameters.json')
    if not os.path.exists(params_file):
        print(f"Warning: parameters file not found: '{params_file}'")
        params = {}
    else:
        with open(params_file, 'r') as f:
            params = json.load(f)
        print("Parameters loaded.")
    summary_file = os.path.join(sweep_dir, '1_all_simulations_summary.csv')
    if not os.path.exists(summary_file):
        print(f"Error: summary file not found: '{summary_file}'")
        return
    try:
        full_df = pd.read_csv(summary_file)
        print("Summary data loaded.")
    except Exception as e:
        print(f"Error reading summary file: {e}")
        return
    if full_df.empty:
        print("Summary data is empty. Aborting.")
        return
    cvar_df = pd.DataFrame()
    if 10 in selected_plots or 11 in selected_plots:
        print("\nComputing CVaR 30% data...")
        cvar_df = calculate_cvar_from_excel(sweep_dir, cvar_percentile=0.3)
        if cvar_df.empty:
            print("Warning: CVaR computation failed. Skipping 10/11.")
            if 10 in selected_plots:
                selected_plots.remove(10)
            if 11 in selected_plots:
                selected_plots.remove(11)
    print(f"\nGenerating: {selected_plots}")
    if 2 in selected_plots:
        print("- 2: Combined tradeoffs...")
        plot_combined_tradeoffs(full_df, sweep_dir)
    if any(x in selected_plots for x in [3, 4, 5, 6]):
        print(f"- {[x for x in [3,4,5,6] if x in selected_plots]}: Tradeoff plots...")
        plot_tradeoffs(full_df, sweep_dir, selected_plots)
    if '7-1' in selected_plots:
        print("- 7-1: Demand-specific (average)...")
        plot_demand_specific_graphs(full_df, sweep_dir, 'average')
    if '7-2' in selected_plots:
        print("- 7-2: Demand-specific (by seed)...")
        plot_demand_specific_graphs(full_df, sweep_dir, 'individual')
    if 8 in selected_plots:
        print("- 8: Occupancy analysis...")
        plot_occupancy_analysis(sweep_dir, params)
    if 9 in selected_plots:
        print("- 9: Effectiveness analysis...")
        plot_effectiveness_analysis(full_df, sweep_dir)
    if 10 in selected_plots:
        print("- 10: CVaR tradeoff...")
        plot_cvar_tradeoffs(full_df, cvar_df, sweep_dir)
    if 11 in selected_plots:
        print("- 11: Demand-specific CVaR...")
        plot_demand_specific_cvar_graphs(full_df, cvar_df, sweep_dir)
    if 12 in selected_plots:
        print("- 12: Stats by demand/alpha...")
        print_stats_by_demand_alpha(sweep_dir, full_df)
    print("\nDone.")

if __name__ == '__main__':
    main() 