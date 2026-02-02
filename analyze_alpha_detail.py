import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import glob
from tqdm import tqdm
import numpy as np
from datetime import datetime

"""
analyze_alpha_detail.py
======================
Detailed comparison visualizations for DARP (Demand-Responsive Transit) simulation
sweep results. Compares a user-selected target alpha (or multiple alphas) against
baseline alpha=0.0 (mean-only) across demand levels and random seeds.

Objective function (same as sweep): OFV = (1 - alpha)*Mean_Delay + alpha*Tail_Delay,
  - alpha=0.0: mean-only (efficiency);
  - alpha=1.0: CVaR-only (equity/tail risk).
Baseline for all comparisons is alpha=0.0.

Inputs:
  - Fixed sweep directory (detailed_results with alpha_* folders, time_series_data_*.csv,
    results_s2_*.xlsx, occupancy_data_*.csv). Optionally 1_all_simulations_summary.csv for VKT.
  - Interactive prompts: alpha(s) to compare (e.g. 0.1, 0.5, or 'all'), demand level
    (e.g. demand_1x or 'all').

Outputs (saved under sweep_dir/batch_comparison_vs_a0.0_* or specific_comparison_*):
  1. OFV_Trajectory: cumulative total delay and cumulative CVaR of delay over time (per seed).
  2. CVaR_Ratio: CVaR-to-total delay ratio over time.
  3. CVaR_vs_CumulativePassengers: cumulative CVaR of delay vs cumulative served passengers.
  4. Delay distribution histograms (with percentile lines at 70th, 80th, 90th) and boxplots.
  5. Lorenz curve and CDF of delay (equity).
  6. Detour factor distribution and boxplots (if DetourFactor column exists).
  7. Occupancy trajectory (average occupancy per vehicle over time), if occupancy_data_*.csv exist.
  8. VKT boxplot and distribution (if summary CSV available).

Usage: python analyze_alpha_detail.py
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def annotate_boxplot(ax, data, x_col, y_col):
    """Annotate box plot with Q1, Q3, median, mean, whiskers."""
    x_order = [tick.get_text() for tick in ax.get_xticklabels()]
    for i, cat in enumerate(x_order):
        subset = data[data[x_col] == cat][y_col]
        q1 = subset.quantile(0.25)
        median = subset.median()
        q3 = subset.quantile(0.75)
        mean_val = subset.mean()
        iqr = q3 - q1
        lower_whisker = subset[subset >= q1 - 1.5 * iqr].min()
        upper_whisker = subset[subset <= q3 + 1.5 * iqr].max()
        x_pos = i
        ax.text(x_pos + 0.05, q1, f'Q1={q1:.3f}', va='center', ha='left', fontsize=18, color='green')
        ax.text(x_pos + 0.05, q3, f'Q3={q3:.3f}', va='center', ha='left', fontsize=18, color='green')
        ax.text(x_pos, mean_val, f'average={mean_val:.3f}', va='center', ha='right', fontsize=18, color='red')
        ax.text(x_pos, median, f'median={median:.3f}', va='center', ha='left', fontsize=18, color='blue')
        ax.text(x_pos, upper_whisker, f'Upper Whisker={upper_whisker:.3f}', va='bottom', ha='center', fontsize=18, color='green')
        ax.text(x_pos, lower_whisker, f'Lower Whisker={lower_whisker:.3f}', va='top', ha='center', fontsize=18, color='green')

def plot_lorenz_curve(df_baseline, df_target, delay_col, demand_name, output_dir, target_alpha_str):
    """Plot Lorenz curve comparing baseline vs target."""
    plt.figure(figsize=(14, 10))
    data_b = df_baseline[delay_col].dropna().sort_values().to_numpy()
    if len(data_b) > 0:
        cum_b = np.cumsum(data_b) / data_b.sum()
        plt.plot(np.linspace(0, 1, len(cum_b)), cum_b, color='red', linestyle='--', linewidth=3, label='α=0.0 (Baseline)')
    data_t = df_target[delay_col].dropna().sort_values().to_numpy()
    if len(data_t) > 0:
        cum_t = np.cumsum(data_t) / data_t.sum()
        plt.plot(np.linspace(0, 1, len(cum_t)), cum_t, color='blue', linewidth=3, label=f'α={target_alpha_str}')
    plt.plot([0, 1], [0, 1], color='grey', linestyle=':', linewidth=2, label='Line of Perfect Equality')
    plt.xlabel('Cumulative Share of Passengers (from lowest to highest delay)', fontsize=28)
    plt.ylabel('Cumulative Share of Total Delay', fontsize=28)
    legend = plt.legend(fontsize=31, frameon=True, fancybox=False, edgecolor='black')
    legend.get_frame().set_linewidth(1.5)
    plt.grid(True, alpha=0.5, linewidth=1)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
    plt.savefig(os.path.join(output_dir, f"{demand_name}_Lorenz_Curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_cdf(df_baseline, df_target, delay_col, demand_name, output_dir, target_alpha_str):
    """Plot CDF comparing baseline vs target."""
    plt.figure(figsize=(16, 10))
    sns.ecdfplot(data=df_baseline, x=delay_col, color='red', linestyle='--', linewidth=3, label='α=0.0 (Baseline)')
    sns.ecdfplot(data=df_target, x=delay_col, color='blue', linewidth=3, label=f'α={target_alpha_str}')
    plt.xlabel('Delay (min)', fontsize=28)
    plt.ylabel('Cumulative Probability (P(Delay <= x))', fontsize=28)
    legend = plt.legend(fontsize=31, frameon=True, fancybox=False, edgecolor='black')
    legend.get_frame().set_linewidth(1.5)
    plt.grid(True, alpha=0.5, linewidth=1)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
    plt.savefig(os.path.join(output_dir, f"{demand_name}_Delay_CDF.png"), dpi=300, bbox_inches='tight')
    plt.close()

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

def get_user_input():
    """Prompt for alpha(s) and demand level."""
    print("=============================================")
    print("Detailed comparison visualization settings")
    print("=============================================")
    target_alphas_list = []
    while True:
        alpha_input = input("First alpha to compare (e.g. 0.1, 0.5, all): ").strip().lower()
        if alpha_input == 'all':
            target_alphas_list = 'all'
            break
        try:
            alpha1 = float(alpha_input)
            if 0.0 <= alpha1 <= 1.0:
                target_alphas_list.append(alpha1)
                break
            print("Alpha must be between 0.0 and 1.0.")
        except ValueError:
            print("Enter a number or 'all'.")
    if target_alphas_list != 'all':
        while True:
            alpha_input2 = input("Second alpha (e.g. 0.5, 0.8, or Enter to skip): ").strip().lower()
            if alpha_input2 == '':
                break
            try:
                alpha2 = float(alpha_input2)
                if 0.0 <= alpha2 <= 1.0 and alpha2 != target_alphas_list[0]:
                    target_alphas_list.append(alpha2)
                    break
                print("Alpha must be 0.0–1.0 and different from the first.")
            except ValueError:
                print("Enter a number or Enter.")
    while True:
        demand_input = input("Demand level (e.g. demand_0.7x, demand_1x, all): ").strip().lower()
        if demand_input == 'all':
            target_demand = 'all'
            break
        if demand_input.startswith('demand_'):
            target_demand = demand_input
            break
        print("Use format demand_Xx or 'all'.")
    print(f"Alpha(s): {target_alphas_list}")
    print(f"Demand: {target_demand}")
    print("=============================================")
    return target_alphas_list, target_demand

def main():
    """Load data and run detailed visualizations for selected alpha(s) and demand level."""
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    user_alphas, user_demand = get_user_input()
    color_palette_base = {'baseline': 'red', 'target1': 'blue', 'target2': 'green'}
    color_palette_light = {'baseline': 'lightcoral', 'target1': 'skyblue', 'target2': 'lightgreen'}
    sweep_dir = find_latest_sweep_dir()
    if not sweep_dir:
        print("Error: sweep directory not found. Run run_alpha_sweep first.")
        return
    print(f"Using sweep dir: '{sweep_dir}'")
    summary_csv_path = os.path.join(sweep_dir, '1_all_simulations_summary.csv')
    if not os.path.exists(summary_csv_path):
        print(f"Warning: summary CSV not found; skipping VKT analysis.")
        summary_df = None
    else:
        summary_df = pd.read_csv(summary_csv_path)
        print("Summary data loaded.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if user_alphas == 'all' and user_demand == 'all':
        batch_output_dir = os.path.join(sweep_dir, f"batch_comparison_vs_a0.0_{timestamp}")
        print(f"All comparison outputs will be saved to: '{batch_output_dir}'")
    else:
        folder_name = f"specific_comparison_vs_a0.0_{timestamp}"
        if user_alphas != 'all':
            for ua in user_alphas:
                folder_name += f"_a{ua:.1f}"
        if user_demand != 'all':
            folder_name += f"_{user_demand}"
        batch_output_dir = os.path.join(sweep_dir, folder_name)
        print(f"Comparison outputs will be saved to: '{batch_output_dir}'")
    
    os.makedirs(batch_output_dir, exist_ok=True)

    detailed_results_dir = os.path.join(sweep_dir, "detailed_results")
    baseline_alpha_folder_name = "alpha_0.0"
    alpha_dir_baseline = os.path.join(detailed_results_dir, baseline_alpha_folder_name)
    
    if not os.path.isdir(alpha_dir_baseline):
        print(f"Error: baseline folder '{baseline_alpha_folder_name}' not found.")
        return

    if user_alphas == 'all':
        all_alpha_dirs = glob.glob(os.path.join(detailed_results_dir, 'alpha_*'))
        target_alphas_list_all = []
        for d in all_alpha_dirs:
            folder_name = os.path.basename(d)
            if folder_name == baseline_alpha_folder_name:
                continue
            try:
                alpha_val = float(folder_name.replace('alpha_', ''))
                target_alphas_list_all.append(alpha_val)
            except ValueError:
                continue
        
        target_alphas_list_all.sort(reverse=True)
        if not target_alphas_list_all:
            print("Error: no Alpha folders (other than alpha_0.0) found.")
            return
        multi_alpha_mode = False
        target_alpha_groups = [[a] for a in target_alphas_list_all]
    else:
        multi_alpha_mode = len(user_alphas) > 1
        for ua in user_alphas:
            target_alpha_str = f"{ua:.1f}"
            target_alpha_folder = os.path.join(detailed_results_dir, f"alpha_{target_alpha_str}")
            
            if not os.path.isdir(target_alpha_folder):
                print(f"Error: Alpha folder 'alpha_{target_alpha_str}' not found.")
                return
        
        if multi_alpha_mode:
            print(f"Multi-alpha comparison: {user_alphas} vs baseline.")
            target_alpha_groups = [user_alphas]
        else:
            print(f"Visualizing alpha {user_alphas[0]} only.")
            target_alpha_groups = [[user_alphas[0]]]

    for target_alpha_group in tqdm(target_alpha_groups, desc="Alpha Comparisons"):
        is_multi = len(target_alpha_group) > 1
        display_alphas = list(target_alpha_group)
        display_alpha_strs = [f"{da:.1f}" for da in display_alphas]
        target_alpha_strs = [f"{ta:.1f}" for ta in target_alpha_group]
        alpha_dirs_target = [os.path.join(detailed_results_dir, f"alpha_{tas}") for tas in target_alpha_strs]

        if user_alphas == 'all' and user_demand == 'all':
            output_viz_dir = os.path.join(batch_output_dir, f"a{display_alpha_strs[0]}_vs_a0.0")
        else:
            if is_multi:
                folder_suffix = f"a{'_a'.join(display_alpha_strs)}_vs_a0.0"
            else:
                folder_suffix = f"a{display_alpha_strs[0]}_vs_a0.0"
            if user_demand != 'all':
                folder_suffix += f"_{user_demand}"
            output_viz_dir = os.path.join(batch_output_dir, folder_suffix)
        
        os.makedirs(output_viz_dir, exist_ok=True)
        
        target_alpha = target_alpha_group[0]
        display_alpha = display_alphas[0]
        display_alpha_str = display_alpha_strs[0]
        target_alpha_str = target_alpha_strs[0]
        alpha_dir_target = alpha_dirs_target[0]
        
        target_color_list = ['blue', 'green', 'orange', 'purple']
        target_color_light_list = ['skyblue', 'lightgreen', 'moccasin', 'plum']
        percentile_colors = ['darkred', 'darkblue', 'darkgreen', 'darkorange']
        if user_demand == 'all':
            trajectory_files = glob.glob(os.path.join(alpha_dir_target, "time_series_data_*.csv"))
        else:
            trajectory_files = glob.glob(os.path.join(alpha_dir_target, f"time_series_data_{user_demand}.csv"))
            if not trajectory_files:
                print(f"Warning: no files found for demand level '{user_demand}'.")
                continue
        
        desc_text = f"Plots for a={','.join(target_alpha_strs)}"
        if user_demand != 'all':
            desc_text += f" {user_demand}"
            
        for file_path_target in tqdm(trajectory_files, desc=desc_text, leave=False):
            demand_name_match = re.search(r'time_series_data_(.*)\.csv', os.path.basename(file_path_target))
            if not demand_name_match: continue
            demand_name = demand_name_match.group(1)
            file_path_baseline = os.path.join(alpha_dir_baseline, f"time_series_data_{demand_name}.csv")
            if not os.path.exists(file_path_baseline): continue

            df_ts_target = pd.read_csv(file_path_target)
            df_ts_baseline = pd.read_csv(file_path_baseline)

            def _compat_trajectory(df: pd.DataFrame) -> pd.DataFrame:
                rename_map = {}
                if 'CumulativeDelaySum' in df.columns and 'S2_SumDelay' not in df.columns:
                    rename_map['CumulativeDelaySum'] = 'S2_SumDelay'
                if 'CumulativeTailDelaySum' in df.columns and 'S2_TailSumDelay' not in df.columns:
                    rename_map['CumulativeTailDelaySum'] = 'S2_TailSumDelay'
                if rename_map:
                    df = df.rename(columns=rename_map)
                if 'Seed' not in df.columns:
                    df = df.copy()
                    df['Seed'] = 0
                return df

            df_ts_target = _compat_trajectory(df_ts_target)
            df_ts_baseline = _compat_trajectory(df_ts_baseline)

            num_seeds = int(df_ts_target['Seed'].max()) + 1

            for seed_idx in range(num_seeds):
                df_seed_target = df_ts_target[df_ts_target['Seed'] == seed_idx]
                df_seed_baseline = df_ts_baseline[df_ts_baseline['Seed'] == seed_idx]

                if df_seed_target.empty or df_seed_baseline.empty: continue

                plt.figure(figsize=(18, 9.6))
                plt.plot(df_seed_baseline['Time'], df_seed_baseline['S2_SumDelay'], color='red', linestyle='-', marker='.', markersize=8, linewidth=3, label=f'Total cumulative delay (α=0.0)')
                plt.plot(df_seed_baseline['Time'], df_seed_baseline['S2_TailSumDelay'], color='lightcoral', linestyle='--', linewidth=3, label=f'Total cumulative CVaR of delay (α=0.0)')
                plt.plot(df_seed_target['Time'], df_seed_target['S2_SumDelay'], color='blue', linestyle='-', marker='.', markersize=8, linewidth=3, label=f'Total cumulative delay (α={display_alpha:.1f})')
                plt.plot(df_seed_target['Time'], df_seed_target['S2_TailSumDelay'], color='skyblue', linestyle='--', linewidth=3, label=f'Total cumulative CVaR of delay (α={display_alpha:.1f})')
                plt.xlabel('Time (min)', fontsize=28, fontfamily='Times New Roman')
                plt.ylabel('Total cumulative delay (min)', fontsize=28, fontfamily='Times New Roman')
                plt.grid(True, alpha=0.5, linewidth=1)
                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)
                plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
                plt.savefig(os.path.join(output_viz_dir, f"{demand_name}_Seed{seed_idx+1}_OFV_Trajectory.png"), dpi=300, bbox_inches='tight')
                plt.close()

                df_seed_baseline_copy = df_seed_baseline.copy()
                df_seed_target_copy = df_seed_target.copy()
                df_seed_baseline_copy['CVaR_Ratio'] = df_seed_baseline_copy.apply(
                    lambda row: (row['S2_TailSumDelay'] / row['S2_SumDelay'] * 100) if row['S2_SumDelay'] > 0 else 0, axis=1
                )
                df_seed_target_copy['CVaR_Ratio'] = df_seed_target_copy.apply(
                    lambda row: (row['S2_TailSumDelay'] / row['S2_SumDelay'] * 100) if row['S2_SumDelay'] > 0 else 0, axis=1
                )
                
                plt.figure(figsize=(18, 9.6))
                
                # Baseline (alpha=0.0)
                plt.plot(df_seed_baseline_copy['Time'], df_seed_baseline_copy['CVaR_Ratio'], 
                        color='red', linestyle='-', marker='.', markersize=6, linewidth=3, 
                        label=f'CVaR-to-total delay ratio (α=0.0)', alpha=0.8)
                
                # Target
                plt.plot(df_seed_target_copy['Time'], df_seed_target_copy['CVaR_Ratio'], 
                        color='blue', linestyle='-', marker='.', markersize=6, linewidth=3, 
                        label=f'CVaR-to-total delay ratio (α={display_alpha:.1f})', alpha=0.8)

                plt.xlabel('Time (min)', fontsize=36, fontfamily='Times New Roman')
                plt.ylabel('CVaR-to-total delay ratio (%)', fontsize=36, fontfamily='Times New Roman')
                plt.grid(True, alpha=0.5, linewidth=1)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
                plt.savefig(os.path.join(output_viz_dir, f"{demand_name}_Seed{seed_idx+1}_CVaR_Ratio.png"), dpi=300, bbox_inches='tight')
                plt.close()

                if 'S2_Throughput' in df_seed_baseline.columns:
                    cum_pax_col = 'S2_Throughput'
                elif 'CumulativeServed' in df_seed_baseline.columns:
                    cum_pax_col = 'CumulativeServed'
                elif 'NumServed' in df_seed_baseline.columns:
                    df_seed_baseline_copy['CumulativeServed'] = df_seed_baseline_copy['NumServed'].cumsum()
                    df_seed_target_copy['CumulativeServed'] = df_seed_target_copy['NumServed'].cumsum()
                    cum_pax_col = 'CumulativeServed'
                else:
                    df_seed_baseline_copy['CumulativeServed'] = range(len(df_seed_baseline_copy))
                    df_seed_target_copy['CumulativeServed'] = range(len(df_seed_target_copy))
                    cum_pax_col = 'CumulativeServed'
                df_baseline_filtered = df_seed_baseline_copy[df_seed_baseline_copy[cum_pax_col] > 0].copy()
                df_target_filtered = df_seed_target_copy[df_seed_target_copy[cum_pax_col] > 0].copy()
                
                plt.figure(figsize=(18, 9.6))
                plt.plot(df_baseline_filtered[cum_pax_col], df_baseline_filtered['S2_TailSumDelay'], 
                        color='red', linestyle='-', marker='o', markersize=4, linewidth=3, 
                        label=f'Total cumulative CVaR of delay (α=0.0)', alpha=0.8)
                plt.plot(df_target_filtered[cum_pax_col], df_target_filtered['S2_TailSumDelay'], 
                        color='blue', linestyle='-', marker='o', markersize=4, linewidth=3, 
                        label=f'Total cumulative CVaR of delay (α={display_alpha:.1f})', alpha=0.8)

                plt.xlabel('Cumulative number of served passengers', fontsize=36, fontfamily='Times New Roman')
                plt.ylabel('Total cumulative CVaR of delay (min)', fontsize=36, fontfamily='Times New Roman')
                plt.grid(True, alpha=0.5, linewidth=1)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
                plt.savefig(os.path.join(output_viz_dir, f"{demand_name}_Seed{seed_idx+1}_CVaR_vs_CumulativePassengers.png"), dpi=300, bbox_inches='tight')
                plt.close()

            excel_path_s2_baseline = os.path.join(alpha_dir_baseline, f"results_s2_{demand_name}.xlsx")
            excel_paths_s2_targets = []
            all_excel_exist = os.path.exists(excel_path_s2_baseline)
            for adt in alpha_dirs_target:
                excel_path = os.path.join(adt, f"results_s2_{demand_name}.xlsx")
                excel_paths_s2_targets.append(excel_path)
                if not os.path.exists(excel_path):
                    all_excel_exist = False

            if not all_excel_exist:
                print(f"  Skipping: no results_s2 Excel for {demand_name}.")
            else:
                df_baseline_total = pd.concat(pd.read_excel(excel_path_s2_baseline, sheet_name=None).values(), ignore_index=True)
                df_targets_total = []
                for excel_path in excel_paths_s2_targets:
                    df_t = pd.concat(pd.read_excel(excel_path, sheet_name=None).values(), ignore_index=True)
                    df_targets_total.append(df_t)
                
                delay_col = 'Delay'
                alpha_colors = {'α=0.0 (Baseline)': 'red'}
                alpha_colors_light = {'α=0.0 (Baseline)': 'lightcoral'}
                
                plot_data_list = [df_baseline_total[[delay_col]].assign(Alpha='α=0.0 (Baseline)')]
                for i, (df_t, das) in enumerate(zip(df_targets_total, display_alpha_strs)):
                    alpha_label = f'α={das}'
                    plot_data_list.append(df_t[[delay_col]].assign(Alpha=alpha_label))
                    alpha_colors[alpha_label] = target_color_list[i % len(target_color_list)]
                    alpha_colors_light[alpha_label] = target_color_light_list[i % len(target_color_light_list)]
                
                plot_data = pd.concat(plot_data_list)
                percentiles = [0.7, 0.8, 0.9]
                for p in percentiles:
                    plt.figure(figsize=(20, 10))
                    sns.histplot(data=plot_data, x=delay_col, hue='Alpha', multiple='dodge', shrink=0.8, bins=30,
                                 palette=alpha_colors)
                    percentile_str = int(p * 100)
                    q_baseline = df_baseline_total[delay_col].quantile(p)
                    plt.axvline(q_baseline, color='darkred', linestyle='--', linewidth=3, label=f'{percentile_str}th percentile (α=0.0): {q_baseline:.2f} min')
                    for i, (df_t, das) in enumerate(zip(df_targets_total, display_alpha_strs)):
                        q_target = df_t[delay_col].quantile(p)
                        plt.axvline(q_target, color=percentile_colors[(i+1) % len(percentile_colors)], linestyle='--', linewidth=3, 
                                    label=f'{percentile_str}th percentile (α={das}): {q_target:.2f} min')
                    
                    legend = plt.legend(fontsize=26, frameon=True, fancybox=False, edgecolor='black')
                    legend.get_frame().set_linewidth(1.5)
                    plt.xlabel('Delay (min)', fontsize=28)
                    plt.ylabel('Count', fontsize=28)
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
                    plt.savefig(os.path.join(output_viz_dir, f"{demand_name}_Delay_Distribution_{percentile_str}ile.png"), dpi=300, bbox_inches='tight')
                    plt.close()

                plt.figure(figsize=(14 + 3 * (len(target_alpha_group) - 1), 10))
                ax = sns.boxplot(data=plot_data, x='Alpha', y=delay_col, hue='Alpha', palette=alpha_colors_light, legend=False)
                annotate_boxplot(ax, plot_data, 'Alpha', delay_col)
                plt.xlabel('Alpha', fontsize=28)
                plt.ylabel('Delay (min)', fontsize=28)
                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)
                plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
                plt.savefig(os.path.join(output_viz_dir, f"{demand_name}_Delay_Boxplot.png"), dpi=300, bbox_inches='tight')
                plt.close()
                plot_lorenz_curve(df_baseline_total, df_targets_total[0], delay_col, demand_name, output_viz_dir, display_alpha_strs[0])
                plot_cdf(df_baseline_total, df_targets_total[0], delay_col, demand_name, output_viz_dir, display_alpha_strs[0])
                detour_col = 'DetourFactor'
                all_have_detour = detour_col in df_baseline_total.columns and all(detour_col in df_t.columns for df_t in df_targets_total)
                if all_have_detour:
                    plot_data_detour_list = [df_baseline_total[[detour_col]].assign(Alpha='α=0.0 (Baseline)')]
                    for i, (df_t, das) in enumerate(zip(df_targets_total, display_alpha_strs)):
                        alpha_label = f'α={das}'
                        plot_data_detour_list.append(df_t[[detour_col]].assign(Alpha=alpha_label))
                    
                    plot_data_detour = pd.concat(plot_data_detour_list)
                    plt.figure(figsize=(14 + 3 * (len(target_alpha_group) - 1), 10))
                    ax = sns.boxplot(data=plot_data_detour, x='Alpha', y=detour_col, hue='Alpha', palette=alpha_colors_light, legend=False)
                    annotate_boxplot(ax, plot_data_detour, 'Alpha', detour_col)
                    plt.ylabel('Detour Factor (In-Vehicle Time / Ideal Time)', fontsize=28)
                    plt.xlabel('Alpha', fontsize=28)
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
                    plt.savefig(os.path.join(output_viz_dir, f"{demand_name}_Detour_Boxplot.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                    percentiles = [0.7, 0.8, 0.9]
                    for p in percentiles:
                        plt.figure(figsize=(20, 10))
                        sns.histplot(data=plot_data_detour, x=detour_col, hue='Alpha', multiple='dodge', shrink=0.8, bins=30,
                                     palette=alpha_colors)
                        percentile_str = int(p * 100)
                        q_baseline = df_baseline_total[detour_col].quantile(p)
                        if pd.notna(q_baseline):
                            plt.axvline(q_baseline, color='darkred', linestyle='--', linewidth=3, label=f'{percentile_str}th percentile (α=0.0): {q_baseline:.2f}')
                        for i, (df_t, das) in enumerate(zip(df_targets_total, display_alpha_strs)):
                            q_target = df_t[detour_col].quantile(p)
                            if pd.notna(q_target):
                                plt.axvline(q_target, color=percentile_colors[(i+1) % len(percentile_colors)], linestyle='--', linewidth=3, 
                                            label=f'{percentile_str}th percentile (α={das}): {q_target:.2f}')
                        
                        legend = plt.legend(fontsize=26, frameon=True, fancybox=False, edgecolor='black')
                        legend.get_frame().set_linewidth(1.5)
                        plt.xlabel('Detour Factor (In-Vehicle Time / Ideal Time)', fontsize=28)
                        plt.ylabel('Count', fontsize=28)
                        plt.xticks(fontsize=24)
                        plt.yticks(fontsize=24)
                        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
                        plt.savefig(os.path.join(output_viz_dir, f"{demand_name}_Detour_Distribution_{percentile_str}ile.png"), dpi=300, bbox_inches='tight')
                        plt.close()

            occupancy_path_target = os.path.join(alpha_dir_target, f"occupancy_data_{demand_name}.csv")
            occupancy_path_baseline = os.path.join(alpha_dir_baseline, f"occupancy_data_{demand_name}.csv")

            if os.path.exists(occupancy_path_target) and os.path.exists(occupancy_path_baseline):
                df_occ_target = pd.read_csv(occupancy_path_target)
                df_occ_baseline = pd.read_csv(occupancy_path_baseline)
                df_occ_target_s2 = df_occ_target[df_occ_target['Scenario'] == 'S2']
                df_occ_baseline_s2 = df_occ_baseline[df_occ_baseline['Scenario'] == 'S2']
                
                if not df_occ_target_s2.empty and not df_occ_baseline_s2.empty:
                    plot_data_occ = pd.concat([
                        df_occ_baseline_s2.assign(Alpha='α=0.0 (Baseline)'),
                        df_occ_target_s2.assign(Alpha=f'α={display_alpha_str}')
                    ])
                    plt.figure(figsize=(18, 8))
                    sns.lineplot(data=plot_data_occ, x='Time', y='Occupancy', hue='Alpha', 
                                 palette={'α=0.0 (Baseline)': 'red', f'α={display_alpha_str}': 'blue'}, 
                                 errorbar='sd', linewidth=3)
                    plt.xlabel('Time (min)', fontsize=28, fontfamily='Times New Roman')
                    plt.ylabel('Average Occupancy per Vehicle', fontsize=28, fontfamily='Times New Roman')
                    legend = plt.legend(fontsize=36, prop={'family': 'Times New Roman'},
                                       frameon=True, fancybox=False, edgecolor='black')
                    legend.get_frame().set_linewidth(1.5)
                    plt.grid(True, alpha=0.5, linewidth=1)
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
                    plt.savefig(os.path.join(output_viz_dir, f"{demand_name}_Occupancy_Trajectory.png"), dpi=300, bbox_inches='tight')
                    plt.close()

            if summary_df is not None:
                alpha_filter = (summary_df['Alpha'] == 0.0)
                for ta in target_alpha_group:
                    alpha_filter = alpha_filter | (summary_df['Alpha'] == ta)
                
                vkt_data = summary_df[
                    (summary_df['Demand'] == demand_name) &
                    (summary_df['Scenario'] == 'S2') &
                    alpha_filter
                ]
                
                if not vkt_data.empty:
                    vkt_data_plot = vkt_data.copy()
                    vkt_data_plot['Alpha'] = vkt_data_plot['Alpha'].apply(lambda x: f'α={x:.1f}' if x != 0.0 else 'α=0.0 (Baseline)')
                    vkt_colors = {'α=0.0 (Baseline)': 'lightcoral'}
                    for i, das in enumerate(display_alpha_strs):
                        vkt_colors[f'α={das}'] = target_color_light_list[i % len(target_color_light_list)]
                    plt.figure(figsize=(14 + 3 * (len(target_alpha_group) - 1), 10))
                    order = [f'α={das}' for das in display_alpha_strs] + ['α=0.0 (Baseline)']
                    order = [o for o in order if o in vkt_data_plot['Alpha'].unique()]

                    ax = sns.boxplot(data=vkt_data_plot, x='Alpha', y='Total_VKT', order=order, hue='Alpha',
                                     palette=vkt_colors, legend=False)
                    annotate_boxplot(ax, vkt_data_plot, 'Alpha', 'Total_VKT')
                    plt.ylabel('Total VKT (across all vehicles)', fontsize=28)
                    plt.xlabel('', fontsize=28)
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
                    plt.savefig(os.path.join(output_viz_dir, f"{demand_name}_VKT_Boxplot.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                    vkt_hist_colors = {'α=0.0 (Baseline)': 'red'}
                    for i, das in enumerate(display_alpha_strs):
                        vkt_hist_colors[f'α={das}'] = target_color_list[i % len(target_color_list)]
                    plt.figure(figsize=(16, 10))
                    sns.histplot(data=vkt_data_plot, x='Total_VKT', hue='Alpha', multiple='dodge', shrink=0.8,
                                 palette=vkt_hist_colors)
                    plt.xlabel('Total VKT (across all vehicles)', fontsize=28)
                    plt.ylabel('Count (Number of Seeds)', fontsize=28)
                    legend = plt.legend(fontsize=26, frameon=True, fancybox=False, edgecolor='black')
                    legend.get_frame().set_linewidth(1.5)
                    plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.98)
                    plt.savefig(os.path.join(output_viz_dir, f"{demand_name}_VKT_Distribution.png"), dpi=300, bbox_inches='tight')
                    plt.close()
            
    if user_alphas == 'all' and user_demand == 'all':
        print("\nDetailed comparison visualizations completed.")
        print("\nGenerated plot types:")
        print("  - OFV_Trajectory: cumulative delay and CVaR")
        print("  - CVaR_Ratio: CVaR-to-total delay ratio")
        print("  - CVaR_vs_CumulativePassengers: CVaR vs cumulative passengers")
        print("  - Delay/Detour distributions and boxplots")
        print("  - Lorenz Curve, CDF (equity)")
        print("  - Occupancy/VKT analysis")
    else:
        conditions = []
        if user_alphas != 'all':
            alpha_display = [f"{a:.1f}" for a in user_alphas]
            conditions.append(f"Alpha={alpha_display}")
        if user_demand != 'all':
            conditions.append(f"Demand={user_demand}")
        condition_text = ", ".join(conditions)
        print(f"\nDetailed comparison completed for: {condition_text}.")
        print("\nGenerated plot types:")
        print("  - OFV_Trajectory: cumulative delay and CVaR")
        print("  - CVaR_Ratio: CVaR-to-total delay ratio")
        print("  - CVaR_vs_CumulativePassengers: CVaR vs cumulative passengers")
        print("  - Delay/Detour distributions and boxplots")
        print("  - Lorenz Curve, CDF (equity)")
        print("  - Occupancy/VKT analysis")

if __name__ == '__main__':
    main() 