import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

"""
visualize_case.py
=================
Visualizes case distribution and alpha-vs-baseline outcomes for DARP simulation
sweep results. Alpha is used as in data: OFV = (1 - alpha)*Mean_Delay + alpha*Tail_Delay
(alpha=0.0 mean-only, alpha=1.0 CVaR-only; no display reversal).

Modes (interactive menu):
  1. Case Distribution: Stacked bar chart of Win-Win / Trade-Off / Similar / Other / Lose-Lose
     shares by alpha for a chosen demand level. Uses case_statistics_detailed.csv or recomputes
     from time_series_data_*.csv with configurable Similar threshold (5%, 10%, 15%, 20%).
  2. Alpha Scatter Plot: Scatter of relative delay change vs relative CVaR change (vs alpha=0.0)
     per seed, colored by alpha. Optional: plot for specific alphas (e.g. 0.3 only) or all 0.1~0.9.
  3. Both: Run (1) and (2) for selected demand(s).
  4. Passenger Timeline: Gantt-style plot of pickup/drop-off times per passenger for a given
     demand, alpha, and seed (seed input 1-based 1~50; data uses 0-based 0~49).

Inputs:
  - case_statistics_detailed.csv (from analyze_case_statistics.py) for mode 1/3.
  - Sweep directory (detailed_results/alpha_*/time_series_data_*.csv, results_s2_*.xlsx,
    passenger_records_*.csv) for scatter and passenger timeline.

Outputs: PNGs under case_statistics/visualizations/ (or same dir as stats file).

Usage: python visualize_case.py
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_statistics_file(base_dir=None):
    """Locate case_statistics_detailed.csv under base_dir (default: BASE_DIR/Alpha_Demand_Sweep_Results)."""
    search_base = base_dir if base_dir is not None else os.path.join(BASE_DIR, "Alpha_Demand_Sweep_Results")
    if not os.path.isdir(search_base):
        print(f"Warning: directory not found: '{search_base}'")
        return None
    for sweep_dir in sorted(glob.glob(os.path.join(search_base, "sweep_*")), key=os.path.basename, reverse=True):
        stats_file = os.path.join(sweep_dir, "case_statistics", "case_statistics_detailed.csv")
        if os.path.exists(stats_file):
            return stats_file
    print(f"Warning: case_statistics_detailed.csv not found under '{search_base}'")
    return None

def find_sweep_dir(base_dir=None):
    """Locate latest sweep directory under base_dir (default: BASE_DIR/Alpha_Demand_Sweep_Results)."""
    search_base = base_dir if base_dir is not None else os.path.join(BASE_DIR, "Alpha_Demand_Sweep_Results")
    if not os.path.isdir(search_base):
        print(f"Warning: directory not found: '{search_base}'")
        return None
    sweep_dirs = glob.glob(os.path.join(search_base, "sweep_*"))
    if not sweep_dirs:
        print(f"Warning: no sweep_* directory found under '{search_base}'")
        return None
    return max(sweep_dirs, key=lambda p: os.path.basename(p))

def recalculate_all_cases_from_raw_data(sweep_dir, demand_level, threshold=0.1):
    """Recalculate Win-Win/Trade-Off/Lose-Lose/Similar/Other counts from raw time_series_data using given Similar threshold. Returns dict alpha -> counts."""
    detailed_results_dir = os.path.join(sweep_dir, "detailed_results")
    alpha_0_dir = os.path.join(detailed_results_dir, "alpha_0.0")
    baseline_file = os.path.join(alpha_0_dir, f"time_series_data_{demand_level}.csv")
    
    if not os.path.exists(baseline_file):
        return {}
    df_baseline = pd.read_csv(baseline_file)
    rename_map = {}
    if 'CumulativeDelaySum' in df_baseline.columns and 'S2_SumDelay' not in df_baseline.columns:
        rename_map['CumulativeDelaySum'] = 'S2_SumDelay'
    if 'CumulativeTailDelaySum' in df_baseline.columns and 'S2_TailSumDelay' not in df_baseline.columns:
        rename_map['CumulativeTailDelaySum'] = 'S2_TailSumDelay'
    if rename_map:
        df_baseline = df_baseline.rename(columns=rename_map)
    
    if 'Seed' not in df_baseline.columns:
        df_baseline['Seed'] = 0
    
    baseline_values = {}
    for seed in df_baseline['Seed'].unique():
        df_seed = df_baseline[df_baseline['Seed'] == seed]
        if not df_seed.empty:
            baseline_values[seed] = {
                'delay': df_seed['S2_SumDelay'].iloc[-1],
                'cvar': df_seed['S2_TailSumDelay'].iloc[-1]
            }
    target_alphas = [round(0.1 * i, 1) for i in range(1, 10)]
    case_counts_all = {}
    
    for alpha in target_alphas:
        alpha_str = f"{alpha:.1f}"
        alpha_dir = os.path.join(detailed_results_dir, f"alpha_{alpha_str}")
        target_file = os.path.join(alpha_dir, f"time_series_data_{demand_level}.csv")
        
        if not os.path.exists(target_file):
            case_counts_all[alpha] = {
                'Win-Win': 0, 'Trade-Off': 0, 'Lose-Lose': 0,
                'Similar': 0, 'Other': 0, 'Total_Seeds': 0
            }
            continue
        
        df_target = pd.read_csv(target_file)
        
        rename_map = {}
        if 'CumulativeDelaySum' in df_target.columns and 'S2_SumDelay' not in df_target.columns:
            rename_map['CumulativeDelaySum'] = 'S2_SumDelay'
        if 'CumulativeTailDelaySum' in df_target.columns and 'S2_TailSumDelay' not in df_target.columns:
            rename_map['CumulativeTailDelaySum'] = 'S2_TailSumDelay'
        if rename_map:
            df_target = df_target.rename(columns=rename_map)
        
        if 'Seed' not in df_target.columns:
            df_target['Seed'] = 0
        
        case_counts = {
            'Win-Win': 0, 'Trade-Off': 0, 'Lose-Lose': 0,
            'Similar': 0, 'Other': 0, 'Total_Seeds': 0
        }
        
        for seed in df_target['Seed'].unique():
            df_seed = df_target[df_target['Seed'] == seed]
            if not df_seed.empty and seed in baseline_values:
                target_delay = df_seed['S2_SumDelay'].iloc[-1]
                target_cvar = df_seed['S2_TailSumDelay'].iloc[-1]
                
                baseline_delay = baseline_values[seed]['delay']
                baseline_cvar = baseline_values[seed]['cvar']
                if baseline_delay > 0:
                    delay_ratio = (target_delay - baseline_delay) / baseline_delay
                else:
                    delay_ratio = 0.0
                
                if baseline_cvar > 0:
                    cvar_ratio = (target_cvar - baseline_cvar) / baseline_cvar
                else:
                    cvar_ratio = 0.0
                
                delay_similar = -threshold <= delay_ratio <= threshold
                cvar_similar = -threshold <= cvar_ratio <= threshold
                
                if delay_similar and cvar_similar:
                    case_counts['Similar'] += 1
                else:
                    delay_improved = target_delay < baseline_delay
                    cvar_improved = target_cvar < baseline_cvar
                    
                    if delay_improved and cvar_improved:
                        case_counts['Win-Win'] += 1
                    elif not delay_improved and cvar_improved:
                        case_counts['Trade-Off'] += 1
                    elif not delay_improved and not cvar_improved:
                        case_counts['Lose-Lose'] += 1
                    else:
                        case_counts['Other'] += 1
                
                case_counts['Total_Seeds'] += 1
        
        case_counts_all[alpha] = case_counts
    
    return case_counts_all

def plot_case_distribution(df_stats, demand_level, output_dir, sweep_dir=None, similar_threshold=0.1):
    """Plot stacked bar of Win-Win/Trade-Off/Similar/Other/Lose-Lose share by alpha for one demand level. sweep_dir: optional for recomputing Similar from raw data."""
    df_demand = df_stats[df_stats['Demand'] == demand_level].copy()
    if df_demand.empty:
        print(f"Warning: no data for '{demand_level}'.")
        return
    
    df_demand['Display_Alpha'] = df_demand['Alpha']
    df_demand = df_demand.sort_values('Display_Alpha')
    if sweep_dir is not None:
        case_counts_all = recalculate_all_cases_from_raw_data(sweep_dir, demand_level, similar_threshold)
        for idx, row in df_demand.iterrows():
            alpha = row['Alpha']
            if alpha in case_counts_all:
                counts = case_counts_all[alpha]
                df_demand.at[idx, 'Win-Win'] = counts['Win-Win']
                df_demand.at[idx, 'Trade-Off'] = counts['Trade-Off']
                df_demand.at[idx, 'Lose-Lose'] = counts['Lose-Lose']
                df_demand.at[idx, 'Similar'] = counts['Similar']
                df_demand.at[idx, 'Other'] = counts['Other']
                df_demand.at[idx, 'Total_Seeds'] = counts['Total_Seeds']
    else:
        if 'Similar' not in df_demand.columns:
            df_demand['Similar'] = 0
        if 'Other' not in df_demand.columns:
            df_demand['Other'] = 0
    
    df_demand['WinWin_Pct'] = df_demand['Win-Win'] / df_demand['Total_Seeds'] * 100
    df_demand['TradeOff_Pct'] = df_demand['Trade-Off'] / df_demand['Total_Seeds'] * 100
    df_demand['LoseLose_Pct'] = df_demand['Lose-Lose'] / df_demand['Total_Seeds'] * 100
    df_demand['Similar_Pct'] = df_demand['Similar'] / df_demand['Total_Seeds'] * 100
    if 'Other' in df_demand.columns:
        df_demand['Other_Pct'] = df_demand['Other'] / df_demand['Total_Seeds'] * 100
    else:
        df_demand['Other_Pct'] = 0.0
    
    alphas = df_demand['Display_Alpha'].values
    winwin_pct = df_demand['WinWin_Pct'].values
    tradeoff_pct = df_demand['TradeOff_Pct'].values
    loselose_pct = df_demand['LoseLose_Pct'].values
    similar_pct = df_demand['Similar_Pct'].values
    other_pct = df_demand['Other_Pct'].values
    fig, ax = plt.subplots(figsize=(16, 10))
    bar_width = 0.08
    x_positions = np.arange(len(alphas))
    
    bottom1 = np.zeros(len(alphas))
    bars1 = ax.bar(x_positions, winwin_pct, bar_width * 5, 
                   label='Win-Win (Delay↓ & CVaR↓)', 
                   color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    bottom2 = winwin_pct
    bars2 = ax.bar(x_positions, tradeoff_pct, bar_width * 5, 
                   bottom=bottom2,
                   label='Trade-Off (Delay↑ but CVaR↓)', 
                   color='#f39c12', edgecolor='black', linewidth=1.5)
    
    bottom3 = winwin_pct + tradeoff_pct
    threshold_pct = int(similar_threshold * 100)
    bars3 = ax.bar(x_positions, similar_pct, bar_width * 5, 
                   bottom=bottom3,
                   label=f'Similar (|Change| ≤ {threshold_pct}%)', 
                   color='#95a5a6', edgecolor='black', linewidth=1.5)
    
    bottom4 = winwin_pct + tradeoff_pct + similar_pct
    bars4 = ax.bar(x_positions, other_pct, bar_width * 5, 
                   bottom=bottom4,
                   label='Other (Delay↓ but CVaR↑)', 
                   color='#9b59b6', edgecolor='black', linewidth=1.5)
    
    bottom5 = winwin_pct + tradeoff_pct + similar_pct + other_pct
    bars5 = ax.bar(x_positions, loselose_pct, bar_width * 5, 
                   bottom=bottom5,
                   label='Lose-Lose (Delay↑ & CVaR↑)', 
                   color='#e74c3c', edgecolor='black', linewidth=1.5)
    for i, (alpha, ww, to, sim, oth, ll) in enumerate(zip(alphas, winwin_pct, tradeoff_pct, similar_pct, other_pct, loselose_pct)):
        if ww > 3:
            ax.text(i, ww/2, f'{ww:.1f}%', 
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        if to > 3:
            ax.text(i, ww + to/2, f'{to:.1f}%', 
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        if sim > 3:
            ax.text(i, ww + to + sim/2, f'{sim:.1f}%', 
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        
        if oth > 3:
            ax.text(i, ww + to + sim + oth/2, f'{oth:.1f}%', 
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        if ll > 3:
            ax.text(i, ww + to + sim + oth + ll/2, f'{ll:.1f}%', 
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('Alpha (α)', fontsize=28, fontfamily='Times New Roman')
    ax.set_ylabel('Percentage of Seeds (%)', fontsize=28, fontfamily='Times New Roman')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{a:.1f}' for a in alphas], fontsize=24)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_yticklabels([f'{y}%' for y in range(0, 101, 10)], fontsize=24)
    ax.set_ylim(0, 100)
    legend = ax.legend(fontsize=22, loc='upper left', frameon=True, 
                      fancybox=False, edgecolor='black')
    legend.get_frame().set_linewidth(2.0)
    ax.grid(True, axis='y', alpha=0.3, linewidth=1, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.97, top=0.97)
    threshold_pct = int(similar_threshold * 100)
    output_file = os.path.join(output_dir, f'case_distribution_{demand_level}_threshold{threshold_pct}pct.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file} (Similar threshold: {threshold_pct}%)")
    plt.close()
    threshold_pct = int(similar_threshold * 100)
    print(f"\nCase distribution for {demand_level} (Similar threshold: {threshold_pct}%):")
    print("-" * 80)
    for i, (display_alpha, ww, to, sim, oth, ll, total) in enumerate(zip(
        df_demand['Display_Alpha'], df_demand['Win-Win'], 
        df_demand['Trade-Off'], 
        df_demand.get('Similar', pd.Series([0]*len(df_demand))),
        df_demand.get('Other', pd.Series([0]*len(df_demand))),
        df_demand['Lose-Lose'],
        df_demand['Total_Seeds'])):
        print(f"α={display_alpha:.1f}: Win-Win {ww:2d}/{total} ({ww/total*100:5.1f}%)  |  "
              f"Trade-Off {to:2d}/{total} ({to/total*100:5.1f}%)  |  "
              f"Similar {sim:2d}/{total} ({sim/total*100:5.1f}%)  |  "
              f"Other {oth:2d}/{total} ({oth/total*100:5.1f}%)  |  "
              f"Lose-Lose {ll:2d}/{total} ({ll/total*100:5.1f}%)")

def plot_alpha_scatter(sweep_dir, demand_level, output_dir, target_display_alphas=None):
    """Scatter plot of delay and CVaR change by Alpha vs baseline (alpha=0.0)."""
    detailed_results_dir = os.path.join(sweep_dir, "detailed_results")
    alpha_0_dir = os.path.join(detailed_results_dir, "alpha_0.0")
    baseline_file = os.path.join(alpha_0_dir, f"time_series_data_{demand_level}.csv")
    
    if not os.path.exists(baseline_file):
        print("Warning: baseline file not found (alpha_0.0).")
        return
    df_baseline = pd.read_csv(baseline_file)
    
    rename_map = {}
    if 'CumulativeDelaySum' in df_baseline.columns and 'S2_SumDelay' not in df_baseline.columns:
        rename_map['CumulativeDelaySum'] = 'S2_SumDelay'
    if 'CumulativeTailDelaySum' in df_baseline.columns and 'S2_TailSumDelay' not in df_baseline.columns:
        rename_map['CumulativeTailDelaySum'] = 'S2_TailSumDelay'
    if rename_map:
        df_baseline = df_baseline.rename(columns=rename_map)
    if 'Seed' not in df_baseline.columns:
        df_baseline['Seed'] = 0
    
    baseline_values = {}
    for seed in df_baseline['Seed'].unique():
        df_seed = df_baseline[df_baseline['Seed'] == seed]
        if not df_seed.empty:
            baseline_values[seed] = {
                'delay': df_seed['S2_SumDelay'].iloc[-1],
                'cvar': df_seed['S2_TailSumDelay'].iloc[-1]
            }
    
    default_alphas = [round(0.1 * i, 1) for i in range(1, 10)]
    if target_display_alphas is None:
        target_alphas = default_alphas
    else:
        converted = []
        for a in target_display_alphas:
            val = round(float(a), 1)
            if val in default_alphas:
                converted.append(val)
            else:
                print(f"Warning: Alpha {a} not in 0.1~0.9; ignoring.")
        target_alphas = sorted(set(converted)) if converted else default_alphas
        if not target_alphas:
            print("Warning: no valid Alpha; using default 0.1~0.9.")
            target_alphas = default_alphas
    display_alphas = list(target_alphas)
    scatter_data = []
    for alpha_val in target_alphas:
        alpha_str = f"{alpha_val:.1f}"
        alpha_dir = os.path.join(detailed_results_dir, f"alpha_{alpha_str}")
        target_file = os.path.join(alpha_dir, f"time_series_data_{demand_level}.csv")
        
        if not os.path.exists(target_file):
            print(f"Warning: Alpha {alpha_str} file not found; skipping.")
            continue
        df_target = pd.read_csv(target_file)
        
        rename_map = {}
        if 'CumulativeDelaySum' in df_target.columns and 'S2_SumDelay' not in df_target.columns:
            rename_map['CumulativeDelaySum'] = 'S2_SumDelay'
        if 'CumulativeTailDelaySum' in df_target.columns and 'S2_TailSumDelay' not in df_target.columns:
            rename_map['CumulativeTailDelaySum'] = 'S2_TailSumDelay'
        if rename_map:
            df_target = df_target.rename(columns=rename_map)
        if 'Seed' not in df_target.columns:
            df_target['Seed'] = 0
        for seed in df_target['Seed'].unique():
            df_seed = df_target[df_target['Seed'] == seed]
            if not df_seed.empty and seed in baseline_values:
                target_delay = df_seed['S2_SumDelay'].iloc[-1]
                target_cvar = df_seed['S2_TailSumDelay'].iloc[-1]
                
                baseline_delay = baseline_values[seed]['delay']
                baseline_cvar = baseline_values[seed]['cvar']
                if baseline_delay > 0:
                    delay_ratio = (target_delay - baseline_delay) / baseline_delay
                else:
                    delay_ratio = np.nan
                
                if baseline_cvar > 0:
                    cvar_ratio = (target_cvar - baseline_cvar) / baseline_cvar
                else:
                    cvar_ratio = np.nan
                
                if not (np.isnan(delay_ratio) or np.isnan(cvar_ratio)):
                    scatter_data.append({
                        'alpha': alpha_val,
                        'seed': seed,
                        'delay_ratio': delay_ratio,
                        'cvar_ratio': cvar_ratio
                    })
    
    if not scatter_data:
        print(f"Warning: no scatter data for '{demand_level}'.")
        return
    df_scatter = pd.DataFrame(scatter_data)
    fig, ax = plt.subplots(figsize=(16, 10))
    min_alpha = min(display_alphas)
    max_alpha = max(display_alphas)
    alpha_range = max_alpha - min_alpha
    colormap = plt.cm.viridis
    normalized_alphas = [(a - min_alpha) / alpha_range if alpha_range > 0 else 0.5 
                         for a in df_scatter['alpha']]
    scatter = ax.scatter(df_scatter['delay_ratio'], df_scatter['cvar_ratio'],
                        c=normalized_alphas, cmap=colormap, 
                        s=120, alpha=0.7, edgecolors='black', linewidths=0.8,
                        vmin=0, vmax=1)
    unique_display_alphas = sorted(df_scatter['alpha'].unique())
    if len(unique_display_alphas) > 1:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Alpha (α)', fontsize=24, fontfamily='Times New Roman')
        cbar.ax.tick_params(labelsize=20)
        cbar_ticks = np.linspace(0, 1, len(unique_display_alphas))
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f'{a:.1f}' for a in unique_display_alphas])
    else:
        single_alpha = unique_display_alphas[0]
        ax.text(0.02, 0.98, f'α={single_alpha:.1f}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=22, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    alpha_means = []
    for display_alpha in sorted(display_alphas):
        df_alpha = df_scatter[df_scatter['alpha'] == display_alpha]
        if not df_alpha.empty:
            delay_mean = df_alpha['delay_ratio'].mean()
            cvar_mean = df_alpha['cvar_ratio'].mean()
            alpha_means.append({
                'alpha': display_alpha,
                'delay_mean': delay_mean,
                'cvar_mean': cvar_mean
            })
    ax.set_xlabel('Relative Delay Change (vs α=0.0)', fontsize=28, fontfamily='Times New Roman')
    ax.set_ylabel('Relative CVaR Change (vs α=0.0)', fontsize=28, fontfamily='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.grid(True, alpha=0.3, linewidth=1, linestyle='--')
    ax.set_axisbelow(True)
    ax.axhline(y=0.0, color='gray', linestyle='-', linewidth=2, alpha=0.7)
    ax.axvline(x=0.0, color='gray', linestyle='-', linewidth=2, alpha=0.7)
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.95)
    output_file = os.path.join(output_dir, f'alpha_scatter_{demand_level}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    print(f"\nRelative change stats for {demand_level} (vs α=0.0):")
    print("=" * 80)
    for display_alpha in sorted(display_alphas):
        df_alpha = df_scatter[df_scatter['alpha'] == display_alpha]
        if not df_alpha.empty:
            delay_ratio_mean = df_alpha['delay_ratio'].mean()
            cvar_ratio_mean = df_alpha['cvar_ratio'].mean()
            delay_ratio_median = df_alpha['delay_ratio'].median()
            cvar_ratio_median = df_alpha['cvar_ratio'].median()
            print(f"α={display_alpha:.1f}:")
            print(f"  Delay change: mean={delay_ratio_mean:.3f} ({delay_ratio_mean*100:+.1f}%), median={delay_ratio_median:.3f} ({delay_ratio_median*100:+.1f}%)")
            print(f"  CVaR change: mean={cvar_ratio_mean:.3f} ({cvar_ratio_mean*100:+.1f}%), median={cvar_ratio_median:.3f} ({cvar_ratio_median*100:+.1f}%)")
            print(f"  Seeds: {len(df_alpha)}")
            print()
    print("=" * 80)
    print("Trend analysis by Alpha:")
    print("-" * 80)
    
    if len(alpha_means) > 1:
        alpha_means_df = pd.DataFrame(alpha_means).sort_values('alpha')
        z_delay = np.polyfit(alpha_means_df['alpha'], alpha_means_df['delay_mean'], 1)
        z_cvar = np.polyfit(alpha_means_df['alpha'], alpha_means_df['cvar_mean'], 1)
        corr_delay = np.corrcoef(alpha_means_df['alpha'], alpha_means_df['delay_mean'])[0, 1]
        slope_delay = z_delay[0]
        corr_cvar = np.corrcoef(alpha_means_df['alpha'], alpha_means_df['cvar_mean'])[0, 1]
        slope_cvar = z_cvar[0]
        print(f"1. Delay change vs Alpha: corr={corr_delay:.3f}, slope={slope_delay:.4f}")
        if slope_delay > 0:
            print("   -> Higher alpha -> higher delay change (right)")
        elif slope_delay < 0:
            print("   -> Higher alpha -> lower delay change (left)")
        else:
            print("   -> No clear trend")
        print(f"\n2. CVaR change vs Alpha: corr={corr_cvar:.3f}, slope={slope_cvar:.4f}")
        if slope_cvar > 0:
            print("   -> Higher alpha -> higher CVaR change (up)")
        elif slope_cvar < 0:
            print("   -> Higher alpha -> lower CVaR change (down)")
        else:
            print("   -> No clear trend")
        print("\n3. Summary:")
        if slope_delay > 0 and slope_cvar < 0:
            print("   -> Trade-off: delay worsens, CVaR improves")
        elif slope_delay > 0 and slope_cvar > 0:
            print("   -> Lose-Lose: both worsen")
        elif slope_delay < 0 and slope_cvar < 0:
            print("   -> Win-Win: both improve")
        elif slope_delay < 0 and slope_cvar > 0:
            print("   -> Reverse trade-off: delay improves, CVaR worsens")
        print("\n4. Mean position by Alpha:")
        for idx, row in alpha_means_df.iterrows():
            print(f"   α={row['alpha']:.1f}: Delay={row['delay_mean']:.3f}, CVaR={row['cvar_mean']:.3f}")

def plot_all_demands(df_stats, output_dir, sweep_dir=None):
    """Run case distribution visualization for all demand levels and Similar thresholds (5%, 10%, 15%, 20%)."""
    demand_levels = sorted(df_stats['Demand'].unique())
    thresholds = [0.05, 0.1, 0.15, 0.2]
    print(f"\nDemand levels: {list(demand_levels)}")
    print(f"Generating visualizations for {len(demand_levels)} demand level(s).")
    print(f"Similar thresholds: {[f'{int(t*100)}%' for t in thresholds]}\n")
    
    for demand in demand_levels:
        for threshold in thresholds:
            print(f"\n{'='*60}")
            print(f"Visualizing: {demand} (Similar threshold: {int(threshold*100)}%)")
            print('='*60)
            plot_case_distribution(df_stats, demand, output_dir, sweep_dir, threshold)

def plot_passenger_timeline(sweep_dir, demand_level, input_alpha, seed, output_dir):
    """Plot passenger pickup/drop-off timeline (Gantt-style) for given demand, alpha, seed. Seed input 1-based (1~50); data uses 0-based (0~49). Alpha as in data (no reversal)."""
    actual_alpha = round(float(input_alpha), 1)
    alpha_str = f"{actual_alpha:.1f}"
    
    detailed_results_dir = os.path.join(sweep_dir, "detailed_results")
    alpha_dir = os.path.join(detailed_results_dir, f"alpha_{alpha_str}")
    passenger_records_file = os.path.join(alpha_dir, f"passenger_records_{demand_level}.csv")
    passenger_data = None
    seed_in_data = seed - 1
    if os.path.exists(passenger_records_file):
        print(f"Loading passenger data: {passenger_records_file}")
        df_all = pd.read_csv(passenger_records_file)
        if 'Seed' in df_all.columns:
            passenger_data = df_all[df_all['Seed'] == seed_in_data].copy()
            if passenger_data.empty:
                print(f"Warning: no data for Seed={seed_in_data} (input seed={seed}). Available: {sorted(df_all['Seed'].unique())}")
        else:
            if seed == 1:
                passenger_data = df_all.copy()
            else:
                print(f"Warning: passenger_records has no Seed column; seed={seed} requested.")
                passenger_data = None
    if passenger_data is None or passenger_data.empty:
        s2_file = os.path.join(alpha_dir, f"results_s2_{demand_level}.xlsx")
        if os.path.exists(s2_file):
            sheet_name = f'seed_{seed_in_data}'
            print(f"Loading passenger data: {s2_file}, sheet={sheet_name} (input Seed={seed})")
            try:
                passenger_data = pd.read_excel(s2_file, sheet_name=sheet_name)
                print(f"  Loaded sheet '{sheet_name}'")
            except Exception as e:
                print(f"  Failed to load sheet '{sheet_name}': {e}")
                try:
                    passenger_data = pd.read_excel(s2_file, sheet_name=seed_in_data)
                    print(f"  Loaded by index {seed_in_data}")
                except Exception as e2:
                    print(f"  Failed by index: {e2}")
                    try:
                        xl_file = pd.ExcelFile(s2_file)
                        print(f"  Available sheets: {xl_file.sheet_names[:10]}...")
                    except Exception:
                        pass
                    passenger_data = None
    if passenger_data is None or passenger_data.empty:
        print(f"Error: no passenger data for {demand_level}, α={actual_alpha}, seed={seed}.")
        print("Checked files:")
        print(f"  - {passenger_records_file}")
        print(f"  - {s2_file}")
        return
    boarding_col = None
    alighting_col = None
    passenger_id_col = None
    boarding_candidates = ['BoardingTime', 'Boarding_Time', 'boarding_time', 'PickupTime', 'pickup_time']
    alighting_candidates = ['AlightingTime', 'Alighting_Time', 'alighting_time', 'DropoffTime', 'dropoff_time', 'DeliveryTime']
    id_candidates = ['PassengerID', 'Passenger_ID', 'passenger_id', 'ID', 'id', 'PassID']
    
    for col in passenger_data.columns:
        if col in boarding_candidates:
            boarding_col = col
        if col in alighting_candidates:
            alighting_col = col
        if col in id_candidates:
            passenger_id_col = col
    if boarding_col is None:
        for col in passenger_data.columns:
            if 'board' in col.lower() or 'pickup' in col.lower():
                boarding_col = col
                break
    
    if alighting_col is None:
        for col in passenger_data.columns:
            if 'alight' in col.lower() or 'dropoff' in col.lower() or 'delivery' in col.lower():
                alighting_col = col
                break
    
    if passenger_id_col is None:
        for col in passenger_data.columns:
            if 'passenger' in col.lower() or 'pass' in col.lower() or col.lower() == 'id':
                passenger_id_col = col
                break
    if boarding_col is None or alighting_col is None:
        print("Error: boarding/alighting time columns not found.")
        print(f"Available columns: {list(passenger_data.columns)}")
        return
    if passenger_id_col is None:
        passenger_data['PassengerID'] = range(len(passenger_data))
        passenger_id_col = 'PassengerID'
    print(f"Using columns: ID={passenger_id_col}, boarding={boarding_col}, alighting={alighting_col}")
    passenger_data = passenger_data.dropna(subset=[boarding_col, alighting_col])
    if passenger_data.empty:
        print("Warning: no valid passenger data.")
        return
    passenger_data = passenger_data.sort_values(boarding_col)
    fig, ax = plt.subplots(figsize=(18, max(10, len(passenger_data) * 0.15)))
    for idx, row in passenger_data.iterrows():
        passenger_id = row[passenger_id_col]
        boarding_time = row[boarding_col]
        alighting_time = row[alighting_col]
        ax.plot([boarding_time, alighting_time], [passenger_id, passenger_id],
               linewidth=2.5, color='#3498db', alpha=0.7, solid_capstyle='round')
        ax.scatter([boarding_time], [passenger_id], s=80, color='#2ecc71', 
                  marker='o', edgecolors='black', linewidths=1, zorder=3, label='Pickup Time' if idx == 0 else '')
        ax.scatter([alighting_time], [passenger_id], s=80, color='#e74c3c', 
                  marker='s', edgecolors='black', linewidths=1, zorder=3, label='Drop-off Time' if idx == 0 else '')
    ax.set_xlabel('Time (minutes)', fontsize=24, fontfamily='Times New Roman')
    ax.set_ylabel('Passenger ID', fontsize=24, fontfamily='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=18)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(by_label.values(), by_label.keys(), fontsize=36, loc='upper left',
                      frameon=True, fancybox=False, edgecolor='black',
                      markerscale=2.0, handlelength=2.0, handletextpad=1.0)
    legend.get_frame().set_linewidth(2.0)
    legend.get_frame().set_boxstyle('round', pad=0.4)
    ax.grid(True, alpha=0.3, linewidth=1, linestyle='--', axis='x')
    ax.set_axisbelow(True)
    if len(passenger_data) > 0:
        y_min = passenger_data[passenger_id_col].min() - 1
        y_max = passenger_data[passenger_id_col].max() + 1
        ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'passenger_timeline_{demand_level}_alpha{actual_alpha}_seed{seed}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    print(f"\nPassenger stats: {demand_level}, α={actual_alpha}, Seed={seed}")
    print("=" * 80)
    print(f"Total passengers: {len(passenger_data)}")
    print(f"Mean boarding time: {passenger_data[boarding_col].mean():.2f} min")
    print(f"Mean alighting time: {passenger_data[alighting_col].mean():.2f} min")
    print(f"Mean ride time: {(passenger_data[alighting_col] - passenger_data[boarding_col]).mean():.2f} min")
    print(f"Time range: {passenger_data[boarding_col].min():.2f} ~ {passenger_data[alighting_col].max():.2f} min")

def plot_all_demands_scatter(sweep_dir, output_dir):
    """Run scatter plot for all demand levels (Alpha 0.1~0.9)."""
    detailed_results_dir = os.path.join(sweep_dir, "detailed_results")
    alpha_0_dir = os.path.join(detailed_results_dir, "alpha_0.0")
    baseline_dir = alpha_0_dir
    if not os.path.exists(alpha_0_dir):
        print("Warning: Alpha 0.0 directory not found.")
        return
    files = os.listdir(baseline_dir)
    demand_levels = []
    for f in files:
        if f.startswith('time_series_data_') and f.endswith('.csv'):
            demand = f.replace('time_series_data_', '').replace('.csv', '')
            demand_levels.append(demand)
    
    demand_levels = sorted(demand_levels)
    print(f"\nDemand levels: {list(demand_levels)}")
    print(f"Generating scatter plots for {len(demand_levels)} demand level(s).\n")
    for demand in demand_levels:
        print(f"\n{'='*60}")
        print(f"Scatter plot: {demand}")
        print('='*60)
        plot_alpha_scatter(sweep_dir, demand, output_dir)

def main():
    """Interactive menu: load stats, choose visualization type (1=Case Dist, 2=Scatter, 3=Both, 4=Passenger Timeline), then run."""
    print("=" * 80)
    print("Case distribution visualization")
    print("=" * 80)
    stats_file = find_statistics_file()
    if not stats_file:
        print("Error: statistics file not found. Run 'analyze_case_statistics.py' first.")
        return
    print(f"Loaded: {stats_file}")
    df_stats = pd.read_csv(stats_file)
    output_dir = os.path.dirname(stats_file)
    output_viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(output_viz_dir, exist_ok=True)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    sweep_dir = find_sweep_dir()
    print("\nVisualization type:")
    print("  1. Case Distribution (stacked bar)")
    print("  2. Alpha Scatter Plot (delay/CVaR change vs α=0.0)")
    print("  3. Both (1 + 2)")
    print("  4. Passenger Timeline (pickup/drop-off Gantt)")
    print("\n" + "="*80)
    viz_type = input("Choose (1, 2, 3, 4): ").strip()
    print("="*80)
    if viz_type not in ['1', '2', '3', '4']:
        print("Warning: invalid choice. Enter 1, 2, 3, or 4.")
        return
    if viz_type == '4':
        if not sweep_dir:
            print("Warning: sweep directory not found; cannot create Passenger Timeline.")
            return
        print("\nDemand levels:")
        demand_levels = sorted(df_stats['Demand'].unique())
        for i, demand in enumerate(demand_levels, 1):
            print(f"  {i}. {demand}")
        print("\n" + "="*80)
        user_input = input("Demand level (number or name, e.g. 1, demand_1x): ").strip()
        print("="*80)
        selected_demand = None
        user_input_lower = user_input.lower()
        
        if user_input.isdigit():
            input_num = int(user_input)
            if 1 <= input_num <= len(demand_levels):
                selected_demand = demand_levels[input_num - 1]
            else:
                print(f"\nWarning: invalid number. Enter 1~{len(demand_levels)}.")
                return
        elif user_input_lower in [d.lower() for d in demand_levels]:
            selected_demand = [d for d in demand_levels if d.lower() == user_input_lower][0]
        else:
            print(f"\nWarning: invalid input '{user_input}'.")
            return
        print("\n" + "="*80)
        print("Alpha (data α, OFV: 0.0=mean-only, 1.0=CVaR-only)")
        alpha_input = input("Alpha (0.0~1.0): ").strip()
        print("="*80)
        try:
            selected_alpha = round(float(alpha_input), 1)
            if selected_alpha < 0.0 or selected_alpha > 1.0:
                print(f"Warning: Alpha must be 0.0~1.0. Input: {alpha_input}")
                return
        except ValueError:
            print(f"Warning: '{alpha_input}' is not a number.")
            return
        print("\n" + "="*80)
        print("Seed (1-based 1~50; data uses 0-based 0~49)")
        seed_input = input("Seed (1~50): ").strip()
        print("="*80)
        try:
            selected_seed = int(seed_input)
            if selected_seed < 1 or selected_seed > 50:
                print("Warning: Seed must be 1~50.")
                return
        except ValueError:
            print(f"Warning: '{seed_input}' is not an integer.")
            return
        print(f"\n{'='*60}")
        print(f"Creating Passenger Timeline: {selected_demand}, α={selected_alpha}, Seed={selected_seed}")
        print('='*60)
        plot_passenger_timeline(sweep_dir, selected_demand, selected_alpha, selected_seed, output_viz_dir)
        print("\n" + "=" * 80)
        print("Done. Output folder:")
        print(f"  {output_viz_dir}")
        print("=" * 80)
        return
    print("\nDemand levels:")
    demand_levels = sorted(df_stats['Demand'].unique())
    for i, demand in enumerate(demand_levels, 1):
        print(f"  {i}. {demand}")
    all_option_num = len(demand_levels) + 1
    print(f"  {all_option_num}. all")
    print("\n" + "="*80)
    user_input = input("Demand (number, name, or 'all'): ").strip()
    print("="*80)
    selected_demand = None
    user_input_lower = user_input.lower()
    if user_input.isdigit():
        input_num = int(user_input)
        if 1 <= input_num <= len(demand_levels):
            selected_demand = demand_levels[input_num - 1]
        elif input_num == all_option_num:
            selected_demand = 'all'
        else:
            print(f"\nWarning: invalid number. Enter 1~{all_option_num}.")
            return
    elif user_input_lower == 'all':
        selected_demand = 'all'
    elif user_input_lower in [d.lower() for d in demand_levels]:
        selected_demand = [d for d in demand_levels if d.lower() == user_input_lower][0]
    else:
        print(f"\nWarning: invalid input '{user_input}'. Options: 1~{all_option_num}, {list(demand_levels)}, or 'all'")
        return
    if viz_type in ['1', '3']:
        thresholds = [0.05, 0.1, 0.15, 0.2]
        if selected_demand == 'all':
            plot_all_demands(df_stats, output_viz_dir, sweep_dir)
        else:
            for threshold in thresholds:
                print(f"\n{'='*60}")
                print(f"Case Distribution: {selected_demand} (Similar threshold: {int(threshold*100)}%)")
                print('='*60)
                plot_case_distribution(df_stats, selected_demand, output_viz_dir, sweep_dir, threshold)
    if viz_type in ['2', '3']:
        if not sweep_dir:
            print("Warning: sweep directory not found; cannot create scatter plot.")
        else:
            if selected_demand == 'all':
                plot_all_demands_scatter(sweep_dir, output_viz_dir)
            else:
                print("\n" + "="*80)
                print("Alpha for scatter (data α). all = 0.1~0.9, or e.g. 0.3 or 0.3,0.5,0.7")
                alpha_input = input("Alpha (default: all): ").strip().lower()
                print("="*80)
                default_alphas = [round(0.1 * i, 1) for i in range(1, 10)]
                selected_alphas = default_alphas
                if alpha_input and alpha_input != 'all':
                    raw_tokens = alpha_input.replace(',', ' ').split()
                    tmp = []
                    for tok in raw_tokens:
                        try:
                            val = round(float(tok), 1)
                            if val in default_alphas:
                                tmp.append(val)
                            else:
                                print(f"Warning: Alpha {tok} not in 0.1~0.9; ignoring.")
                        except ValueError:
                            print(f"Warning: '{tok}' is not a number; ignoring.")
                    tmp = sorted(set(tmp))
                    if not tmp:
                        print("Warning: no valid Alpha; using 0.1~0.9.")
                        selected_alphas = default_alphas
                    else:
                        selected_alphas = tmp
                print(f"\n{'='*60}")
                print(f"Scatter plot: {selected_demand}, Alpha: {selected_alphas}")
                print('='*60)
                plot_alpha_scatter(sweep_dir, selected_demand, output_viz_dir,
                                   target_display_alphas=selected_alphas)
    print("\n" + "=" * 80)
    print("Done. Output folder:")
    print(f"  {output_viz_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()

