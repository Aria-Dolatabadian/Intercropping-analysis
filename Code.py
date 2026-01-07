# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # Set style for better-looking plots
# sns.set_style("whitegrid")
# plt.rcParams['figure.figsize'] = (12, 8)
#
#
# # ============================================================================
# # PART 1: DATA LOADING
# # ============================================================================
#
# def load_intercropping_data(input_file='intercropping_data.csv'):
#     """
#     Load intercropping data from CSV file
#     """
#     try:
#         df = pd.read_csv(input_file)
#         print(f"✓ Data loaded successfully from '{input_file}'")
#         print(f"  Total records: {len(df)}")
#         print(f"  Columns: {', '.join(df.columns)}")
#
#         # Calculate metrics if not already present
#         if 'LER' not in df.columns:
#             print("\n  Computing LER, TOI, RYT, and pLER metrics...")
#             df['LER'] = (df['crop1_intercrop_yield'] / df['crop1_monoculture_yield']) + \
#                         (df['crop2_intercrop_yield'] / df['crop2_monoculture_yield'])
#             df['TOI'] = (df['crop1_intercrop_yield'] + df['crop2_intercrop_yield']) / \
#                         df[['crop1_monoculture_yield', 'crop2_monoculture_yield']].max(axis=1)
#             df['RYT'] = df['LER']  # Same as LER for two crops
#             df['pLER_crop1'] = df['crop1_intercrop_yield'] / df['crop1_monoculture_yield']
#             df['pLER_crop2'] = df['crop2_intercrop_yield'] / df['crop2_monoculture_yield']
#             print("  ✓ Metrics calculated")
#
#         return df
#     except FileNotFoundError:
#         print(f"✗ Error: File '{input_file}' not found!")
#         print("  Please ensure the CSV file is in the same directory as this script.")
#         return None
#     except Exception as e:
#         print(f"✗ Error loading data: {str(e)}")
#         return None
#
#
# # ============================================================================
# # PART 2: DATA ANALYSIS
# # ============================================================================
#
# def analyze_intercropping_data(df):
#     """
#     Perform comprehensive analysis on intercropping data
#     """
#     print("\n" + "=" * 70)
#     print("INTERCROPPING ANALYSIS RESULTS")
#     print("=" * 70)
#
#     # Basic statistics
#     print("\n1. OVERALL STATISTICS")
#     print("-" * 70)
#     metrics = ['LER', 'TOI', 'RYT', 'pLER_crop1', 'pLER_crop2']
#     print(df[metrics].describe().round(3))
#
#     # Advantage analysis
#     print("\n2. INTERCROPPING ADVANTAGE ANALYSIS")
#     print("-" * 70)
#     ler_advantage = (df['LER'] > 1).sum()
#     ler_disadvantage = (df['LER'] < 1).sum()
#     ler_neutral = (df['LER'] == 1).sum()
#
#     print(f"Experiments with LER > 1 (Advantage): {ler_advantage} ({ler_advantage / len(df) * 100:.1f}%)")
#     print(f"Experiments with LER < 1 (Disadvantage): {ler_disadvantage} ({ler_disadvantage / len(df) * 100:.1f}%)")
#     print(f"Experiments with LER = 1 (Neutral): {ler_neutral} ({ler_neutral / len(df) * 100:.1f}%)")
#     print(f"\nAverage LER: {df['LER'].mean():.3f}")
#     print(f"Average yield advantage: {(df['LER'].mean() - 1) * 100:.1f}%")
#
#     # Crop pair analysis
#     print("\n3. ANALYSIS BY CROP COMBINATION")
#     print("-" * 70)
#     crop_pair_stats = df.groupby(['crop1', 'crop2']).agg({
#         'LER': ['mean', 'std', 'min', 'max'],
#         'TOI': 'mean',
#         'RYT': 'mean'
#     }).round(3)
#     print(crop_pair_stats)
#
#     # Site analysis
#     print("\n4. ANALYSIS BY SITE")
#     print("-" * 70)
#     site_stats = df.groupby('site')['LER'].agg(['mean', 'std', 'count']).round(3)
#     print(site_stats)
#
#     # Season analysis
#     print("\n5. ANALYSIS BY SEASON")
#     print("-" * 70)
#     season_stats = df.groupby('season')['LER'].agg(['mean', 'std', 'count']).round(3)
#     print(season_stats)
#
#     # Irrigation analysis
#     print("\n6. ANALYSIS BY IRRIGATION")
#     print("-" * 70)
#     irrigation_stats = df.groupby('irrigation')['LER'].agg(['mean', 'std', 'count']).round(3)
#     print(irrigation_stats)
#
#     return df
#
#
# # ============================================================================
# # PART 3: VISUALIZATION
# # ============================================================================
#
# def create_visualizations(df):
#     """
#     Create comprehensive visualizations for intercropping analysis
#     """
#     print("\n" + "=" * 70)
#     print("GENERATING VISUALIZATIONS")
#     print("=" * 70)
#
#     # Create a figure with multiple subplots
#     fig = plt.figure(figsize=(20, 15))
#
#     # 1. LER Distribution
#     ax1 = plt.subplot(3, 3, 1)
#     df['LER'].hist(bins=30, edgecolor='black', alpha=0.7)
#     plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label='LER = 1')
#     plt.xlabel('Land Equivalent Ratio (LER)')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of LER Values')
#     plt.legend()
#
#     # 2. LER by Crop Combination
#     ax2 = plt.subplot(3, 3, 2)
#     df['crop_pair'] = df['crop1'] + ' + ' + df['crop2']
#     crop_ler = df.groupby('crop_pair')['LER'].mean().sort_values()
#     crop_ler.plot(kind='barh', color='steelblue')
#     plt.axvline(x=1, color='red', linestyle='--', linewidth=2)
#     plt.xlabel('Mean LER')
#     plt.title('Average LER by Crop Combination')
#     plt.tight_layout()
#
#     # 3. Scatter: LER vs TOI
#     ax3 = plt.subplot(3, 3, 3)
#     plt.scatter(df['LER'], df['TOI'], alpha=0.6, c=df['LER'], cmap='viridis')
#     plt.axhline(y=1, color='red', linestyle='--', alpha=0.5)
#     plt.axvline(x=1, color='red', linestyle='--', alpha=0.5)
#     plt.xlabel('LER')
#     plt.ylabel('TOI')
#     plt.title('LER vs TOI Relationship')
#     plt.colorbar(label='LER Value')
#
#     # 4. Partial LER Comparison
#     ax4 = plt.subplot(3, 3, 4)
#     pler_data = pd.DataFrame({
#         'Crop 1': df['pLER_crop1'],
#         'Crop 2': df['pLER_crop2']
#     })
#     pler_data.boxplot()
#     plt.axhline(y=0.5, color='red', linestyle='--', label='Equal contribution')
#     plt.ylabel('Partial LER')
#     plt.title('Distribution of Partial LER Values')
#     plt.legend()
#
#     # 5. LER by Season
#     ax5 = plt.subplot(3, 3, 5)
#     season_data = df.groupby('season')['LER'].apply(list)
#     positions = range(1, len(season_data) + 1)
#     plt.boxplot(season_data.values, labels=season_data.index)
#     plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
#     plt.ylabel('LER')
#     plt.title('LER Distribution by Season')
#     plt.xticks(rotation=45)
#
#     # 6. LER by Site
#     ax6 = plt.subplot(3, 3, 6)
#     site_data = df.groupby('site')['LER'].apply(list)
#     plt.boxplot(site_data.values, labels=site_data.index)
#     plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
#     plt.ylabel('LER')
#     plt.title('LER Distribution by Site')
#
#     # 7. Yield Comparison: Monoculture vs Intercrop
#     ax7 = plt.subplot(3, 3, 7)
#     width = 0.35
#     x = np.arange(2)
#     mono_yields = [df['crop1_monoculture_yield'].mean(), df['crop2_monoculture_yield'].mean()]
#     inter_yields = [df['crop1_intercrop_yield'].mean(), df['crop2_intercrop_yield'].mean()]
#
#     plt.bar(x - width / 2, mono_yields, width, label='Monoculture', alpha=0.8)
#     plt.bar(x + width / 2, inter_yields, width, label='Intercrop', alpha=0.8)
#     plt.xlabel('Crop')
#     plt.ylabel('Yield (kg/ha)')
#     plt.title('Average Yields: Monoculture vs Intercrop')
#     plt.xticks(x, ['Crop 1', 'Crop 2'])
#     plt.legend()
#
#     # 8. Correlation Heatmap
#     ax8 = plt.subplot(3, 3, 8)
#     corr_cols = ['LER', 'TOI', 'RYT', 'pLER_crop1', 'pLER_crop2',
#                  'crop1_intercrop_yield', 'crop2_intercrop_yield']
#     corr_matrix = df[corr_cols].corr()
#     sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
#                 square=True, linewidths=1, cbar_kws={"shrink": 0.8})
#     plt.title('Correlation Matrix of Metrics')
#     plt.xticks(rotation=45, ha='right')
#     plt.yticks(rotation=0)
#
#     # 9. LER by Irrigation Type
#     ax9 = plt.subplot(3, 3, 9)
#     irrigation_data = df.groupby('irrigation')['LER'].apply(list)
#     plt.boxplot(irrigation_data.values, labels=irrigation_data.index)
#     plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
#     plt.ylabel('LER')
#     plt.title('LER Distribution by Irrigation Type')
#
#     plt.tight_layout()
#     plt.savefig('intercropping_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
#     print("✓ Comprehensive visualization saved as 'intercropping_analysis_comprehensive.png'")
#
#     # Additional detailed plot for crop pairs
#     fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
#     fig2.suptitle('Detailed Analysis by Crop Combination', fontsize=16, y=1.02)
#
#     crop_pairs = df['crop_pair'].unique()
#     for idx, (ax, crop_pair) in enumerate(zip(axes.flat, crop_pairs)):
#         subset = df[df['crop_pair'] == crop_pair]
#
#         # Scatter plot with color coding
#         scatter = ax.scatter(subset['pLER_crop1'], subset['pLER_crop2'],
#                              c=subset['LER'], cmap='RdYlGn',
#                              s=100, alpha=0.6, edgecolors='black')
#         ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
#         ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
#         ax.set_xlabel('pLER Crop 1')
#         ax.set_ylabel('pLER Crop 2')
#         ax.set_title(f'{crop_pair}\n(Mean LER: {subset["LER"].mean():.2f})')
#         ax.grid(True, alpha=0.3)
#
#         plt.colorbar(scatter, ax=ax, label='LER')
#
#     # Remove extra subplot if odd number of crop pairs
#     if len(crop_pairs) < 6:
#         for idx in range(len(crop_pairs), 6):
#             fig2.delaxes(axes.flat[idx])
#
#     plt.tight_layout()
#     plt.savefig('intercropping_by_crop_pairs.png', dpi=300, bbox_inches='tight')
#     print("✓ Crop pair analysis saved as 'intercropping_by_crop_pairs.png'")
#
#     plt.show()
#
#
# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================
#
# def main():
#     """
#     Main function to run the complete analysis pipeline
#     """
#     print("\n" + "=" * 70)
#     print("INTERCROPPING ANALYSIS SYSTEM")
#     print("=" * 70)
#     print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#
#     # Step 1: Load data
#     print("\n[STEP 1] Loading intercropping data from CSV...")
#     df = load_intercropping_data('intercropping_data.csv')
#
#     if df is None:
#         print("\n✗ Analysis aborted due to data loading error.")
#         return None
#
#     # Step 2: Analyze data
#     print("\n[STEP 2] Performing statistical analysis...")
#     df = analyze_intercropping_data(df)
#
#     # Step 3: Create visualizations
#     print("\n[STEP 3] Creating visualizations...")
#     create_visualizations(df)
#
#     print("\n" + "=" * 70)
#     print("ANALYSIS COMPLETE!")
#     print("=" * 70)
#     print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print("\nGenerated files:")
#     print("  1. intercropping_analysis_comprehensive.png - Main visualizations")
#     print("  2. intercropping_by_crop_pairs.png - Detailed crop pair analysis")
#
#     return df
#
#
# # Run the analysis
# if __name__ == "__main__":
#     df_results = main()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

def load_intercropping_data(input_file='intercropping_data.csv'):
    """
    Load intercropping data from CSV file
    """
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Data loaded successfully from '{input_file}'")
        print(f"  Total records: {len(df)}")
        print(f"  Columns: {', '.join(df.columns)}")

        # Calculate metrics if not already present
        if 'LER' not in df.columns:
            print("\n  Computing LER, TOI, RYT, and pLER metrics...")
            df['LER'] = (df['crop1_intercrop_yield'] / df['crop1_monoculture_yield']) + \
                        (df['crop2_intercrop_yield'] / df['crop2_monoculture_yield'])
            df['TOI'] = (df['crop1_intercrop_yield'] + df['crop2_intercrop_yield']) / \
                        df[['crop1_monoculture_yield', 'crop2_monoculture_yield']].max(axis=1)
            df['RYT'] = df['LER']  # Same as LER for two crops
            df['pLER_crop1'] = df['crop1_intercrop_yield'] / df['crop1_monoculture_yield']
            df['pLER_crop2'] = df['crop2_intercrop_yield'] / df['crop2_monoculture_yield']
            print("  ✓ Metrics calculated")

        return df
    except FileNotFoundError:
        print(f"✗ Error: File '{input_file}' not found!")
        print("  Please ensure the CSV file is in the same directory as this script.")
        return None
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        return None


# ============================================================================
# PART 2: DATA ANALYSIS
# ============================================================================

def export_tables_to_files(tables):
    """
    Export all analysis tables to CSV and Excel files
    """
    print("\n" + "=" * 70)
    print("EXPORTING TABLES")
    print("=" * 70)

    # Create output directory if it doesn't exist
    import os
    output_dir = 'analysis_tables'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Export each table as CSV
    for table_name, table_df in tables.items():
        csv_filename = f"{output_dir}/{table_name}.csv"
        table_df.to_csv(csv_filename)
        print(f"✓ Exported: {csv_filename}")

    # Export all tables to a single Excel file with multiple sheets
    excel_filename = f"{output_dir}/all_analysis_tables.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        for table_name, table_df in tables.items():
            # Truncate sheet name if too long (Excel limit is 31 characters)
            sheet_name = table_name[:31]
            table_df.to_excel(writer, sheet_name=sheet_name)
    print(f"✓ Exported all tables to: {excel_filename}")

    # Create a summary report in text format
    report_filename = f"{output_dir}/analysis_summary_report.txt"
    with open(report_filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("INTERCROPPING ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")

        for table_name, table_df in tables.items():
            f.write(f"\n{table_name.upper().replace('_', ' ')}\n")
            f.write("-" * 70 + "\n")
            f.write(table_df.to_string())
            f.write("\n\n")

    print(f"✓ Exported summary report: {report_filename}")
    print(f"\n  All tables saved in '{output_dir}/' directory")


def analyze_intercropping_data(df):
    """
    Perform comprehensive analysis on intercropping data and export tables
    """
    print("\n" + "=" * 70)
    print("INTERCROPPING ANALYSIS RESULTS")
    print("=" * 70)

    # Create a dictionary to store all tables
    tables = {}

    # Basic statistics
    print("\n1. OVERALL STATISTICS")
    print("-" * 70)
    metrics = ['LER', 'TOI', 'RYT', 'pLER_crop1', 'pLER_crop2']
    overall_stats = df[metrics].describe().round(3)
    print(overall_stats)
    tables['overall_statistics'] = overall_stats

    # Advantage analysis
    print("\n2. INTERCROPPING ADVANTAGE ANALYSIS")
    print("-" * 70)
    ler_advantage = (df['LER'] > 1).sum()
    ler_disadvantage = (df['LER'] < 1).sum()
    ler_neutral = (df['LER'] == 1).sum()

    print(f"Experiments with LER > 1 (Advantage): {ler_advantage} ({ler_advantage / len(df) * 100:.1f}%)")
    print(f"Experiments with LER < 1 (Disadvantage): {ler_disadvantage} ({ler_disadvantage / len(df) * 100:.1f}%)")
    print(f"Experiments with LER = 1 (Neutral): {ler_neutral} ({ler_neutral / len(df) * 100:.1f}%)")
    print(f"\nAverage LER: {df['LER'].mean():.3f}")
    print(f"Average yield advantage: {(df['LER'].mean() - 1) * 100:.1f}%")

    advantage_summary = pd.DataFrame({
        'Category': ['LER > 1 (Advantage)', 'LER < 1 (Disadvantage)', 'LER = 1 (Neutral)'],
        'Count': [ler_advantage, ler_disadvantage, ler_neutral],
        'Percentage': [ler_advantage / len(df) * 100, ler_disadvantage / len(df) * 100, ler_neutral / len(df) * 100],
        'Mean_LER': [df[df['LER'] > 1]['LER'].mean() if ler_advantage > 0 else 0,
                     df[df['LER'] < 1]['LER'].mean() if ler_disadvantage > 0 else 0,
                     1.0]
    }).round(3)
    tables['advantage_analysis'] = advantage_summary

    # Crop pair analysis
    print("\n3. ANALYSIS BY CROP COMBINATION")
    print("-" * 70)
    crop_pair_stats = df.groupby(['crop1', 'crop2']).agg({
        'LER': ['mean', 'std', 'min', 'max', 'count'],
        'TOI': ['mean', 'std'],
        'RYT': ['mean', 'std']
    }).round(3)
    crop_pair_stats.columns = ['_'.join(col).strip() for col in crop_pair_stats.columns.values]
    print(crop_pair_stats)
    tables['crop_combination_analysis'] = crop_pair_stats

    # Site analysis
    print("\n4. ANALYSIS BY SITE")
    print("-" * 70)
    site_stats = df.groupby('site').agg({
        'LER': ['mean', 'std', 'min', 'max', 'count'],
        'TOI': 'mean',
        'RYT': 'mean'
    }).round(3)
    site_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                          for col in site_stats.columns.values]
    print(site_stats)
    tables['site_analysis'] = site_stats

    # Season analysis
    print("\n5. ANALYSIS BY SEASON")
    print("-" * 70)
    season_stats = df.groupby('season').agg({
        'LER': ['mean', 'std', 'min', 'max', 'count'],
        'TOI': 'mean',
        'RYT': 'mean'
    }).round(3)
    season_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                            for col in season_stats.columns.values]
    print(season_stats)
    tables['season_analysis'] = season_stats

    # Irrigation analysis
    print("\n6. ANALYSIS BY IRRIGATION")
    print("-" * 70)
    irrigation_stats = df.groupby('irrigation').agg({
        'LER': ['mean', 'std', 'min', 'max', 'count'],
        'TOI': 'mean',
        'RYT': 'mean'
    }).round(3)
    irrigation_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                                for col in irrigation_stats.columns.values]
    print(irrigation_stats)
    tables['irrigation_analysis'] = irrigation_stats

    # Correlation matrix
    print("\n7. CORRELATION MATRIX")
    print("-" * 70)
    corr_cols = ['LER', 'TOI', 'RYT', 'pLER_crop1', 'pLER_crop2',
                 'crop1_intercrop_yield', 'crop2_intercrop_yield']
    correlation_matrix = df[corr_cols].corr().round(3)
    print(correlation_matrix)
    tables['correlation_matrix'] = correlation_matrix

    # Yield comparison summary
    print("\n8. YIELD COMPARISON SUMMARY")
    print("-" * 70)
    yield_summary = pd.DataFrame({
        'Metric': ['Crop 1 Monoculture (kg/ha)', 'Crop 1 Intercrop (kg/ha)',
                   'Crop 2 Monoculture (kg/ha)', 'Crop 2 Intercrop (kg/ha)',
                   'Total Monoculture (kg/ha)', 'Total Intercrop (kg/ha)'],
        'Mean': [df['crop1_monoculture_yield'].mean(), df['crop1_intercrop_yield'].mean(),
                 df['crop2_monoculture_yield'].mean(), df['crop2_intercrop_yield'].mean(),
                 df['crop1_monoculture_yield'].mean() + df['crop2_monoculture_yield'].mean(),
                 df['crop1_intercrop_yield'].mean() + df['crop2_intercrop_yield'].mean()],
        'Std': [df['crop1_monoculture_yield'].std(), df['crop1_intercrop_yield'].std(),
                df['crop2_monoculture_yield'].std(), df['crop2_intercrop_yield'].std(),
                0, 0],
        'Min': [df['crop1_monoculture_yield'].min(), df['crop1_intercrop_yield'].min(),
                df['crop2_monoculture_yield'].min(), df['crop2_intercrop_yield'].min(),
                0, 0],
        'Max': [df['crop1_monoculture_yield'].max(), df['crop1_intercrop_yield'].max(),
                df['crop2_monoculture_yield'].max(), df['crop2_intercrop_yield'].max(),
                0, 0]
    }).round(2)
    print(yield_summary)
    tables['yield_comparison'] = yield_summary

    # Export all tables
    export_tables_to_files(tables)

    return df


# ============================================================================
# PART 3: VISUALIZATION
# ============================================================================

def create_visualizations(df):
    """
    Create comprehensive visualizations for intercropping analysis
    """
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. LER Distribution
    ax1 = plt.subplot(3, 3, 1)
    df['LER'].hist(bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label='LER = 1')
    plt.xlabel('Land Equivalent Ratio (LER)')
    plt.ylabel('Frequency')
    plt.title('Distribution of LER Values')
    plt.legend()

    # 2. LER by Crop Combination
    ax2 = plt.subplot(3, 3, 2)
    df['crop_pair'] = df['crop1'] + ' + ' + df['crop2']
    crop_ler = df.groupby('crop_pair')['LER'].mean().sort_values()
    crop_ler.plot(kind='barh', color='steelblue')
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Mean LER')
    plt.title('Average LER by Crop Combination')
    plt.tight_layout()

    # 3. Scatter: LER vs TOI
    ax3 = plt.subplot(3, 3, 3)
    plt.scatter(df['LER'], df['TOI'], alpha=0.6, c=df['LER'], cmap='viridis')
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('LER')
    plt.ylabel('TOI')
    plt.title('LER vs TOI Relationship')
    plt.colorbar(label='LER Value')

    # 4. Partial LER Comparison
    ax4 = plt.subplot(3, 3, 4)
    pler_data = pd.DataFrame({
        'Crop 1': df['pLER_crop1'],
        'Crop 2': df['pLER_crop2']
    })
    pler_data.boxplot()
    plt.axhline(y=0.5, color='red', linestyle='--', label='Equal contribution')
    plt.ylabel('Partial LER')
    plt.title('Distribution of Partial LER Values')
    plt.legend()

    # 5. LER by Season
    ax5 = plt.subplot(3, 3, 5)
    season_data = df.groupby('season')['LER'].apply(list)
    positions = range(1, len(season_data) + 1)
    plt.boxplot(season_data.values, labels=season_data.index)
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.ylabel('LER')
    plt.title('LER Distribution by Season')
    plt.xticks(rotation=45)

    # 6. LER by Site
    ax6 = plt.subplot(3, 3, 6)
    site_data = df.groupby('site')['LER'].apply(list)
    plt.boxplot(site_data.values, labels=site_data.index)
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.ylabel('LER')
    plt.title('LER Distribution by Site')

    # 7. Yield Comparison: Monoculture vs Intercrop
    ax7 = plt.subplot(3, 3, 7)
    width = 0.35
    x = np.arange(2)
    mono_yields = [df['crop1_monoculture_yield'].mean(), df['crop2_monoculture_yield'].mean()]
    inter_yields = [df['crop1_intercrop_yield'].mean(), df['crop2_intercrop_yield'].mean()]

    plt.bar(x - width / 2, mono_yields, width, label='Monoculture', alpha=0.8)
    plt.bar(x + width / 2, inter_yields, width, label='Intercrop', alpha=0.8)
    plt.xlabel('Crop')
    plt.ylabel('Yield (kg/ha)')
    plt.title('Average Yields: Monoculture vs Intercrop')
    plt.xticks(x, ['Crop 1', 'Crop 2'])
    plt.legend()

    # 8. Correlation Heatmap
    ax8 = plt.subplot(3, 3, 8)
    corr_cols = ['LER', 'TOI', 'RYT', 'pLER_crop1', 'pLER_crop2',
                 'crop1_intercrop_yield', 'crop2_intercrop_yield']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # 9. LER by Irrigation Type
    ax9 = plt.subplot(3, 3, 9)
    irrigation_data = df.groupby('irrigation')['LER'].apply(list)
    plt.boxplot(irrigation_data.values, labels=irrigation_data.index)
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.ylabel('LER')
    plt.title('LER Distribution by Irrigation Type')

    plt.tight_layout()
    plt.savefig('intercropping_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    print("✓ Comprehensive visualization saved as 'intercropping_analysis_comprehensive.png'")

    # Additional detailed plot for crop pairs
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Detailed Analysis by Crop Combination', fontsize=16, y=1.02)

    crop_pairs = df['crop_pair'].unique()
    for idx, (ax, crop_pair) in enumerate(zip(axes.flat, crop_pairs)):
        subset = df[df['crop_pair'] == crop_pair]

        # Scatter plot with color coding
        scatter = ax.scatter(subset['pLER_crop1'], subset['pLER_crop2'],
                             c=subset['LER'], cmap='RdYlGn',
                             s=100, alpha=0.6, edgecolors='black')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('pLER Crop 1')
        ax.set_ylabel('pLER Crop 2')
        ax.set_title(f'{crop_pair}\n(Mean LER: {subset["LER"].mean():.2f})')
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label='LER')

    # Remove extra subplot if odd number of crop pairs
    if len(crop_pairs) < 6:
        for idx in range(len(crop_pairs), 6):
            fig2.delaxes(axes.flat[idx])

    plt.tight_layout()
    plt.savefig('intercropping_by_crop_pairs.png', dpi=300, bbox_inches='tight')
    print("✓ Crop pair analysis saved as 'intercropping_by_crop_pairs.png'")

    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run the complete analysis pipeline
    """
    print("\n" + "=" * 70)
    print("INTERCROPPING ANALYSIS SYSTEM")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Load data
    print("\n[STEP 1] Loading intercropping data from CSV...")
    df = load_intercropping_data('intercropping_data.csv')

    if df is None:
        print("\n✗ Analysis aborted due to data loading error.")
        return None

    # Step 2: Analyze data
    print("\n[STEP 2] Performing statistical analysis...")
    df = analyze_intercropping_data(df)

    # Step 3: Create visualizations
    print("\n[STEP 3] Creating visualizations...")
    create_visualizations(df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files:")
    print("  TABLES (in 'analysis_tables/' directory):")
    print("    - overall_statistics.csv")
    print("    - advantage_analysis.csv")
    print("    - crop_combination_analysis.csv")
    print("    - site_analysis.csv")
    print("    - season_analysis.csv")
    print("    - irrigation_analysis.csv")
    print("    - correlation_matrix.csv")
    print("    - yield_comparison.csv")
    print("    - all_analysis_tables.xlsx (all tables in one Excel file)")
    print("    - analysis_summary_report.txt (formatted text report)")
    print("\n  VISUALIZATIONS:")
    print("    - intercropping_analysis_comprehensive.png")
    print("    - intercropping_by_crop_pairs.png")

    return df


# Run the analysis
if __name__ == "__main__":
    df_results = main()
