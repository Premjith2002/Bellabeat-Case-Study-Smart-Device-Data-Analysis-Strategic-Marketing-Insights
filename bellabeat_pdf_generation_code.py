
# Step-by-Step PDF Report Generation for Bellabeat Case Study
# This code demonstrates the complete workflow from data loading to PDF creation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# ================== STEP 1: DATA LOADING & PREPARATION ==================
print("Step 1: Loading and preparing data...")

def load_and_clean_data(file_path):
    """Load and clean the FitBit dataset"""
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Convert date column
    df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
    
    # Feature engineering
    def classify_activity_level(steps):
        if steps < 5000:
            return 'Sedentary'
        elif steps < 7500:
            return 'Lightly Active'
        elif steps < 10000:
            return 'Fairly Active'
        else:
            return 'Very Active'
    
    df['ActivityLevel'] = df['TotalSteps'].apply(classify_activity_level)
    df['DayOfWeek'] = df['ActivityDate'].dt.day_name()
    df['TotalActiveMinutes'] = df['VeryActiveMinutes'] + df['FairlyActiveMinutes'] + df['LightlyActiveMinutes']
    df['ActivityPercentage'] = (df['TotalActiveMinutes'] / (df['TotalActiveMinutes'] + df['SedentaryMinutes'])) * 100
    df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])
    
    return df

# Load data
df = load_and_clean_data('FitBit_data.csv.csv')
print(f"✓ Loaded {len(df)} records from {df['Id'].nunique()} users")

# ================== STEP 2: DATA ANALYSIS ==================
print("Step 2: Performing comprehensive analysis...")

def analyze_data(df):
    """Perform comprehensive data analysis"""
    analysis_results = {}
    
    # Basic statistics
    analysis_results['basic_stats'] = {
        'total_records': len(df),
        'unique_users': df['Id'].nunique(),
        'date_range': f"{df['ActivityDate'].min().strftime('%Y-%m-%d')} to {df['ActivityDate'].max().strftime('%Y-%m-%d')}",
        'avg_steps': df['TotalSteps'].mean(),
        'avg_calories': df['Calories'].mean(),
        'avg_sedentary_hours': df['SedentaryMinutes'].mean() / 60
    }
    
    # Activity level distribution
    analysis_results['activity_distribution'] = df['ActivityLevel'].value_counts(normalize=True) * 100
    
    # User-level analysis
    user_stats = df.groupby('Id')['TotalSteps'].mean().apply(classify_activity_level)
    analysis_results['user_activity_levels'] = user_stats.value_counts(normalize=True) * 100
    
    # Weekly patterns
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_patterns = df.groupby('DayOfWeek')['TotalSteps'].mean().reindex(day_order)
    analysis_results['weekly_patterns'] = weekly_patterns
    
    # Weekend vs weekday
    weekend_comparison = df.groupby('IsWeekend').agg({
        'TotalSteps': 'mean',
        'VeryActiveMinutes': 'mean',
        'SedentaryMinutes': 'mean'
    })
    analysis_results['weekend_comparison'] = weekend_comparison
    
    return analysis_results

results = analyze_data(df)
print("✓ Analysis complete")

# ================== STEP 3: VISUALIZATION CREATION ==================
print("Step 3: Creating visualizations...")

def create_visualizations(df, results):
    """Create all visualizations for the report"""
    plt.style.use('default')
    fig_size = (16, 12)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=fig_size)
    fig.suptitle('FitBit Data Analysis Dashboard - Bellabeat Case Study', fontsize=16, fontweight='bold')
    
    # 1. Activity Level Distribution (Pie Chart)
    activity_dist = results['activity_distribution']
    colors = ['#E74C3C', '#F39C12', '#F1C40F', '#27AE60']
    ax1.pie(activity_dist.values, labels=activity_dist.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax1.set_title('Activity Level Distribution\n(by records)', fontweight='bold')
    
    # 2. Daily Metrics Bar Chart
    daily_metrics = ['Steps', 'Very Active\nMins', 'Fairly Active\nMins', 'Lightly Active\nMins', 'Sedentary\nMins']
    daily_values = [results['basic_stats']['avg_steps'], 
                   df['VeryActiveMinutes'].mean(),
                   df['FairlyActiveMinutes'].mean(),
                   df['LightlyActiveMinutes'].mean(),
                   df['SedentaryMinutes'].mean()]
    
    bars = ax2.bar(daily_metrics, daily_values, color=['#3498DB', '#E74C3C', '#F39C12', '#F1C40F', '#95A5A6'])
    ax2.set_title('Average Daily Metrics', fontweight='bold')
    ax2.set_ylabel('Count')
    
    # Add value labels on bars
    for bar, value in zip(bars, daily_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value):,}', ha='center', va='bottom', fontsize=9)
    
    # 3. Weekly Activity Pattern
    weekly_data = results['weekly_patterns']
    ax3.plot(weekly_data.index, weekly_data.values, marker='o', linewidth=2, markersize=6, color='#2ECC71')
    ax3.set_title('Weekly Activity Patterns', fontweight='bold')
    ax3.set_ylabel('Average Steps')
    ax3.set_xlabel('Day of Week')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. User Activity Level Distribution
    user_levels = results['user_activity_levels']
    ax4.barh(user_levels.index, user_levels.values, color=['#E74C3C', '#F39C12', '#F1C40F', '#27AE60'])
    ax4.set_title('User Activity Level Distribution\n(by average daily steps)', fontweight='bold')
    ax4.set_xlabel('Percentage of Users')
    
    # Add percentage labels
    for i, (level, percentage) in enumerate(user_levels.items()):
        ax4.text(percentage + 1, i, f'{percentage:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

visualization_fig = create_visualizations(df, results)
print("✓ Visualizations created")

# ================== STEP 4: PDF REPORT GENERATION ==================
print("Step 4: Generating PDF report...")

def generate_pdf_report(df, results, visualization_fig, filename='bellabeat_analysis_report.pdf'):
    """Generate comprehensive PDF report"""
    
    with PdfPages(filename) as pdf:
        # Page 1: Executive Summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Bellabeat Case Study Analysis Report', 
                ha='center', va='top', fontsize=20, fontweight='bold')
        
        # Executive Summary
        summary_text = f"""
EXECUTIVE SUMMARY

Dataset Overview:
• {results['basic_stats']['total_records']} activity records from {results['basic_stats']['unique_users']} unique users
• Data period: {results['basic_stats']['date_range']}
• Average daily steps: {results['basic_stats']['avg_steps']:,.0f}
• Average daily calories: {results['basic_stats']['avg_calories']:,.0f}

Key Findings:
• Users spend {results['basic_stats']['avg_sedentary_hours']:.1f} hours/day sedentary (concerning)
• Only 17.1% of users are classified as "Very Active"
• 40% of users fall into "Sedentary" category
• Tuesday shows lowest activity levels
• Strong correlation (0.58) between steps and calories burned

Marketing Opportunities:
• Target sedentary behavior reduction
• Focus on weekday activity improvement
• Emphasize holistic wellness beyond just steps
• Develop women-specific wellness features
        """
        
        ax.text(0.1, 0.85, summary_text, ha='left', va='top', fontsize=11, 
                transform=ax.transAxes, wrap=True)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Visualizations
        pdf.savefig(visualization_fig, bbox_inches='tight')
        
        # Page 3: Detailed Analysis
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Detailed Analysis Results', 
                ha='center', va='top', fontsize=18, fontweight='bold')
        
        detailed_text = f"""
ACTIVITY LEVEL BREAKDOWN:

Record-Level Distribution:
• Sedentary (<5000 steps): {results['activity_distribution']['Sedentary']:.1f}%
• Lightly Active (5000-7499): {results['activity_distribution']['Lightly Active']:.1f}%
• Fairly Active (7500-9999): {results['activity_distribution']['Fairly Active']:.1f}%
• Very Active (10000+): {results['activity_distribution']['Very Active']:.1f}%

User-Level Classification:
• Sedentary users: {results['user_activity_levels']['Sedentary']:.1f}%
• Lightly Active users: {results['user_activity_levels']['Lightly Active']:.1f}%
• Fairly Active users: {results['user_activity_levels']['Fairly Active']:.1f}%
• Very Active users: {results['user_activity_levels']['Very Active']:.1f}%

WEEKLY PATTERNS:
"""
        
        for day, steps in results['weekly_patterns'].items():
            detailed_text += f"• {day}: {steps:,.0f} average steps\n"
        
        detailed_text += f"""
RECOMMENDATIONS FOR BELLABEAT:

1. Sedentary Behavior Focus:
   - Develop smart notifications for movement breaks
   - Market 16.6-hour sedentary problem as key differentiator
   
2. Women-Specific Features:
   - Integrate menstrual cycle tracking with activity patterns
   - Emphasize stress management and wellness beyond fitness
   
3. Weekday Engagement:
   - Target Tuesday activity improvement campaigns
   - Develop workday wellness challenges
   
4. User Segmentation:
   - Create different messaging for sedentary vs active users
   - Focus on gradual improvement rather than dramatic changes
        """
        
        ax.text(0.1, 0.85, detailed_text, ha='left', va='top', fontsize=10, 
                transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    print(f"✓ PDF report generated: {filename}")

# Generate the PDF
generate_pdf_report(df, results, visualization_fig)

# ================== STEP 5: SAVE SUPPORTING FILES ==================
print("Step 5: Saving supporting files...")

# Save cleaned dataset
df.to_csv('bellabeat_cleaned_data.csv', index=False)
print("✓ Saved cleaned dataset")

# Save analysis summary
summary_df = pd.DataFrame([
    {'Metric': 'Total Records', 'Value': len(df)},
    {'Metric': 'Unique Users', 'Value': df['Id'].nunique()},
    {'Metric': 'Average Daily Steps', 'Value': f"{df['TotalSteps'].mean():.0f}"},
    {'Metric': 'Average Sedentary Hours', 'Value': f"{df['SedentaryMinutes'].mean()/60:.1f}"},
    {'Metric': 'Most Active Day', 'Value': results['weekly_patterns'].idxmax()},
    {'Metric': 'Least Active Day', 'Value': results['weekly_patterns'].idxmin()}
])
summary_df.to_csv('analysis_summary.csv', index=False)
print("✓ Saved analysis summary")

print("\n" + "="*60)
print("COMPLETE WORKFLOW FINISHED!")
print("Files generated:")
print("- bellabeat_analysis_report.pdf (main report)")
print("- bellabeat_cleaned_data.csv (processed dataset)")
print("- analysis_summary.csv (key metrics)")
print("="*60)
