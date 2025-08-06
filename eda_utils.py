import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional, List, Dict, Any

def get_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Get missing values information safely"""
    try:
        missing_data = df.isnull().sum()
        missing_percent = (df.isnull().sum() / len(df)) * 100
        
        return {
            "missing_counts": missing_data.to_dict(),
            "missing_percentages": missing_percent.to_dict(),
            "total_missing": int(missing_data.sum()),
            "columns_with_missing": missing_data[missing_data > 0].index.tolist()
        }
    except Exception as e:
        print(f"Error in get_missing_values: {str(e)}")
        return {"error": str(e)}

def plot_distributions(df: pd.DataFrame, churn_col: str) -> List[str]:
    """Plot distributions for numeric columns only"""
    try:
        plot_paths = []
        
        # Get only numeric columns (excluding the churn column)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if churn_col in numeric_cols:
            numeric_cols.remove(churn_col)
        
        # Get categorical columns (excluding the churn column)  
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if churn_col in categorical_cols:
            categorical_cols.remove(churn_col)
            
        # Plot numeric distributions
        if numeric_cols:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
                
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
                
            plt.tight_layout()
            numeric_path = "plots/numeric_distributions.png"
            plt.savefig(numeric_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(numeric_path)
        
        # Plot categorical distributions  
        if categorical_cols:
            n_cols = min(2, len(categorical_cols))
            n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
                
            for i, col in enumerate(categorical_cols):
                if i < len(axes):
                    # Limit to top 10 categories to avoid overcrowding
                    value_counts = df[col].value_counts().head(10)
                    value_counts.plot(kind='bar', ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Count')
                    axes[i].tick_params(axis='x', rotation=45)
            
            # Hide empty subplots
            for i in range(len(categorical_cols), len(axes)):
                axes[i].set_visible(False)
                
            plt.tight_layout()
            categorical_path = "plots/categorical_distributions.png"
            plt.savefig(categorical_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(categorical_path)
            
        return plot_paths
        
    except Exception as e:
        print(f"Error in plot_distributions: {str(e)}")
        return []

def plot_correlation(df: pd.DataFrame) -> Optional[str]:
    """Plot correlation matrix for numeric columns only"""
    try:
        # Get only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            print("Not enough numeric columns for correlation matrix")
            return None
            
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create correlation plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix of Numeric Features')
        plt.tight_layout()
        
        plot_path = "plots/correlation_matrix.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error in plot_correlation: {str(e)}")
        return None

def plot_churn_by_feature(df: pd.DataFrame, churn_col: str) -> List[str]:
    """Plot churn rates by different features"""
    try:
        plot_paths = []
        
        # Get numeric columns (excluding churn column)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if churn_col in numeric_cols:
            numeric_cols.remove(churn_col)
            
        # Get categorical columns (excluding churn column)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if churn_col in categorical_cols:
            categorical_cols.remove(churn_col)
        
        # For numeric features - create binned analysis
        if numeric_cols:
            n_cols = min(2, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
                
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    # Create bins for numeric feature
                    df['binned'] = pd.cut(df[col], bins=5, duplicates='drop')
                    churn_by_bin = df.groupby('binned')[churn_col].value_counts(normalize=True).unstack()
                    
                    if churn_by_bin is not None and not churn_by_bin.empty:
                        churn_by_bin.plot(kind='bar', ax=axes[i], stacked=True)
                        axes[i].set_title(f'Churn Rate by {col}')
                        axes[i].set_xlabel(f'{col} (binned)')
                        axes[i].set_ylabel('Proportion')
                        axes[i].tick_params(axis='x', rotation=45)
                        axes[i].legend(title=churn_col)
            
            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
                
            plt.tight_layout()
            numeric_churn_path = "plots/churn_by_numeric_features.png"
            plt.savefig(numeric_churn_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(numeric_churn_path)
            
            # Clean up temporary column
            if 'binned' in df.columns:
                df.drop('binned', axis=1, inplace=True)
        
        # For categorical features
        if categorical_cols:
            n_cols = min(2, len(categorical_cols))
            n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
                
            for i, col in enumerate(categorical_cols):
                if i < len(axes):
                    # Calculate churn rate by category
                    churn_by_cat = df.groupby(col)[churn_col].value_counts(normalize=True).unstack()
                    
                    if churn_by_cat is not None and not churn_by_cat.empty:
                        churn_by_cat.plot(kind='bar', ax=axes[i])
                        axes[i].set_title(f'Churn Rate by {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Proportion')
                        axes[i].tick_params(axis='x', rotation=45)
                        axes[i].legend(title=churn_col)
            
            # Hide empty subplots
            for i in range(len(categorical_cols), len(axes)):
                axes[i].set_visible(False)
                
            plt.tight_layout()
            categorical_churn_path = "plots/churn_by_categorical_features.png"
            plt.savefig(categorical_churn_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(categorical_churn_path)
            
        return plot_paths
        
    except Exception as e:
        print(f"Error in plot_churn_by_feature: {str(e)}")
        return []

def plot_timeline(df: pd.DataFrame, time_col: str, churn_col: str) -> Optional[str]:
    """Plot timeline analysis if time column exists"""
    try:
        if not time_col or time_col not in df.columns:
            return None
            
        # Ensure time column is numeric
        if not pd.api.types.is_numeric_dtype(df[time_col]):
            print(f"Time column {time_col} is not numeric")
            return None
            
        plt.figure(figsize=(12, 6))
        
        # Create time bins
        df['time_binned'] = pd.cut(df[time_col], bins=10, duplicates='drop')
        
        # Calculate churn rate over time
        churn_by_time = df.groupby('time_binned')[churn_col].value_counts(normalize=True).unstack()
        
        if churn_by_time is not None and not churn_by_time.empty:
            # Plot the churn rate over time
            if len(churn_by_time.columns) > 1:
                churn_by_time.iloc[:, -1].plot(kind='line', marker='o')
                plt.title(f'Churn Rate Over Time ({time_col})')
                plt.xlabel(f'{time_col} (binned)')
                plt.ylabel('Churn Rate')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = "plots/churn_timeline.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Clean up temporary column
                df.drop('time_binned', axis=1, inplace=True)
                
                return plot_path
        
        plt.close()
        return None
        
    except Exception as e:
        print(f"Error in plot_timeline: {str(e)}")
        if 'time_binned' in df.columns:
            df.drop('time_binned', axis=1, inplace=True)
        return None

def generate_pdf_report(plot_paths: List[str]) -> str:
    """Generate a PDF report with all plots"""
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        
        pdf_path = "plots/eda_report.pdf"
        
        with PdfPages(pdf_path) as pdf:
            for plot_path in plot_paths:
                if plot_path and os.path.exists(plot_path.lstrip('/')):
                    img = plt.imread(plot_path.lstrip('/'))
                    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 size
                    ax.imshow(img)
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                    
        return pdf_path
        
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        return ""