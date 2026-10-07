import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(data_path='data/train.csv', output_dir='static/charts'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    # 1. Price Distribution Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(df['SalePrice'], kde=True, color='skyblue')
    plt.title('Property Sale Price Distribution')
    plt.xlabel('Sale Price ($)')
    plt.savefig(f'{output_dir}/price_distribution.png')
    plt.close()

    # 2. Living Area vs Sale Price
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', hue='OverallQual', palette='viridis')
    plt.title('Above Grade Living Area vs Sale Price')
    plt.xlabel('Living Area (sq ft)')
    plt.ylabel('Sale Price ($)')
    plt.savefig(f'{output_dir}/area_vs_price.png')
    plt.close()

    # 3. Neighborhood Price Comparison
    plt.figure(figsize=(12, 6))
    top_neighborhoods = df.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False).head(10)
    sns.barplot(x=top_neighborhoods.values, y=top_neighborhoods.index, palette='magma')
    plt.title('Top 10 Most Expensive Neighborhoods (Median Price)')
    plt.xlabel('Median Price ($)')
    plt.savefig(f'{output_dir}/top_neighborhoods.png')
    plt.close()

    print("EDA Visualizations created successfully in static/charts/")

if __name__ == '__main__':
    run_eda()